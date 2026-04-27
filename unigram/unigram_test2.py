from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
import torch
import torch.nn as nn
import lightning as L

try:
    from unigram.model import MLPLM
    from unigram.dataset import UnigramDataModule
    from unigram.visualizer import DataMgr, Recorder
except ModuleNotFoundError:
    from model import MLPLM
    from dataset import UnigramDataModule
    from visualizer import DataMgr, Recorder

class HyperBridge:
    PROPOSAL_EXP_NAME: str = "exp"
    PROPOSAL_UNIF_NAME: str = "unif"

    @staticmethod
    def sample_chi(ns,dtype=torch.float64):
        nshape = ns.shape
        ns = ns.reshape(-1)
        M = ns.sum().item()
        x = torch.randn(M, device=ns.device, dtype=dtype).square()
        chi2 = torch.segment_reduce(x,'sum',lengths=ns)
        return chi2.sqrt().reshape(nshape)

    @staticmethod
    @torch.no_grad()
    def binary_bridge(ts):
        ns = torch.poisson(ts/8).to(torch.int64)
        ss = ts.sqrt() * HyperBridge.sample_chi(2*ns+3, ts.dtype)
        vs = torch.rand_like(ts)
        ps = torch.acosh(vs.square() + (1-vs.square())*torch.cosh(ss))
        us = torch.rand_like(ts)
        thetas = 2 * torch.atan((-ps).exp() * torch.tan(torch.pi * (us - 0.5)))
        return (ps,thetas)

    # logits        (N,V)     float64 (converts)
    # targets       (N,)      int64
    # rhos          (N,)      float64
    # thetas        (N,)      float64
    @staticmethod
    def binary_bridge_loss(logits, targets, rhos, thetas):
        (N,) = targets.shape
        (N,V) = logits.shape
        device = rhos.device
        assert(rhos.shape == (N,))
        assert(thetas.shape == (N,))
        assert(targets.dtype == torch.int64)
        assert(rhos.dtype == torch.float64)
        assert(thetas.dtype == torch.float64)
        # construct phis
        phis = (
            torch.arange(V, device=device, dtype=torch.float64) + 0.5
        ) * (2 * torch.pi / V)
        # first, we get the horosphere distances
        alphas = thetas[:,None] - phis[None,:]  # angular offsets between z and v
        cos_alphas = alphas.cos()
        sin_alphas = alphas.sin()
        log_two = torch.log(torch.tensor(2.0, device=device, dtype=torch.float64))
        horosphere_dists = log_two - torch.logaddexp((1 - cos_alphas).log() + rhos[:,None], (1 + cos_alphas).log() - rhos[:,None])
        # remake mu and subtract the target
        mu = (horosphere_dists + logits.to(torch.float64)).softmax(-1)
        mu = mu - torch.nn.functional.one_hot(targets,V).to(torch.float64)
        # next, we transform the angles alpha after motion by rho
        betas = torch.atan2(sin_alphas, rhos.cosh()[:,None] * cos_alphas - rhos.sinh()[:,None])
        cos_errors = (betas.cos() * mu).sum(-1)
        sin_errors = (betas.sin() * mu).sum(-1)
        return (cos_errors.square() + sin_errors.square())/2

    @staticmethod
    def binary_nelbo_loss(logits, targets, rhos, thetas, proposal_weight):
        bridge = HyperBridge.binary_bridge_loss(
            logits=logits,
            targets=targets,
            rhos=rhos,
            thetas=thetas,
        )
        return bridge * proposal_weight.to(dtype=bridge.dtype)

    @staticmethod
    def proposal(
        proposal_type: str,
        shape,
        device,
        dtype,
        unif_min: float,
        unif_max: float,
        exp_rate: float,
    ):
        proposal_type = proposal_type.lower()
        interval = float(unif_max - unif_min)
        if interval < 0:
            raise ValueError("proposal requires unif_max >= unif_min")

        if proposal_type == HyperBridge.PROPOSAL_UNIF_NAME:
            ts = unif_min + interval * torch.rand(shape, device=device, dtype=dtype)
            weights = torch.full_like(ts, interval)
            return ts, weights
        elif proposal_type == HyperBridge.PROPOSAL_EXP_NAME:
            if exp_rate <= 0:
                raise ValueError("proposal_exp_rate must be > 0")
            if interval == 0:
                ts = torch.full(shape, unif_min, device=device, dtype=dtype)
                return ts, torch.zeros_like(ts)
            u = torch.rand(shape, device=device, dtype=dtype).clamp(
                min=1e-12,
                max=1 - 1e-12,
            )
            normalizer = 1 - torch.exp(
                torch.tensor(-exp_rate * interval, device=device, dtype=dtype)
            )
            ts = unif_min - torch.log1p(-u * normalizer) / exp_rate
            density = exp_rate * torch.exp(-exp_rate * (ts - unif_min)) / normalizer
            return ts, density.reciprocal()
        else:
            raise NotImplementedError(f"proposal_type={proposal_type} is not implemented.")

    @staticmethod
    def hyper_proposal(
        proposal_type: str,
        shape,
        device,
        dtype,
        dt: float = 0.01,
        T: int = 1000,
        exp_rate: float = 1.0,
    ):
        if dt is None or T is None or dt <= 0.0 or T <= 0:
            raise ValueError("dt and T must be positive for hyper_proposal.")

        total_time = dt * T
        unif_min = float(max(dt, 1e-8))
        unif_max = float(max(total_time, unif_min))

        ts, proposal_weight = HyperBridge.proposal(
            proposal_type=proposal_type,
            shape=shape,
            device=device,
            dtype=dtype,
            unif_min=unif_min,
            unif_max=unif_max,
            exp_rate=exp_rate,
        )
        return ts, proposal_weight
        

class HyperbolicDLM(L.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.bridge = HyperBridge()
        self.recorder = Recorder()
        self._test_step_offset = 0
        self._val_epoch_loss_total = 0.0
        self._val_epoch_weight = 0
        self._test_epoch_loss_total = 0.0
        self._test_epoch_weight = 0
        if int(config.hyper_dim) != 2:
            raise ValueError("unigram_test2.py expects hyper_dim=2 for polar coordinates.")

        self.model = MLPLM(
            vocab_size=config.vocab_size,
            io_dim=config.hyper_dim,
            hidden_size=config.hidden_size,
            depth=config.depth,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=float(self.config.lr))

    def _compute_losses(self, batch: torch.Tensor):
        targets = batch.reshape(-1).to(device=self.device, dtype=torch.long)
        batch_size = targets.shape[0]

        ts, proposal_weight = self.bridge.hyper_proposal(
            proposal_type=self.config.proposal_type,
            shape=(batch_size,),
            device=self.device,
            dtype=torch.float64,
            dt=self.config.hyper_dt,
            T=self.config.hyper_T,
            exp_rate=self.config.proposal_exp_rate,
        )

        rhos, thetas = self.bridge.binary_bridge(ts=ts)
        # Feed the bridge perturbations directly in polar coordinates.
        z = torch.stack([rhos, thetas], dim=-1).to(dtype=torch.float32)
        logits = self.model(z=z, t=ts.to(dtype=torch.float32))
        nelbo = self.bridge.binary_nelbo_loss(
            logits=logits,
            targets=targets,
            rhos=rhos,
            thetas=thetas,
            proposal_weight=proposal_weight,
        )
        ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        return {
            "loss": nelbo,
            "nelbo_loss": nelbo,
            "ce": ce,
            "ts": ts.to(dtype=torch.float32),
            "proposal_weight": proposal_weight.to(dtype=torch.float32),
            "rhos": rhos.to(dtype=torch.float32),
            "thetas": thetas.to(dtype=torch.float32),
        }

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        del batch_idx
        losses = self._compute_losses(batch)
        loss = losses["loss"].mean()
        self.recorder.add("train_loss", step=int(self.global_step) + 1, val=loss)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_nelbo_loss",
            losses["nelbo_loss"].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log("train_ce", losses["ce"].mean(), on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        del batch_idx
        losses = self._compute_losses(batch)
        loss = losses["loss"].mean()
        batch_size = int(batch.reshape(-1).shape[0])
        if not self.trainer.sanity_checking:
            self._val_epoch_loss_total += float(loss.detach().cpu()) * batch_size
            self._val_epoch_weight += batch_size
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val_nelbo_loss", losses["nelbo_loss"].mean(), on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("val_ce", losses["ce"].mean(), on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def on_validation_epoch_start(self):
        self._val_epoch_loss_total = 0.0
        self._val_epoch_weight = 0

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or self._val_epoch_weight == 0:
            return
        mean_val_loss = self._val_epoch_loss_total / self._val_epoch_weight
        self.recorder.add("val_loss", step=int(self.global_step), val=mean_val_loss)

    def on_test_start(self):
        self._test_step_offset = max(
            self.recorder.last_step("train_loss"),
            int(self.global_step),
        )
        self._test_epoch_loss_total = 0.0
        self._test_epoch_weight = 0

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        del batch_idx
        losses = self._compute_losses(batch)
        loss = losses["loss"].mean()
        batch_size = int(batch.reshape(-1).shape[0])
        self._test_epoch_loss_total += float(loss.detach().cpu()) * batch_size
        self._test_epoch_weight += batch_size
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("test_nelbo_loss", losses["nelbo_loss"].mean(), on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("test_ce", losses["ce"].mean(), on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def on_test_epoch_end(self):
        if self._test_epoch_weight == 0:
            return
        mean_test_loss = self._test_epoch_loss_total / self._test_epoch_weight
        self.recorder.add(
            "test_loss",
            step=self._test_step_offset + 1,
            val=mean_test_loss,
        )

@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    defaults = OmegaConf.create(
        {
            "vocab_size": 3,
            "hyper_dim": 2,
            "hidden_size": 128,
            "train_size": 20_000,
            "depth": 3,
            "hyper_T": 1000,
            "hyper_dt": 0.01,
            "val_size": 4_000,
            "batch_size": 256,
            "max_steps": 2_000,
            "proposal_type": "exp",
            "proposal_exp_rate": 1.0,
            "lr": 1e-5,
            "ps": [0.2, 0.2, 0.6],
            "seed": 0,
            "num_workers": 0,
            "folder": "unigram_test2_losses_sp${max_steps}_lr${lr}_pt${proposal_type}_per${proposal_exp_rate}",
            "loss_plot_path": "plot.jpg",
            "loss_data_path": "data.json",
            "loss_plot_ma_window": None,
        }
    )
    cfg = OmegaConf.merge(defaults, cfg)
    print(OmegaConf.to_yaml(cfg))

    L.seed_everything(int(cfg.seed), workers=True)
    datamodule = UnigramDataModule(config=cfg)
    model = HyperbolicDLM(config=cfg)
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_steps=int(cfg.max_steps),
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=datamodule)
    test_metrics = trainer.test(model, datamodule=datamodule, verbose=False)
    data_mgr = DataMgr(cfg.folder)
    saved = data_mgr.save(
        recorder=model.recorder,
        data_file=cfg.loss_data_path,
        fig_file=cfg.loss_plot_path,
        moving_average_window=cfg.loss_plot_ma_window,
    )
    print(f"Saved loss data to: {saved['data_path']}")
    print(f"Saved loss plot to: {saved['figure_path']}")
    if test_metrics:
        print("Test metrics:")
        for name, value in test_metrics[0].items():
            print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
    
