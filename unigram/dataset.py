import lightning as L
import torch
from omegaconf import ListConfig
from torch.utils.data import DataLoader, Dataset


def process_ps(ps):
    ps = torch.as_tensor(ps, dtype=torch.float32).reshape(-1)
    if ps.numel() == 1:
        p = float(ps.item())
        ps = torch.tensor([p, 1.0 - p], dtype=torch.float32)
    if ps.numel() == 0:
        raise ValueError("ps must contain at least one probability.")
    if torch.any(ps < 0):
        raise ValueError("ps must be non-negative.")
    total = float(ps.sum().item())
    if total <= 0.0:
        raise ValueError("ps must sum to a positive value.")

    ret_ps = ps / total
    if not torch.all(ret_ps > 0):
        raise ValueError(f"Every entry of ps should be > 0")
    return ret_ps


def counts_from_ps(size: int, ps: torch.Tensor) -> torch.Tensor:
    expected = float(size) * ps
    counts = expected.floor().to(torch.long)
    remainder = int(size - counts.sum().item())
    if remainder > 0:
        order = (expected - counts.to(expected.dtype)).argsort(descending=True)
        counts[order[:remainder]] += 1
    return counts

class UnigramDataset(Dataset):
    """Exact unigram dataset with sequence length 1."""

    def __init__(self, size: int, ps, seed: int):
        super().__init__()
        ps = process_ps(ps)
        counts = counts_from_ps(size=size, ps=ps)
        tokens = torch.arange(ps.numel(), dtype=torch.long)
        dataset = torch.repeat_interleave(tokens, counts)

        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(dataset.numel(), generator=generator)
        self.tokens = dataset[permutation]

    def __len__(self):
        return self.tokens.numel()

    def __getitem__(self, index: int):
        return self.tokens[index]

class UnigramDataModule(L.LightningDataModule):
    PS_NAIVE = "naive_ps"
    PS_CMPLX = "cmplx_ps"
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        # Resolve config.ps (a named string spec or an explicit list) into a
        # concrete probability list, and pin the vocab size to its length.
        self.ps = self.ps_generator()
        self.vocab_size = len(self.ps)
        print(f"Entropy: {self.entropy(self.ps)}")

    def entropy(self, ps):
        ps = process_ps(ps)
        return -(ps * torch.log(ps)).sum()

    STRING_PS = (PS_NAIVE, PS_CMPLX, "cmplx_ps1", "c1e3_exp1.0", "c1e4_exp1.0", "c1e5_exp1.0")

    @staticmethod
    def _exp_decay_ps(n: int, lam: float = 1.0):
        """Deterministic exponential decay: p_i proportional to exp(-lam * i),
        i = 0..n-1, normalized to sum 1.

        exp(-i) underflows to 0 for i >~ 100 (and 0 trips process_ps's
        strictly-positive check / breaks log(ps) in OptimalModel), so the
        negligible tail is floored to a tiny positive value and renormalized.
        The floor (~1e-30) is far below any token that ever appears in the data,
        so it does not materially change the distribution.
        """
        i = torch.arange(n, dtype=torch.float64)
        p = torch.exp(-lam * i)
        p = p / p.sum()
        p = p.clamp_min(1e-30)
        p = p / p.sum()
        return p.tolist()

    def ps_generator(self):
        ps = self.config.ps
        if isinstance(ps, str):
            if ps == self.PS_NAIVE:
                ret_ps = [0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
            elif ps == self.PS_CMPLX:
                ret_ps = [0.31,0.01,0.2,0.01,0.01,0.3,0.08,0.04,0.03,0.01]
            elif ps == "cmplx_ps1":
                ret_ps = [0.11,0.10,0.10,0.11,0.11,0.10,0.10,0.10,0.09,0.08]
            elif ps == "c1e3_exp1.0":
                ret_ps = self._exp_decay_ps(n=100, lam=1.0)
            elif ps == "c1e4_exp1.0":
                ret_ps = self._exp_decay_ps(n=1000, lam=1.0)
            elif ps == "c1e5_exp1.0":
                ret_ps = self._exp_decay_ps(n=10000, lam=1.0)
            else:
                raise ValueError(
                    f"config.ps={ps!r} is not a supported string spec; "
                    f"expected one of {self.STRING_PS} or an explicit list of probabilities."
                )
        elif isinstance(ps, (list, tuple, ListConfig)):
            ret_ps = [float(x) for x in ps]
        else:
            raise ValueError(
                f"config.ps has unsupported type {type(ps).__name__}; "
                f"expected a list of probabilities or one of the string specs {self.STRING_PS}."
            )
        return ret_ps

    def setup(self, stage: str | None = None):
        del stage
        self.train_dataset = UnigramDataset(
            size=self.config.train_size, ps=self.ps, seed=self.config.seed
        )
        self.val_dataset = UnigramDataset(
            size=self.config.val_size, ps=self.ps, seed=self.config.seed + 1
        )
        self.test_dataset = UnigramDataset(
            size=self.config.test_size, ps=self.ps, seed=self.config.seed + 2
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
