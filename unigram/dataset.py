import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

def process_ps(ps):
    if isinstance(ps, list) or isinstance(ps, tuple):
        ps = torch.FloatTensor(ps)
        sum_ps = ps.sum().item()
        if len(ps) == 1 and sum_ps < 1.0:
            ps = torch.concat(ps, torch.FloatTensor([1.0 - sum_ps]))
    elif not torch.is_tensor(ps):
        ext_ps = [ps] + [1.0 - ps]
        ps = torch.FloatTensor(ext_ps)
    return ps

class UnigramDataset(Dataset):
    """Exact unigram dataset with sequence length 1."""

    def __init__(self, size: int, ps: float, seed: int):
        super().__init__()

        ps = process_ps(ps)

        token_repeat_times = float(size) * ps
        tokens = torch.arange(n_token, dtype=torch.long)
        perm_indice = torch.repeat_interleave(tokens, token_repeat_times)

        generator = torch.Generator().manual_seed(seed)
        rand_perm = torch.randperm(perm_indice.numel(), generator=generator)
        self.tokens = tokens[perm_indice[rand_perm]]

    def __len__(self):
        return self.tokens.numel()

    def __getitem__(self, index: int):
        return self.tokens[index]

class UnigramDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        print(f"Entropy: {self.entropy(p=self.config.p_a)}")

    # def entropy(self, p: float):
    #     p = torch.FloatTensor([p])
    #     return - (p * torch.log(p) + (1.0 - p) * torch.log((1.0 - p)))

    def entropy(self, ps):
        ps = process_ps(ps)
        return - (ps * torch.log(ps)).sum()

    def setup(self, stage: str | None = None):
        del stage
        self.train_dataset = UnigramDataset(
            size=self.config.train_size, ps=self.config.ps, seed=self.config.seed
        )
        self.val_dataset = UnigramDataset(
            size=self.config.val_size, ps=self.config.ps, seed=self.config.seed + 1
        )
        self.test_dataset = UnigramDataset(
            size=self.config.val_size, ps=self.config.ps, seed=self.config.seed + 2
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
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
