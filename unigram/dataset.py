import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class UnigramDataset(Dataset):
    """Exact unigram dataset with sequence length 1."""

    def __init__(self, size: int, p_a: float, seed: int):
        super().__init__()
        n_a = int(round(size * p_a))
        n_b = size - n_a
        tokens = torch.cat(
            [
                torch.zeros(n_a, dtype=torch.long),
                torch.ones(n_b, dtype=torch.long),
            ]
        )
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(tokens.numel(), generator=generator)
        self.tokens = tokens[permutation]

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

    def entropy(self, p: float):
        p = torch.FloatTensor([p])
        return - (p * torch.log(p) + (1.0 - p) * torch.log((1.0 - p)))

    def setup(self, stage: str | None = None):
        del stage
        self.train_dataset = UnigramDataset(
            size=self.config.train_size, p_a=self.config.p_a, seed=self.config.seed
        )
        self.val_dataset = UnigramDataset(
            size=self.config.val_size, p_a=self.config.p_a, seed=self.config.seed + 1
        )
        self.test_dataset = UnigramDataset(
            size=self.config.val_size, p_a=self.config.p_a, seed=self.config.seed + 2
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
