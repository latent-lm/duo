import lightning as L
import torch
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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        print(f"Entropy: {self.entropy(self.config.ps)}")

    def entropy(self, ps):
        ps = process_ps(ps)
        return -(ps * torch.log(ps)).sum()

    def setup(self, stage: str | None = None):
        del stage
        self.train_dataset = UnigramDataset(
            size=self.config.train_size, ps=self.config.ps, seed=self.config.seed
        )
        self.val_dataset = UnigramDataset(
            size=self.config.val_size, ps=self.config.ps, seed=self.config.seed + 1
        )
        self.test_dataset = UnigramDataset(
            size=self.config.test_size, ps=self.config.ps, seed=self.config.seed + 2
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
