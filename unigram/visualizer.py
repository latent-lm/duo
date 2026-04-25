from pathlib import Path

import torch


class Recorder:
    def __init__(self):
        self.history_dict = {}

    def add(self, name: str, step: int, val):
        if isinstance(val, torch.Tensor):
            val = float(val.detach().cpu())
        else:
            val = float(val)
        self.history_dict.setdefault(name, []).append((int(step), val))

    def get_items(self, name: str):
        return self.history_dict.get(name, [])

    def get_series(self, name: str):
        items = self.get_items(name)
        if not items:
            return [], []
        steps, values = zip(*items)
        return list(steps), list(values)

    def last_step(self, name: str):
        items = self.get_items(name)
        if not items:
            return 0
        return int(items[-1][0])

    def get_keys(self):
        return self.history_dict.keys()


def plot_loss_curves(
    recorder: Recorder,
    output_path,
    series_names=("train_loss", "val_loss", "test_loss"),
    title: str = "Loss vs Steps",
):
    available_series = [
        name for name in series_names if recorder.get_items(name)
    ]
    if not available_series:
        raise ValueError("Recorder does not contain any loss series to plot.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in available_series:
        steps, values = recorder.get_series(name)
        ax.plot(steps, values, linewidth=2, label=name.replace("_", " "))

    ax.set_xlabel("steps")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if len(available_series) > 1:
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path



