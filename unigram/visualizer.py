import json
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

    def to_dict(self):
        return {
            name: [
                {"step": int(step), "value": float(value)}
                for step, value in items
            ]
            for name, items in self.history_dict.items()
        }

    def save(self, file):
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return file

def _moving_average(values, window: int):
    if window <= 1:
        return list(values)

    smoothed = []
    running_sum = 0.0
    for idx, value in enumerate(values):
        running_sum += value
        if idx >= window:
            running_sum -= values[idx - window]
        count = min(idx + 1, window)
        smoothed.append(running_sum / count)
    return smoothed


def plot_loss_curves(
    recorder: Recorder,
    output_path,
    series_names=("train_loss", "val_loss", "test_loss"),
    title: str = "Loss vs Steps",
    moving_average_window: int | None = None,
    show_raw: bool = True,
):
    available_series = [
        name for name in series_names if recorder.get_items(name)
    ]
    if not available_series:
        raise ValueError("Recorder does not contain any loss series to plot.")
    if moving_average_window is not None and moving_average_window < 1:
        raise ValueError("moving_average_window must be >= 1.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in available_series:
        steps, values = recorder.get_series(name)
        label = name.replace("_", " ")
        if len(steps) == 1:
            ax.scatter(steps, values, s=48, label=label)
            continue
        if moving_average_window is not None and moving_average_window > 1:
            if show_raw:
                ax.plot(steps, values, linewidth=1.2, alpha=0.3, label=f"{label} raw")
            smoothed = _moving_average(values, moving_average_window)
            ax.plot(
                steps,
                smoothed,
                linewidth=2.2,
                label=f"{label} ma{moving_average_window}",
            )
        else:
            ax.plot(steps, values, linewidth=2, label=label)

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

class DataMgr:
    def __init__(self, folder: str):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def path(self, name):
        return self.folder / name

    def save(
        self,
        recorder: Recorder,
        data_file="loss_history.json",
        fig_file="loss_curves.png",
        **plot_kwargs,
    ):
        data_path = recorder.save(self.path(data_file))
        fig_path = plot_loss_curves(
            recorder=recorder,
            output_path=self.path(fig_file),
            **plot_kwargs,
        )
        return {"data_path": data_path, "figure_path": fig_path}
