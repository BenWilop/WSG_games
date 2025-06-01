import torch as t
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Any
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


class IndexedDataset(Dataset):
    def __init__(self, original_dataset: Dataset):
        self.original_dataset = original_dataset

    def __len__(self) -> int:
        return len(self.original_dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        data_item = self.original_dataset[idx]
        return data_item, idx


@dataclass
class HistogramData:
    """
    Stores data and configuration for plotting a single histogram.
    """

    frequencies: np.ndarray
    bins: np.ndarray  # Sequence of bin edges (length N+1 for N bins)
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    x_scale_log: bool = False
    y_scale_log: bool = False
    color: str = "blue"

    def plot_histogram_data(self, ax: plt.Axes) -> None:
        """
        Plots on a given Matplotlib Axes.
        """
        assert len(self.bins) == len(self.frequencies) + 1
        if self.frequencies.size == 0:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="gray",
            )
        else:
            bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
            widths = np.diff(self.bins)
            ax.bar(
                bin_centers,
                self.frequencies,
                width=widths,
                color=self.color,
                alpha=0.7,
                edgecolor="black",
            )

        ax.set_title(self.title, fontsize=12)
        ax.set_xlabel(self.xlabel, fontsize=10)
        ax.set_ylabel(self.ylabel, fontsize=10)

        ax.set_xscale("log" if self.x_scale_log else "linear")
        ax.set_yscale("log" if self.y_scale_log else "linear")

        # Enforce that log scaled axes do not go below 0.1
        if self.x_scale_log:
            current_xmin, current_xmax = ax.get_xlim()
            new_xmin = max(0.1, current_xmin if current_xmin > 0 else 0.1)
            if new_xmin >= current_xmax:
                new_xmin = 0.1
                current_xmax = new_xmin * 10
            ax.set_xlim(left=new_xmin, right=current_xmax)

        if self.y_scale_log:
            current_ymin, current_ymax = ax.get_ylim()
            new_ymin = max(0.1, current_ymin if current_ymin > 0 else 0.1)
            if new_ymin >= current_ymax:
                new_ymin = 0.1
                current_ymax = new_ymin * 10
            ax.set_ylim(bottom=new_ymin, top=current_ymax)
            if self.frequencies.size == 0:
                ax.set_ylim(bottom=0.1, top=1)

        ax.grid(True, linestyle="--", alpha=0.6)
