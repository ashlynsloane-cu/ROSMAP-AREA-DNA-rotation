"""Shared Hsp60/10-inspired publication styling for ROSMAP AREA figures."""
from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

AREA_COLOR = "#7A8FA8"
CONV_COLOR = "#B8C0CA"
INK = "#283038"
MID = "#66727C"
GRID = "#E6E6E6"


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.labelweight": "bold",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
    })


def style_axes(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="both", which="major", length=3, width=0.5,
                   color=INK, labelcolor=INK)
    ax.tick_params(axis="y", which="major", length=0)
    ax.set_axisbelow(True)
    ax.grid(False)
    if grid_axis in {"x", "y"}:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.45)


def method_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", color=AREA_COLOR, linestyle="None",
               markersize=5, label="AREA"),
        Line2D([0], [0], marker="s", color=CONV_COLOR,
               markeredgecolor=INK, linestyle="None", markersize=5,
               label="Conventional"),
    ]


def save_publication(fig: plt.Figure, output_dir, stem: str) -> None:
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Fixed physical dimensions, as in the Hsp60/10 publication pipeline.
    fig.savefig(output_dir / f"{stem}.png", dpi=600, facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", facecolor="white")
    fig.savefig(output_dir / f"{stem}.svg", facecolor="white")
