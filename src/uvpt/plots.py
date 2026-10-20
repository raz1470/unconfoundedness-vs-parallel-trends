"""Matplotlib figures for the diagnostics and estimates.

Each function draws on an ``ax`` (created if not given) and returns it, so
figures can be composed into grids.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uvpt.simulate import POST_WINDOW, PRE_WINDOW

CONTROL = "#2a78d6"
TREATED = "#eb6834"
INK = "#1a1a19"
MUTED = "#6b6a63"
GRID = "#e6e5e0"
POST_FILL = "#fbe3d8"

STYLE = {
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlelocation": "left",
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
}


def use_style() -> None:
    """Apply the repo's plot style globally."""
    plt.rcParams.update(STYLE)


def _ax(ax):
    return ax if ax is not None else plt.subplots(figsize=(6, 4))[1]


def _grid(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_event_means(
    means: pd.DataFrame,
    ax=None,
    title: str | None = None,
    labels: tuple[str, str] = ("Control", "Treated"),
    label_ends: bool = True,
):
    """Treated and control means by event time with 95% bands."""
    ax = _ax(ax)
    for g, colour, label in ((0, CONTROL, labels[0]), (1, TREATED, labels[1])):
        q = means[means["treated"] == g].sort_values("event_time")
        ax.fill_between(
            q["event_time"], q["ci_low"], q["ci_high"], color=colour, alpha=0.18, lw=0
        )
        ax.plot(
            q["event_time"],
            q["mean"],
            color=colour,
            lw=2,
            marker="o",
            ms=4,
            label=label,
        )
        if label_ends:
            ax.annotate(
                label,
                (q["event_time"].iloc[-1], q["mean"].iloc[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                color=INK,
                fontsize=9,
            )
    ax.axvline(-0.5, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("Event time (weeks)")
    ax.set_ylabel("Mean outcome")
    ax.set_xticks(sorted(means["event_time"].unique()))
    if title:
        ax.set_title(title)
    _grid(ax)
    return ax


def plot_gaps(
    gaps: pd.DataFrame,
    ax=None,
    title: str | None = None,
    pre: tuple[int, int] = PRE_WINDOW,
    post: tuple[int, int] = POST_WINDOW,
    shade: bool = True,
):
    """Treated minus control by event time, with the pre and post windows shaded."""
    ax = _ax(ax)
    if shade:
        ax.axvspan(pre[0] - 0.5, pre[1] + 0.5, color=GRID, alpha=0.7, lw=0)
        ax.axvspan(post[0] - 0.5, post[1] + 0.5, color=POST_FILL, alpha=0.6, lw=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.fill_between(
        gaps["event_time"], gaps["ci_low"], gaps["ci_high"], color=INK, alpha=0.12, lw=0
    )
    ax.plot(gaps["event_time"], gaps["gap"], color=INK, lw=2, marker="o", ms=4)
    ax.set_xlabel("Event time (weeks)")
    ax.set_ylabel("Treated minus control")
    ax.set_xticks(gaps["event_time"].tolist())
    if title:
        ax.set_title(title)
    _grid(ax)
    return ax


def plot_overlap(
    ps: pd.Series, d: pd.Series, ax=None, title: str | None = None, bins: int = 40
):
    """Propensity score histograms for treated and control, as densities."""
    ax = _ax(ax)
    edges = np.linspace(0, 1, bins + 1)
    for g, colour, label in ((0, CONTROL, "Control"), (1, TREATED, "Treated")):
        ax.hist(
            ps[d == g],
            bins=edges,
            density=True,
            color=colour,
            alpha=0.45,
            label=label,
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xlabel("Propensity score")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    if title:
        ax.set_title(title)
    _grid(ax)
    return ax


def plot_balance(
    balance: pd.DataFrame, ax=None, title: str | None = None, threshold: float = 0.1
):
    """Love plot of standardised mean differences, raw and weighted."""
    ax = _ax(ax)
    b = balance.iloc[::-1]
    pos = np.arange(len(b))
    ax.axvline(0, color=MUTED, lw=0.8)
    for x in (-threshold, threshold):
        ax.axvline(x, color=MUTED, lw=0.8, ls=":")
    ax.scatter(b["smd_raw"], pos, color=MUTED, s=28, label="Raw", zorder=3)
    if "smd_weighted" in b:
        ax.scatter(
            b["smd_weighted"], pos, color=TREATED, s=28, label="Weighted", zorder=3
        )
    ax.set_yticks(pos, b.index)
    ax.set_xlabel("Standardised mean difference")
    ax.legend(frameon=False, loc="lower right")
    if title:
        ax.set_title(title)
    return ax


def plot_estimates(
    table: pd.DataFrame,
    truth: float | None = None,
    ax=None,
    title: str | None = None,
    order: Sequence[str] | None = None,
):
    """Point estimates with 95% intervals, one row per method."""
    ax = _ax(ax)
    t = table.set_index("method")
    if order is not None:
        t = t.loc[[m for m in order if m in t.index]]
    t = t.iloc[::-1]
    pos = np.arange(len(t))
    ax.hlines(pos, t["ci_low"], t["ci_high"], color=INK, lw=2)
    ax.scatter(t["estimate"], pos, color=INK, s=36, zorder=3)
    if truth is not None:
        ax.axvline(truth, color=TREATED, lw=1.5, ls="--")
        ax.annotate(
            "truth",
            (truth, len(t) - 0.5),
            xytext=(4, 0),
            textcoords="offset points",
            color=TREATED,
            fontsize=9,
        )
    ax.set_yticks(pos, t.index)
    ax.set_xlabel("ATT estimate")
    if title:
        ax.set_title(title)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    return ax


def plot_mc(
    results: pd.DataFrame,
    scenario: str,
    ax=None,
    title: str | None = None,
    order: Sequence[str] | None = None,
):
    """Monte Carlo spread of estimates for one scenario.

    Dot is the mean estimate, the thick line the interquartile range and the
    thin line the 2.5th to 97.5th percentile across replications.
    """
    ax = _ax(ax)
    r = results[results["scenario"] == scenario]
    methods = order or list(dict.fromkeys(r["method"]))
    methods = [m for m in methods if m in set(r["method"])][::-1]
    for i, m in enumerate(methods):
        e = r.loc[r["method"] == m, "estimate"]
        q = e.quantile([0.025, 0.25, 0.75, 0.975]).to_numpy()
        ax.hlines(i, q[0], q[3], color=INK, lw=1)
        ax.hlines(i, q[1], q[2], color=INK, lw=4)
        ax.scatter(e.mean(), i, color=TREATED, s=40, zorder=3)
    truth = r["truth"].mean()
    ax.axvline(truth, color=TREATED, lw=1.5, ls="--")
    ax.set_yticks(range(len(methods)), methods)
    ax.set_xlabel("ATT estimate")
    if title:
        ax.set_title(title)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    return ax
