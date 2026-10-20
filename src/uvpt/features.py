"""Build the feature sets and outcomes for each design.

Everything works on a wide frame with one row per user. Outcome columns are
named by event time (``y_m8`` is t-8, ``y_p0`` is t0), so the timing of every
feature is visible in its name.

Covariate rules:

- Unconfoundedness: condition on the outcome history up to the cutoff (lags,
  slope, recent change) plus non-outcome covariates.
- DiD: non-outcome covariates only. Outcome lags from the pre and post windows
  are excluded, because conditioning on them induces regression to the mean
  (Daw and Hatfield 2018). ``include_early_lags`` adds t-8 to t-5 so the cost
  can be measured.

The functions take column names as arguments so they can be pointed at any
long panel with a user id, event time, binary treatment and outcome.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from uvpt.simulate import COVARIATES, POST_WINDOW, PRE_WINDOW

EARLY_WINDOW = (-8, -5)


def ycol(k: int) -> str:
    """Column name for the outcome at event time ``k``."""
    return f"y_m{-k}" if k < 0 else f"y_p{k}"


def ycols(lo: int, hi: int) -> list[str]:
    """Outcome column names for event times ``lo`` to ``hi`` inclusive."""
    return [ycol(k) for k in range(lo, hi + 1)]


def to_wide(
    panel: pd.DataFrame,
    *,
    user: str = "user_id",
    time: str = "event_time",
    treated: str = "treated",
    outcome: str = "y",
    covariates: Sequence[str] = COVARIATES,
    week: str | None = "week",
) -> pd.DataFrame:
    """One row per user: treatment, covariates, index week and outcome columns.

    ``week`` is the calendar week column of the long panel. The index week is
    its value at event time 0. Pass ``None`` if the panel has no calendar.
    """
    y = panel.pivot(index=user, columns=time, values=outcome)
    y.columns = [ycol(int(k)) for k in y.columns]
    first = panel.groupby(user)[[treated, *covariates]].first()
    out = first.rename(columns={treated: "treated"}).join(y)
    if week is not None:
        at_index = panel.loc[panel[time] == 0].set_index(user)[week]
        out["index_week"] = at_index
    out["treated"] = out["treated"].astype(int)
    return out


def apply_transform(wide: pd.DataFrame, f: Callable | None) -> pd.DataFrame:
    """Apply ``f`` to every outcome column, leaving the rest alone."""
    if f is None:
        return wide
    out = wide.copy()
    cols = [c for c in out.columns if c.startswith(("y_m", "y_p"))]
    out[cols] = f(out[cols].to_numpy())
    return out


def winsoriser(panel: pd.DataFrame, q: float = 0.99, outcome: str = "y"):
    """Return a function that caps weekly outcomes at the pooled ``q`` quantile."""
    cap = float(panel[outcome].quantile(q))
    return lambda v: np.minimum(v, cap)


def season_features(index_week: pd.Series, period: int = 52) -> pd.DataFrame:
    """Sine and cosine of the index week, so seasonality enters smoothly."""
    angle = 2 * np.pi * index_week.to_numpy() / period
    return pd.DataFrame(
        {"season_sin": np.sin(angle), "season_cos": np.cos(angle)},
        index=index_week.index,
    )


def baseline_features(
    wide: pd.DataFrame, covariates: Sequence[str] = COVARIATES
) -> pd.DataFrame:
    """Non-outcome covariates plus index-week seasonality when available."""
    parts = [wide[list(covariates)]]
    if "index_week" in wide:
        parts.append(season_features(wide["index_week"]))
    return pd.concat(parts, axis=1)


def history_features(wide: pd.DataFrame, cutoff: int = -1, start: int = -8):
    """Lags from ``start`` to ``cutoff``, their slope and the recent change.

    Slope is the OLS slope of the lags on event time. Recent change is the mean
    of the last two lags minus the mean of the first four (or of the first
    half when fewer than eight lags are available).
    """
    cols = ycols(start, cutoff)
    lags = wide[cols].to_numpy()
    t = np.arange(start, cutoff + 1, dtype=float)
    tc = t - t.mean()
    slope = (lags - lags.mean(axis=1, keepdims=True)) @ tc / (tc @ tc)
    n_first = min(4, len(cols) // 2)
    recent = lags[:, -2:].mean(axis=1) - lags[:, :n_first].mean(axis=1)
    out = wide[cols].copy()
    out["slope"] = slope
    out["recent_change"] = recent
    return out


def window_mean(wide: pd.DataFrame, window: tuple[int, int]) -> pd.Series:
    """Mean outcome over event times ``window[0]`` to ``window[1]``."""
    return wide[ycols(*window)].mean(axis=1)


@dataclass
class Design:
    """Inputs for one estimator: outcome, treatment and features.

    For a DiD design ``y`` is the post-minus-pre change and ``pre`` holds the
    pre-window mean, so the two-period panel can be rebuilt.
    """

    name: str
    y: pd.Series
    d: pd.Series
    x: pd.DataFrame
    pre: pd.Series | None = None

    def __post_init__(self):
        idx = [self.y.index, self.d.index, self.x.index]
        if self.pre is not None:
            idx.append(self.pre.index)
        if not all(i.equals(idx[0]) for i in idx[1:]):
            raise ValueError("y, d, x and pre must share an index")


def unconf_design(
    wide: pd.DataFrame,
    covariates: Sequence[str] = COVARIATES,
    post: tuple[int, int] = POST_WINDOW,
) -> Design:
    """Unconfoundedness: post-window mean on history through t-1 plus baseline."""
    x = pd.concat(
        [history_features(wide, cutoff=-1), baseline_features(wide, covariates)], axis=1
    )
    return Design("unconf", window_mean(wide, post), wide["treated"], x)


def did_design(
    wide: pd.DataFrame,
    covariates: Sequence[str] = COVARIATES,
    pre: tuple[int, int] = PRE_WINDOW,
    post: tuple[int, int] = POST_WINDOW,
    include_early_lags: bool = False,
) -> Design:
    """Conditional DiD: post mean minus pre mean on baseline covariates.

    With ``include_early_lags`` the outcome at t-8 to t-5 is added as covariates.
    """
    x = baseline_features(wide, covariates)
    name = "did"
    if include_early_lags:
        x = pd.concat([x, wide[ycols(*EARLY_WINDOW)]], axis=1)
        name = "did_lags"
    y_pre = window_mean(wide, pre)
    y = window_mean(wide, post) - y_pre
    return Design(name, y, wide["treated"], x, pre=y_pre)


def placebo_designs(
    wide: pd.DataFrame,
    covariates: Sequence[str] = COVARIATES,
    placebo: tuple[int, int] = PRE_WINDOW,
) -> dict[str, Design]:
    """Both designs moved back in time, with ``placebo`` as a fake post window.

    Features stop at t-5, so the true effect on the placebo outcome is zero.
    A non-zero estimate means the design's assumption fails over t-4 to t-1.
    """
    base = baseline_features(wide, covariates)
    cutoff = placebo[0] - 1
    hist = history_features(wide, cutoff=cutoff)
    y_unconf = window_mean(wide, placebo)
    y_early = window_mean(wide, (EARLY_WINDOW[0], cutoff))
    return {
        "unconf": Design(
            "unconf_placebo", y_unconf, wide["treated"], pd.concat([hist, base], axis=1)
        ),
        "did": Design(
            "did_placebo", y_unconf - y_early, wide["treated"], base, pre=y_early
        ),
    }
