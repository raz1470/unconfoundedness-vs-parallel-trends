"""Simulate a weekly user panel with a self-selected treatment.

Each user has a stable unobserved trait, three observed baseline covariates and
an AR(1) transitory shock. Every user gets one candidate index week. At that
week the user takes the treatment with a probability set by the selection
mechanism of the chosen scenario:

- ``"a"``: observed covariates only. Both designs are valid.
- ``"b"``: observed covariates plus the stable trait. Parallel trends holds,
  unconfoundedness given lagged outcomes fails.
- ``"c"``: observed covariates plus low engagement over t-3 to t-1. Selection is
  on observables (the lags), so unconfoundedness holds. Part of a low recent
  level is a transitory shock that reverts, so parallel trends fails.
- ``"d"``: both the trait and the dip. Each design is biased in the opposite
  direction, which brackets the truth.

Selection is negative throughout: low-trait users and users whose engagement
just dropped are more likely to take the treatment. This matches the setting
behind the Angrist and Pischke bracketing result.

Potential outcomes are generated for every user, so the true ATT is known
exactly on any outcome scale. They are returned separately from the observed
panel so that estimators cannot use them by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

EVENT_TIMES = np.arange(-8, 5)
PRE_WINDOW = (-4, -1)
POST_WINDOW = (0, 3)
COVARIATES = ("x1", "x2", "x3")
SCENARIOS = ("a", "b", "c", "d")


@dataclass(frozen=True)
class SimConfig:
    """Parameters of the data generating process.

    Engagement is in arbitrary weekly units (think minutes per week).
    Selection strengths are log-odds per standard deviation of the driver.
    """

    n_users: int = 10_000
    n_weeks: int = 80
    tau: float = 2.0
    treated_share: float = 0.2
    # Outcome process
    mu: float = 30.0
    sigma_trait: float = 4.0
    rho: float = 0.8
    sigma_eps: float = 3.0
    season_amp: float = 2.0
    beta_x: tuple[float, float, float] = (2.0, -1.5, 1.0)
    gamma_x: tuple[float, float, float] = (-1.5, 0.0, 1.0)
    # Selection
    sel_x: tuple[float, float, float] = (-0.6, 0.4, 0.0)
    sel_trait: float = 0.0
    sel_dip: float = 0.0
    # Heavy tails: multiplicative lognormal noise with mean one
    heavy_tail: bool = False
    heavy_tail_sigma: float = 0.8
    # Eligibility: active weeks in the 4 weeks before index
    activity_threshold: float = 0.0
    min_active_weeks: int = 4
    # "candidate": controls keep the week they declined at.
    # "resample": controls get a week drawn from the treated index weeks.
    control_index: str = "candidate"
    seed: int = 0
    scenario: str = "custom"


SCENARIO_SELECTION = {
    "a": {"sel_trait": 0.0, "sel_dip": 0.0},
    "b": {"sel_trait": 1.0, "sel_dip": 0.0},
    "c": {"sel_trait": 0.0, "sel_dip": 1.0},
    "d": {"sel_trait": 1.0, "sel_dip": 1.0},
}


def scenario_config(scenario: str, **overrides) -> SimConfig:
    """Return the config for scenario ``a`` to ``d`` with optional overrides."""
    if scenario not in SCENARIO_SELECTION:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")
    params = {**SCENARIO_SELECTION[scenario], "scenario": scenario, **overrides}
    return SimConfig(**params)


@dataclass
class SimResult:
    """Observed panel, oracle quantities and the config that produced them."""

    panel: pd.DataFrame
    oracle: pd.DataFrame
    config: SimConfig
    users: pd.DataFrame = field(repr=False)

    def true_att(self, transform=None, window=POST_WINDOW) -> float:
        """True ATT on the mean outcome over ``window``, on the given scale.

        ``transform`` is applied to each weekly potential outcome before
        averaging, so it must match the transform used by the estimator.
        """
        f = transform if transform is not None else (lambda v: v)
        o = self.oracle
        o = o[o["treated"] == 1]
        o = o[o["event_time"].between(*window)]
        diff = f(o["y1"].to_numpy()) - f(o["y0"].to_numpy())
        return float(pd.Series(diff).groupby(o["user_id"].to_numpy()).mean().mean())


def _ar1(rng, n, t, rho, sigma):
    u = np.empty((n, t))
    u[:, 0] = rng.normal(0, sigma / np.sqrt(1 - rho**2), n)
    eps = rng.normal(0, sigma, (n, t))
    for w in range(1, t):
        u[:, w] = rho * u[:, w - 1] + eps[:, w]
    return u


def _window(mat, index_week):
    cols = index_week[:, None] + EVENT_TIMES[None, :]
    return np.take_along_axis(mat, cols, axis=1)


def _standardise(v):
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v - v.mean()


def _calibrate_intercept(lin, share):
    return brentq(lambda a: expit(a + lin).mean() - share, -30, 30)


def simulate(config: SimConfig | None = None, **overrides) -> SimResult:
    """Simulate one dataset. Keyword overrides are applied to ``config``."""
    cfg = replace(config or SimConfig(), **overrides)
    if cfg.control_index not in ("candidate", "resample"):
        raise ValueError("control_index must be 'candidate' or 'resample'")
    rng = np.random.default_rng(cfg.seed)
    n, n_weeks = cfg.n_users, cfg.n_weeks
    lo, hi = -EVENT_TIMES.min(), n_weeks - 1 - EVENT_TIMES.max()
    if hi < lo:
        raise ValueError("n_weeks is too short for the event window")

    x = np.column_stack(
        [rng.normal(size=n), rng.binomial(1, 0.4, n), rng.normal(size=n)]
    ).astype(float)
    trait = rng.normal(0, cfg.sigma_trait, n)
    u = _ar1(rng, n, n_weeks, cfg.rho, cfg.sigma_eps)
    weeks = np.arange(n_weeks)
    season = cfg.season_amp * np.sin(2 * np.pi * weeks / 52)
    level = cfg.mu + trait + x @ np.asarray(cfg.beta_x)
    slope = x @ np.asarray(cfg.gamma_x)
    z0 = level[:, None] + slope[:, None] * weeks[None, :] / 52 + season + u
    noise = (
        rng.lognormal(-0.5 * cfg.heavy_tail_sigma**2, cfg.heavy_tail_sigma, z0.shape)
        if cfg.heavy_tail
        else np.ones_like(z0)
    )
    y0_full = np.clip(z0, 0, None) * noise

    candidate = rng.integers(lo, hi + 1, n)
    y0_cand = _window(y0_full, candidate)
    k = EVENT_TIMES
    dip = y0_cand[:, (k >= -3) & (k <= -1)].mean(axis=1)
    lin = (
        _standardise(x) @ np.asarray(cfg.sel_x)
        - cfg.sel_trait * _standardise(trait)
        - cfg.sel_dip * _standardise(dip)
    )
    p = expit(_calibrate_intercept(lin, cfg.treated_share) + lin)
    treated = rng.binomial(1, p)

    index_week = candidate.copy()
    if cfg.control_index == "resample":
        ctrl = treated == 0
        index_week[ctrl] = rng.choice(candidate[treated == 1], ctrl.sum())

    post = (k >= 0).astype(float)
    y0 = _window(y0_full, index_week)
    z1 = np.clip(_window(z0, index_week) + cfg.tau * post, 0, None)
    y1 = np.where(post[None, :] == 1, z1 * _window(noise, index_week), y0)
    y = np.where(treated[:, None] == 1, y1, y0)

    pre = (k >= -4) & (k <= -1)
    active_pre = (y[:, pre] > cfg.activity_threshold).sum(1)
    keep = active_pre >= cfg.min_active_weeks

    users = pd.DataFrame(
        {
            "user_id": np.arange(n),
            "treated": treated,
            "index_week": index_week,
            "x1": x[:, 0],
            "x2": x[:, 1],
            "x3": x[:, 2],
            "trait": trait,
            "dip": dip,
            "p_select": p,
            "eligible": keep,
        }
    )
    ids = np.flatnonzero(keep)
    m = len(EVENT_TIMES)
    long_ids = np.repeat(ids, m)
    base = {
        "user_id": long_ids,
        "event_time": np.tile(EVENT_TIMES, len(ids)),
        "week": (index_week[ids, None] + EVENT_TIMES[None, :]).ravel(),
        "treated": np.repeat(treated[ids], m),
    }
    panel = pd.DataFrame(
        {
            **base,
            "y": y[ids].ravel(),
            "x1": np.repeat(x[ids, 0], m),
            "x2": np.repeat(x[ids, 1], m),
            "x3": np.repeat(x[ids, 2], m),
        }
    )
    oracle = pd.DataFrame({**base, "y0": y0[ids].ravel(), "y1": y1[ids].ravel()})
    return SimResult(panel=panel, oracle=oracle, config=cfg, users=users)
