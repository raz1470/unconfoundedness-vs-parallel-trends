"""Diagnostics that say which identifying assumption the data can support.

- Event-time means and gaps, raw or ATT-weighted. Controls get weight
  ``p / (1 - p)``, so the weighted control line estimates what the treated
  would have looked like under the propensity model.
- Balance on features through t-1. When the propensity is fitted on those
  same features, weighted balance is close to perfect by construction, so it
  cannot test anything.
- Pre-trends (the DiD check): weight on non-outcome covariates only, express
  each user's outcome relative to their own t-8 to t-5 mean, and look for
  divergence over t-4 to t-1. Flat means parallel trends is plausible.
- Placebo (the unconfoundedness check): fit the propensity on the outcome
  history through t-5 and look for divergence over t-4 to t-1. A drift points
  to an unobserved stable trait. A sharp dip just before index points to
  selection on recent outcomes, which this check cannot validate, since the
  placebo drops exactly those lags.
- ``placebo_test`` runs both DML designs on the placebo window and returns
  the estimates. Their true value is zero.
- Sensitivity: DoubleML's omitted-variable bounds for a fitted estimate.

Every function takes a long panel or a wide frame from ``uvpt.features``, so
it can be run on real data with the same columns.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from uvpt.estimators import CLIP, Estimate, dml_did, dml_unconf, learners
from uvpt.features import (
    EARLY_WINDOW,
    baseline_features,
    history_features,
    placebo_designs,
    window_mean,
    ycol,
    ycols,
)
from uvpt.simulate import COVARIATES, EVENT_TIMES, PRE_WINDOW

Z = 1.959964


def fit_propensity(
    x: pd.DataFrame,
    d: pd.Series,
    kind: str = "lgbm",
    fast: bool = True,
    n_folds: int = 5,
    seed: int = 0,
) -> pd.Series:
    """Cross-fitted propensity scores, one per row of ``x``."""
    _, clf = learners(kind, fast, seed)
    folds = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    ps = cross_val_predict(clf, x, d, cv=folds, method="predict_proba")[:, 1]
    return pd.Series(ps, index=x.index, name="ps")


def propensity_features(
    wide: pd.DataFrame, cutoff: int = -1, covariates: Sequence[str] = COVARIATES
) -> pd.DataFrame:
    """Outcome history through ``cutoff`` plus baseline covariates."""
    return pd.concat(
        [history_features(wide, cutoff=cutoff), baseline_features(wide, covariates)],
        axis=1,
    )


def att_weights(ps: pd.Series, d: pd.Series, clip: float = CLIP) -> pd.Series:
    """Weight 1 for treated and ``p / (1 - p)`` for controls, clipped."""
    p = ps.clip(clip, 1 - clip)
    return pd.Series(np.where(d == 1, 1.0, p / (1 - p)), index=ps.index, name="w")


def weighted_mean_se(y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted mean and its SE, treating the weights as fixed."""
    w = w / w.sum()
    m = float(np.sum(w * y))
    se = float(np.sqrt(np.sum(w**2 * (y - m) ** 2)))
    return m, se


def event_time_means(
    wide: pd.DataFrame,
    weights: pd.Series | None = None,
    event_times: Sequence[int] = EVENT_TIMES,
) -> pd.DataFrame:
    """Mean outcome per group and event time with a 95% interval.

    With ``weights`` the means are weighted and the SE is the usual
    sandwich form for a weighted mean (weights treated as fixed).
    """
    w_all = pd.Series(1.0, index=wide.index) if weights is None else weights
    rows = []
    for g in (0, 1):
        mask = wide["treated"] == g
        for k in event_times:
            y = wide.loc[mask, ycol(int(k))].to_numpy()
            m, se = weighted_mean_se(y, w_all[mask].to_numpy())
            rows.append({"treated": g, "event_time": int(k), "mean": m, "se": se})
    out = pd.DataFrame(rows)
    out["ci_low"] = out["mean"] - Z * out["se"]
    out["ci_high"] = out["mean"] + Z * out["se"]
    return out


def event_time_gaps(means: pd.DataFrame) -> pd.DataFrame:
    """Treated minus control at each event time, from ``event_time_means``."""
    t = means[means["treated"] == 1].set_index("event_time")
    c = means[means["treated"] == 0].set_index("event_time")
    gap = t["mean"] - c["mean"]
    se = np.sqrt(t["se"] ** 2 + c["se"] ** 2)
    return pd.DataFrame(
        {"gap": gap, "se": se, "ci_low": gap - Z * se, "ci_high": gap + Z * se}
    ).reset_index()


def balance_table(
    x: pd.DataFrame, d: pd.Series, weights: pd.Series | None = None
) -> pd.DataFrame:
    """Standardised mean differences, raw and (optionally) weighted.

    The SMD uses the pooled raw standard deviation in both columns, so the
    two are on the same scale.
    """
    d = d.to_numpy().astype(bool)
    sd = np.sqrt((x[d].var() + x[~d].var()) / 2).replace(0, np.nan)
    raw = (x[d].mean() - x[~d].mean()) / sd
    out = pd.DataFrame({"smd_raw": raw})
    if weights is not None:
        w = weights.to_numpy()[~d]
        wmean = (x[~d].mul(w, axis=0)).sum() / w.sum()
        out["smd_weighted"] = (x[d].mean() - wmean) / sd
    return out


def _window_summary(gaps, window):
    g = gaps[gaps["event_time"].between(*window)]
    return float(g["gap"].mean()), float((g["gap"] / g["se"]).abs().max())


def pretrend_event_time(
    wide: pd.DataFrame,
    reference: tuple[int, int] = EARLY_WINDOW,
    window: tuple[int, int] = PRE_WINDOW,
    kind: str = "lgbm",
    fast: bool = True,
    seed: int = 0,
    covariates: Sequence[str] = COVARIATES,
) -> dict:
    """Pre-trend check for DiD: covariate-weighted gaps in outcome changes.

    Each user's outcome is expressed relative to their own mean over
    ``reference``, so stable level differences drop out. Controls are weighted
    by a propensity fitted on non-outcome covariates only, matching the DiD
    design. Under conditional parallel trends the gap is zero over ``window``.
    """
    rel = wide.copy()
    cols = [c for c in ycols(-8, 4) if c in rel]
    rel[cols] = rel[cols].sub(window_mean(wide, reference), axis=0)
    x = baseline_features(wide, covariates)
    ps = fit_propensity(x, wide["treated"], kind, fast, seed=seed)
    w = att_weights(ps, wide["treated"])
    gaps = event_time_gaps(event_time_means(rel, w))
    mean_gap, max_z = _window_summary(gaps, window)
    return {
        "propensity": ps,
        "weights": w,
        "gaps": gaps,
        "pretrend_mean_gap": mean_gap,
        "pretrend_max_abs_z": max_z,
    }


def placebo_event_time(
    wide: pd.DataFrame,
    cutoff: int = -5,
    kind: str = "lgbm",
    fast: bool = True,
    seed: int = 0,
    covariates: Sequence[str] = COVARIATES,
) -> dict:
    """Weighted event-time gaps with a propensity fitted through ``cutoff``.

    Returns the propensity, the weighted gaps and a summary of the mean gap
    over ``cutoff + 1`` to t-1 (the placebo window), where it should be zero.
    """
    x = propensity_features(wide, cutoff, covariates)
    ps = fit_propensity(x, wide["treated"], kind, fast, seed=seed)
    w = att_weights(ps, wide["treated"])
    gaps = event_time_gaps(event_time_means(wide, w))
    mean_gap, max_z = _window_summary(gaps, (cutoff + 1, -1))
    return {
        "propensity": ps,
        "weights": w,
        "gaps": gaps,
        "placebo_mean_gap": mean_gap,
        "placebo_max_abs_z": max_z,
    }


def placebo_test(
    wide: pd.DataFrame,
    kind: str = "lgbm",
    fast: bool = True,
    seed: int = 0,
    covariates: Sequence[str] = COVARIATES,
) -> pd.DataFrame:
    """Both DML designs on the placebo window t-4 to t-1, features through t-5.

    The true effect is zero, so an interval that excludes zero rejects that
    design's assumption over the placebo window.
    """
    designs = placebo_designs(wide, covariates)
    kw = {"kind": kind, "fast": fast, "seed": seed}
    rows = [
        dml_unconf(designs["unconf"], **kw).as_row(),
        dml_did(designs["did"], **kw).as_row(),
    ]
    out = pd.DataFrame(rows)[["method", "estimate", "se", "ci_low", "ci_high"]]
    out["rejects_zero"] = (out["ci_low"] > 0) | (out["ci_high"] < 0)
    return out


def sensitivity(
    est: Estimate, cf_y: float = 0.03, cf_d: float = 0.03, rho: float = 1.0
) -> dict:
    """DoubleML omitted-variable bounds for a fitted DML estimate.

    ``cf_y`` and ``cf_d`` are the share of residual variation in the outcome
    and in the treatment (Riesz representer) an unobserved confounder would
    explain. Returns the bounds on the estimate, the bounds on its 95%
    interval and the robustness values.
    """
    if est.model is None:
        raise ValueError("sensitivity needs an Estimate from a DML estimator")
    model = est.model
    model.sensitivity_analysis(cf_y=cf_y, cf_d=cf_d, rho=rho)
    p = model.sensitivity_params
    return {
        "method": est.method,
        "estimate": est.estimate,
        "theta_lower": float(np.ravel(p["theta"]["lower"])[0]),
        "theta_upper": float(np.ravel(p["theta"]["upper"])[0]),
        "ci_lower": float(np.ravel(p["ci"]["lower"])[0]),
        "ci_upper": float(np.ravel(p["ci"]["upper"])[0]),
        "rv": float(np.ravel(p["rv"])[0]),
        "rva": float(np.ravel(p["rva"])[0]),
    }


# Two-sided 5% with a Bonferroni correction over the four pre-window weeks.
FLAG_Z = 2.498

READINGS = {
    (False, False): (
        "No pre-period divergence in either check. Both designs are plausible "
        "and their estimates should agree."
    ),
    (False, True): (
        "Pre-trends are flat but the unconfoundedness placebo drifts. This is "
        "the signature of a stable unobserved trait: prefer DiD."
    ),
    (True, False): (
        "Pre-trends diverge but the unconfoundedness placebo is flat. Parallel "
        "trends fails: prefer unconfoundedness and report DiD as the other bound."
    ),
    (True, True): (
        "Pre-trends diverge, so parallel trends fails. The unconfoundedness "
        "placebo cannot tell selection on recent outcomes from a stable trait. "
        "Prefer unconfoundedness if selection plausibly runs on recent "
        "engagement, and report both estimates as bounds."
    ),
}


def diagnose(
    wide: pd.DataFrame,
    kind: str = "lgbm",
    fast: bool = True,
    seed: int = 0,
    covariates: Sequence[str] = COVARIATES,
    z: float = FLAG_Z,
) -> dict:
    """Run both pre-period checks and map the result to the decision rule.

    A check flags when any week in t-4 to t-1 has ``|z| > z``. Returns both
    summaries, both flags and a one-paragraph reading.
    """
    kw = {"kind": kind, "fast": fast, "seed": seed, "covariates": covariates}
    pre = pretrend_event_time(wide, **kw)
    plac = placebo_event_time(wide, **kw)
    flags = (pre["pretrend_max_abs_z"] > z, plac["placebo_max_abs_z"] > z)
    return {
        "pretrend": pre,
        "placebo": plac,
        "pretrend_flag": bool(flags[0]),
        "placebo_flag": bool(flags[1]),
        "reading": READINGS[flags],
    }
