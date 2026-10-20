"""Estimators for the ATT: naive, unconfoundedness DML and DiD DML.

Both DML estimators use cross-fitting and the doubly robust score for the
ATT. The design (which outcome, which covariates) comes from
``uvpt.features``; the estimator only sees a ``Design``. That separation is
the point: the same DML machinery gives different answers depending on which
identifying assumption the design encodes.

- ``dml_unconf``: ``DoubleMLIRM`` with ``score="ATTE"``.
- ``dml_did``: ``DoubleMLDIDBinary`` on a two-period panel (pre-window mean
  and post-window mean), the doubly robust DiD of Sant'Anna and Zhao (2020).
  ``DoubleMLDID`` gives the same estimator but is deprecated in DoubleML 0.11.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import doubleml as dml
import numpy as np
import pandas as pd
from doubleml.did import DoubleMLDIDBinary
from doubleml.utils.propensity_score_processing import PSProcessorConfig
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy import stats
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from uvpt.features import Design, did_design, unconf_design

CLIP = 0.01


@dataclass
class Estimate:
    """One ATT estimate with its 95% interval and overlap summary."""

    method: str
    estimate: float
    se: float
    n: int
    n_treated: int
    overlap: dict = field(default_factory=dict)
    model: object = field(default=None, repr=False)

    @property
    def ci(self) -> tuple[float, float]:
        z = stats.norm.ppf(0.975)
        return self.estimate - z * self.se, self.estimate + z * self.se

    def as_row(self) -> dict:
        lo, hi = self.ci
        return {
            "method": self.method,
            "estimate": self.estimate,
            "se": self.se,
            "ci_low": lo,
            "ci_high": hi,
            "n": self.n,
            "n_treated": self.n_treated,
            **{f"overlap_{k}": v for k, v in self.overlap.items()},
        }


def learners(kind: str = "lgbm", fast: bool = False, seed: int = 0):
    """Return ``(regressor, classifier)`` nuisance learners.

    ``kind`` is ``"lgbm"`` (LightGBM) or ``"hgb"`` (scikit-learn histogram
    gradient boosting). ``fast`` uses fewer trees, for Monte Carlo runs.
    """
    n_trees = 150 if fast else 400
    if kind == "lgbm":
        common = {
            "n_estimators": n_trees,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 50,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "random_state": seed,
            "n_jobs": 1,
            "verbose": -1,
        }
        return LGBMRegressor(**common), LGBMClassifier(**common)
    if kind == "hgb":
        common = {
            "max_iter": n_trees,
            "learning_rate": 0.05,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 50,
            "random_state": seed,
        }
        return (
            HistGradientBoostingRegressor(**common),
            HistGradientBoostingClassifier(**common),
        )
    raise ValueError(f"kind must be 'lgbm' or 'hgb', got {kind!r}")


def _frame(design: Design) -> tuple[pd.DataFrame, list[str]]:
    x_cols = list(design.x.columns)
    df = design.x.copy()
    df["_y"] = design.y.to_numpy()
    df["_d"] = design.d.to_numpy()
    return df, x_cols


def overlap_summary(ps: np.ndarray, d: np.ndarray, clip: float = CLIP) -> dict:
    """Propensity overlap for the ATT.

    - ``share_clipped``: share of controls whose propensity is at or above
      ``1 - clip``, where ATT weights ``p / (1 - p)`` are capped.
    - ``ess_controls``: Kish effective sample size of the control weights.
    - ``max_weight_share``: largest single control weight over the total.
    """
    ps = np.asarray(ps, dtype=float)
    d = np.asarray(d).astype(bool)
    pc = np.clip(ps[~d], clip, 1 - clip)
    w = pc / (1 - pc)
    return {
        "ps_mean_treated": float(ps[d].mean()),
        "ps_mean_control": float(ps[~d].mean()),
        "ps_max_control": float(ps[~d].max()),
        "share_clipped": float((ps[~d] >= 1 - clip).mean()),
        "ess_controls": float(w.sum() ** 2 / (w**2).sum()),
        "max_weight_share": float(w.max() / w.sum()),
    }


def naive(design: Design) -> Estimate:
    """Difference in mean outcome between treated and control, Welch SE."""
    d = design.d.to_numpy().astype(bool)
    y = design.y.to_numpy()
    est = y[d].mean() - y[~d].mean()
    se = np.sqrt(y[d].var(ddof=1) / d.sum() + y[~d].var(ddof=1) / (~d).sum())
    return Estimate(f"naive_{design.name}", float(est), float(se), len(y), int(d.sum()))


def _set_seed(seed):
    # DoubleML draws its fold split from numpy's global generator.
    np.random.seed(seed)


def dml_unconf(
    design: Design,
    kind: str = "lgbm",
    fast: bool = False,
    n_folds: int = 5,
    clip: float = CLIP,
    seed: int = 0,
) -> Estimate:
    """ATT under unconfoundedness with ``DoubleMLIRM(score="ATTE")``."""
    df, x_cols = _frame(design)
    data = dml.DoubleMLData(df, y_col="_y", d_cols="_d", x_cols=x_cols)
    ml_g, ml_m = learners(kind, fast, seed)
    _set_seed(seed)
    model = dml.DoubleMLIRM(
        data,
        ml_g=ml_g,
        ml_m=ml_m,
        n_folds=n_folds,
        score="ATTE",
        ps_processor_config=PSProcessorConfig(clipping_threshold=clip),
    )
    model.fit(store_predictions=True)
    ps = model.predictions["ml_m"][:, 0, 0]
    return Estimate(
        design.name,
        float(model.coef[0]),
        float(model.se[0]),
        len(df),
        int(df["_d"].sum()),
        overlap_summary(ps, df["_d"], clip),
        model,
    )


def two_period_panel(design: Design) -> pd.DataFrame:
    """Long frame with a pre row (t=0) and a post row (t=1) per user.

    ``g`` is the first treated period: 1 for treated users, infinity for
    never-treated controls, as ``DoubleMLPanelData`` expects.
    """
    if design.pre is None:
        raise ValueError("DiD design needs the pre-window mean in design.pre")
    g = np.where(design.d.to_numpy() == 1, 1.0, np.inf)
    ids = design.x.index.to_numpy()
    rows = []
    for t, y in ((0, design.pre), (1, design.pre + design.y)):
        rows.append(design.x.assign(_id=ids, _t=t, _y=y.to_numpy(), _g=g))
    return pd.concat(rows, ignore_index=True)


def dml_did(
    design: Design,
    kind: str = "lgbm",
    fast: bool = False,
    n_folds: int = 5,
    clip: float = CLIP,
    seed: int = 0,
) -> Estimate:
    """ATT under conditional parallel trends with ``DoubleMLDIDBinary``.

    ``design`` must come from ``uvpt.features.did_design`` (or carry ``pre``).
    """
    long = two_period_panel(design)
    x_cols = list(design.x.columns)
    data = dml.DoubleMLPanelData(
        long, y_col="_y", d_cols="_g", t_col="_t", id_col="_id", x_cols=x_cols
    )
    ml_g, ml_m = learners(kind, fast, seed)
    _set_seed(seed)
    model = DoubleMLDIDBinary(
        data,
        g_value=1,
        t_value_pre=0,
        t_value_eval=1,
        ml_g=ml_g,
        ml_m=ml_m,
        n_folds=n_folds,
        ps_processor_config=PSProcessorConfig(clipping_threshold=clip),
    )
    model.fit(store_predictions=True)
    ps = model.predictions["ml_m"][:, 0, 0]
    d = design.d.to_numpy()
    return Estimate(
        design.name,
        float(model.coef[0]),
        float(model.se[0]),
        len(d),
        int(d.sum()),
        overlap_summary(ps, d, clip),
        model,
    )


def estimate_all(
    wide: pd.DataFrame,
    kind: str = "lgbm",
    fast: bool = False,
    seed: int = 0,
    include_did_lags: bool = True,
) -> pd.DataFrame:
    """Naive, unconfoundedness DML and DiD DML on one wide frame, as a table."""
    kw = {"kind": kind, "fast": fast, "seed": seed}
    unc = unconf_design(wide)
    results = [
        naive(unc),
        dml_unconf(unc, **kw),
        dml_did(did_design(wide), **kw),
    ]
    if include_did_lags:
        results.append(dml_did(did_design(wide, include_early_lags=True), **kw))
    return pd.DataFrame([r.as_row() for r in results])
