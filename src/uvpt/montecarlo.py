"""Monte Carlo over scenarios: bias, RMSE and coverage per method.

Results are appended to one CSV per scenario as each replication finishes,
and finished replications are skipped on the next call. A long run can be
stopped and resumed without losing work.

Replication ``r`` uses simulator seed ``r`` in every scenario, so scenarios
share random numbers and differences between them come from the selection
mechanism alone.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from uvpt.diagnostics import FLAG_Z, placebo_test, pretrend_event_time
from uvpt.estimators import estimate_all
from uvpt.features import to_wide
from uvpt.simulate import SCENARIOS, scenario_config, simulate


def run_one(
    scenario: str,
    rep: int,
    n_users: int = 10_000,
    fast: bool = True,
    with_diagnostics: bool = False,
    **sim_overrides,
) -> pd.DataFrame:
    """One replication: simulate, estimate every method, attach the truth.

    With ``with_diagnostics`` the placebo DML estimates and the pre-trend
    summary are added as extra rows (their truth is zero).
    """
    res = simulate(
        scenario_config(scenario, n_users=n_users, seed=rep, **sim_overrides)
    )
    wide = to_wide(res.panel)
    table = estimate_all(wide, fast=fast, seed=rep)
    table["truth"] = res.true_att()
    if with_diagnostics:
        plac = placebo_test(wide, fast=fast, seed=rep)
        plac["truth"] = 0.0
        pre = pretrend_event_time(wide, fast=fast, seed=rep)
        pt = pd.DataFrame(
            [
                {
                    "method": "pretrend_gap",
                    "estimate": pre["pretrend_mean_gap"],
                    "se": np.nan,
                    "truth": 0.0,
                    "max_abs_z": pre["pretrend_max_abs_z"],
                }
            ]
        )
        table = pd.concat([table, plac, pt], ignore_index=True)
    table.insert(0, "rep", rep)
    table.insert(0, "scenario", scenario)
    return table


def _path(out_dir: Path, scenario: str) -> Path:
    return out_dir / f"mc_{scenario}.csv"


def done_reps(out_dir: Path, scenario: str) -> set[int]:
    """Replications already saved for ``scenario``."""
    p = _path(out_dir, scenario)
    if not p.exists():
        return set()
    return set(pd.read_csv(p, usecols=["rep"])["rep"].unique())


def run(
    scenarios: Iterable[str] = SCENARIOS,
    reps: int = 200,
    out_dir: str | Path = "results",
    n_users: int = 10_000,
    fast: bool = True,
    with_diagnostics: bool = False,
    n_jobs: int = 1,
    batch: int = 10,
    max_reps: int | None = None,
    **sim_overrides,
) -> pd.DataFrame:
    """Run missing replications and return every saved result.

    ``batch`` replications run in parallel and are then appended to disk, so
    at most one batch is lost if the process is killed. ``max_reps`` caps the
    number of new replications in this call, for runs split across sessions.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    budget = np.inf if max_reps is None else max_reps
    for s in scenarios:
        todo = [r for r in range(reps) if r not in done_reps(out, s)]
        for i in range(0, len(todo), batch):
            if budget <= 0:
                break
            chunk = todo[i : i + batch][: int(min(batch, budget))]
            frames = Parallel(n_jobs=n_jobs)(
                delayed(run_one)(s, r, n_users, fast, with_diagnostics, **sim_overrides)
                for r in chunk
            )
            df = pd.concat(frames, ignore_index=True)
            p = _path(out, s)
            df.to_csv(p, mode="a", header=not p.exists(), index=False)
            budget -= len(chunk)
    return load(out, scenarios)


def load(out_dir: str | Path, scenarios: Iterable[str] = SCENARIOS) -> pd.DataFrame:
    """Read every saved replication for ``scenarios``."""
    out = Path(out_dir)
    frames = [pd.read_csv(_path(out, s)) for s in scenarios if _path(out, s).exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Bias, SD, RMSE, mean SE and 95% coverage per scenario and method."""
    r = results.copy()
    r["error"] = r["estimate"] - r["truth"]
    covered = (r["ci_low"] <= r["truth"]) & (r["truth"] <= r["ci_high"])
    r["covered"] = covered.astype(float).where(r["ci_low"].notna())
    g = r.groupby(["scenario", "method"], sort=False)
    out = pd.DataFrame(
        {
            "reps": g["rep"].nunique(),
            "truth": g["truth"].mean(),
            "mean_estimate": g["estimate"].mean(),
            "bias": g["error"].mean(),
            "sd": g["estimate"].std(),
            "rmse": np.sqrt(g["error"].apply(lambda e: np.mean(e**2))),
            "mean_se": g["se"].mean(),
            "coverage": g["covered"].mean(),
        }
    )
    out["mc_se_bias"] = out["sd"] / np.sqrt(out["reps"])
    return out.reset_index()


def rejection_rates(results: pd.DataFrame, z: float = FLAG_Z) -> pd.DataFrame:
    """Share of replications where each diagnostic flags a problem.

    Placebo DML rows flag when their 95% interval excludes zero. The pre-trend
    row flags when any event time in t-4 to t-1 has ``|z| > z``.
    """
    r = results
    plac = r[r["method"].str.endswith("_placebo")].copy()
    plac["flag"] = (plac["ci_low"] > 0) | (plac["ci_high"] < 0)
    pre = r[r["method"] == "pretrend_gap"].copy()
    pre["flag"] = pre["max_abs_z"] > z
    both = pd.concat([plac, pre])
    return (
        both.groupby(["scenario", "method"])["flag"]
        .mean()
        .unstack("method")
        .rename_axis(columns=None)
    )
