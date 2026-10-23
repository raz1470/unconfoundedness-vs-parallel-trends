"""Compute every number and series the article uses and inject it into the page.

The article's charts are drawn in the browser from one JSON object embedded
between the DATA markers in docs/article.html. This script rebuilds that
object from the simulator (seed 1, 10k users, the same draws as the notebook)
and from the Monte Carlo output in results/.

    uv run python tools/article_data.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from uvpt import scenario_config, simulate
from uvpt.diagnostics import (
    att_weights,
    balance_table,
    diagnose,
    event_time_means,
    fit_propensity,
    propensity_features,
)
from uvpt.estimators import estimate_all
from uvpt.features import baseline_features, to_wide, window_mean
from uvpt.montecarlo import load, rejection_rates, summarise
from uvpt.simulate import EVENT_TIMES

SEED = 1
N_USERS = 10_000
ARTICLE = Path("docs/article.html")
MARKERS = re.compile(r"(/\*DATA_START\*/).*?(/\*DATA_END\*/)", re.S)
METHODS = ["naive_unconf", "unconf", "did", "did_lags"]


def r(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def rl(xs, n=3):
    return [r(x, n) for x in xs]


def group_means(wide, weights=None):
    m = event_time_means(wide, weights)
    out = {}
    for g, name in ((1, "treated"), (0, "control")):
        q = m[m["treated"] == g].sort_values("event_time")
        out[name] = {
            "mean": rl(q["mean"]),
            "lo": rl(q["ci_low"]),
            "hi": rl(q["ci_high"]),
        }
    return out


def gaps_series(diag_part):
    g = diag_part["gaps"].sort_values("event_time")
    return {"gap": rl(g["gap"]), "lo": rl(g["ci_low"]), "hi": rl(g["ci_high"])}


def counterfactuals(res, wide):
    """Treated actual, true Y0 and the counterfactual each design implies."""
    d = wide["treated"]
    o = res.oracle[res.oracle["treated"] == 1]
    y0 = o.groupby("event_time")["y0"].mean().reindex(EVENT_TIMES)
    actual = wide.loc[d == 1, [c for c in wide if c.startswith("y_")]].mean()
    # Unconfoundedness: controls weighted on the history through t-1.
    ps_u = fit_propensity(propensity_features(wide, -1), d)
    w_u = att_weights(ps_u, d)
    unconf_cf = group_means(wide, w_u)["control"]["mean"]
    # DiD: treated pre level plus the covariate-weighted control change.
    ps_d = fit_propensity(baseline_features(wide), d)
    w_d = att_weights(ps_d, d)
    ctrl = np.array(group_means(wide, w_d)["control"]["mean"])
    pre = (EVENT_TIMES >= -4) & (EVENT_TIMES <= -1)
    t_pre = window_mean(wide[d == 1], (-4, -1)).mean()
    did_cf = t_pre + ctrl - ctrl[pre].mean()
    return {
        "actual": rl(actual.to_numpy()),
        "true_y0": rl(y0.to_numpy()),
        "unconf_cf": unconf_cf,
        "did_cf": rl(did_cf),
    }


def single_draws():
    out = {}
    for s in "abcd":
        res = simulate(scenario_config(s, n_users=N_USERS, seed=SEED))
        wide = to_wide(res.panel)
        dg = diagnose(wide)
        out[s] = {
            "truth": r(res.true_att()),
            "means": group_means(wide),
            "cf": counterfactuals(res, wide),
            "pretrend": gaps_series(dg["pretrend"]),
            "placebo": gaps_series(dg["placebo"]),
            "pretrend_z": r(dg["pretrend"]["pretrend_max_abs_z"], 1),
            "placebo_z": r(dg["placebo"]["placebo_max_abs_z"], 1),
            "flags": [dg["pretrend_flag"], dg["placebo_flag"]],
            "n_treated": int(wide["treated"].sum()),
            "n": len(wide),
        }
        if s == "c":
            hook = estimate_all(wide).set_index("method")
            out[s]["hook"] = {
                m: {k: r(hook.loc[m, k]) for k in ("estimate", "ci_low", "ci_high")}
                for m in ("unconf", "did")
            }
            x = propensity_features(wide, -1)
            ps = fit_propensity(x, wide["treated"])
            bal = balance_table(x, wide["treated"], att_weights(ps, wide["treated"]))
            out[s]["balance"] = {
                "features": list(bal.index),
                "raw": rl(bal["smd_raw"]),
                "weighted": rl(bal["smd_weighted"]),
            }
    return out


def monte_carlo():
    mc = load("results")
    if mc.empty:
        return None
    est = mc[mc["method"].isin(METHODS)]
    summ = summarise(est).set_index(["scenario", "method"])
    out = {"reps": {}, "methods": {}, "rejection": {}, "bracket_d": None}
    for s in sorted(est["scenario"].unique()):
        e = est[est["scenario"] == s]
        out["reps"][s] = int(e["rep"].nunique())
        out["methods"][s] = {}
        for m in METHODS:
            x = e.loc[e["method"] == m, "estimate"]
            row = summ.loc[(s, m)]
            out["methods"][s][m] = {
                "mean": r(x.mean()),
                "q": rl(x.quantile([0.025, 0.25, 0.75, 0.975])),
                "bias": r(row["bias"]),
                "rmse": r(row["rmse"]),
                "coverage": r(row["coverage"]),
                "sd": r(row["sd"]),
            }
    rr = rejection_rates(mc)
    out["rejection"] = {s: {k: r(v) for k, v in rr.loc[s].items()} for s in rr.index}
    # Mean over weeks -4 to -1 (truth 0) for the two pre-period checks.
    out["window_mean"] = {
        m: {
            s: r(g["estimate"].mean())
            for s, g in mc[mc["method"] == m].groupby("scenario")
        }
        for m in ("pretrend_gap", "unconf_placebo")
    }
    d = est[est["scenario"] == "d"]
    if not d.empty:
        p = d.pivot(index="rep", columns="method", values="estimate")
        t = d.groupby("rep")["truth"].first()
        lo, hi = p[["unconf", "did"]].min(axis=1), p[["unconf", "did"]].max(axis=1)
        out["bracket_d"] = r(((lo < t) & (t < hi)).mean())
    return out


def main():
    data = {
        "event_times": EVENT_TIMES.tolist(),
        "single": single_draws(),
        "mc": monte_carlo(),
    }
    blob = json.dumps(data, separators=(",", ":"))
    html = ARTICLE.read_text()
    new, n = MARKERS.subn(lambda m: f"{m.group(1)}{blob}{m.group(2)}", html)
    if n != 1:
        raise SystemExit("DATA markers not found exactly once in docs/article.html")
    ARTICLE.write_text(new)
    print(f"injected {len(blob):,} bytes into {ARTICLE}")


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    main()
