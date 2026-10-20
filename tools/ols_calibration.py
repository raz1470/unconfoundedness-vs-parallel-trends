"""OLS check that each scenario breaks the intended design.

Runs plain OLS versions of both designs over a few seeds per scenario and
prints mean bias against the true ATT. Run before any DML work: if the
scenarios do not separate here, flexible nuisances will not fix it.

    uv run python tools/ols_calibration.py [n_seeds]
"""

import sys

import numpy as np
import pandas as pd

from uvpt import scenario_config, simulate


def wide(panel):
    w = panel.pivot(index="user_id", columns="event_time", values="y")
    u = panel.groupby("user_id")[["treated", "x1", "x2", "x3"]].first()
    return w, u


def ols_coef(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(X, y, rcond=None)[0][1]


def estimates(res):
    w, u = wide(res.panel)
    d = u["treated"].to_numpy()
    x = u[["x1", "x2", "x3"]].to_numpy()
    post = w.loc[:, 0:3].mean(axis=1).to_numpy()
    pre = w.loc[:, -4:-1].mean(axis=1).to_numpy()
    lags = w.loc[:, -8:-1].to_numpy()
    early = w.loc[:, -8:-5].to_numpy()
    return {
        "naive": post[d == 1].mean() - post[d == 0].mean(),
        "unconf_ols": ols_coef(post, np.column_stack([d, lags, x])),
        "did_ols": ols_coef(post - pre, np.column_stack([d, x])),
        "did_ols_lags": ols_coef(post - pre, np.column_stack([d, x, early])),
    }


def main(n_seeds=5):
    rows = []
    for s in "abcd":
        for seed in range(n_seeds):
            res = simulate(scenario_config(s, seed=seed))
            est = estimates(res)
            truth = res.true_att()
            rows += [
                {"scenario": s, "method": m, "bias": v - truth, "truth": truth}
                for m, v in est.items()
            ]
    df = pd.DataFrame(rows)
    out = df.pivot_table(index="scenario", columns="method", values="bias")
    print(out.round(3))
    print("truth mean", round(df["truth"].mean(), 3))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
