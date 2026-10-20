"""Run or resume the Monte Carlo and print the summary tables.

    uv run python tools/run_montecarlo.py                 # 200 reps, 10k users
    uv run python tools/run_montecarlo.py --fast-mode     # 20 reps, 5k users
    uv run python tools/run_montecarlo.py --max-reps 40   # stop after 40 new reps

Results go to results/ (or --out). Re-running skips finished replications.
"""

import argparse

import pandas as pd

from uvpt.montecarlo import rejection_rates, run, summarise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--n-users", type=int, default=10_000)
    p.add_argument("--scenarios", default="abcd")
    p.add_argument("--out", default="results")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--max-reps", type=int, default=None)
    p.add_argument("--full-learners", action="store_true")
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--fast-mode", action="store_true")
    a = p.parse_args()
    reps, n_users = (20, 5_000) if a.fast_mode else (a.reps, a.n_users)
    res = run(
        scenarios=list(a.scenarios),
        reps=reps,
        out_dir=a.out,
        n_users=n_users,
        fast=not a.full_learners,
        with_diagnostics=a.diagnostics,
        n_jobs=a.n_jobs,
        max_reps=a.max_reps,
    )
    pd.set_option("display.width", 160)
    s = summarise(res[~res["method"].isin(["pretrend_gap"])])
    print(s.round(3).to_string(index=False))
    if a.diagnostics:
        print(rejection_rates(res).round(3))


if __name__ == "__main__":
    main()
