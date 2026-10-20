"""Tests for montecarlo.py in fast mode."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from uvpt.montecarlo import (
    done_reps,
    load,
    rejection_rates,
    run,
    run_one,
    summarise,
)
from uvpt.plots import plot_mc

TINY = {"n_users": 1_500, "fast": True}


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    out = tmp_path_factory.mktemp("mc")
    res = run(scenarios=["a", "c"], reps=2, out_dir=out, **TINY)
    return out, res


class TestRun:
    def test_saves_every_rep_and_method(self, saved):
        out, res = saved
        assert done_reps(out, "a") == {0, 1}
        assert set(res["method"]) == {"naive_unconf", "unconf", "did", "did_lags"}
        assert len(res) == 2 * 2 * 4

    def test_resume_skips_finished(self, saved):
        out, _ = saved
        before = (out / "mc_a.csv").stat().st_size
        res = run(scenarios=["a"], reps=2, out_dir=out, **TINY)
        assert (out / "mc_a.csv").stat().st_size == before
        assert len(res) == 2 * 4

    def test_max_reps_caps_new_work(self, saved):
        out, _ = saved
        run(scenarios=["a"], reps=5, out_dir=out, max_reps=1, **TINY)
        assert done_reps(out, "a") == {0, 1, 2}

    def test_load_missing_dir_is_empty(self, tmp_path):
        assert load(tmp_path).empty


class TestSummary:
    def test_columns_and_values(self):
        res = pd.DataFrame(
            {
                "scenario": ["a"] * 4,
                "rep": [0, 1, 2, 3],
                "method": ["m"] * 4,
                "estimate": [1.0, 3.0, 2.0, 2.0],
                "se": [0.5] * 4,
                "ci_low": [0.0, 2.5, 1.0, 2.1],
                "ci_high": [2.0, 3.5, 3.0, 3.0],
                "truth": [2.0] * 4,
            }
        )
        s = summarise(res).iloc[0]
        assert s["bias"] == pytest.approx(0.0)
        assert s["rmse"] == pytest.approx(np.sqrt(0.5))
        assert s["coverage"] == pytest.approx(0.5)
        assert s["reps"] == 4


class TestDiagnosticsRows:
    def test_rejection_rates(self):
        rows = run_one("c", 0, with_diagnostics=True, **TINY)
        assert {"unconf_placebo", "did_placebo", "pretrend_gap"} <= set(rows["method"])
        rates = rejection_rates(rows.assign(scenario="c"))
        assert set(rates.columns) == {"did_placebo", "pretrend_gap", "unconf_placebo"}
        assert rates.notna().all().all()


class TestPlot:
    def test_plot_mc(self, saved):
        _, res = saved
        ax = plot_mc(res, "a", title="a", order=["did", "unconf"])
        assert len(ax.get_yticks()) == 2
        plt.close(ax.figure)
