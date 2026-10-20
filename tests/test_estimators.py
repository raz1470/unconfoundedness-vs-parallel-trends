"""Tests for estimators.py, including known-effect recovery in scenario a."""

import numpy as np
import pandas as pd
import pytest

from uvpt import scenario_config, simulate
from uvpt.estimators import (
    Estimate,
    dml_did,
    dml_unconf,
    estimate_all,
    learners,
    naive,
    overlap_summary,
    two_period_panel,
)
from uvpt.features import Design, did_design, to_wide, unconf_design

FAST = {"fast": True, "seed": 0}


@pytest.fixture(scope="module")
def easy():
    res = simulate(scenario_config("a", n_users=4_000, seed=5))
    return res, to_wide(res.panel)


@pytest.fixture(scope="module")
def table(easy):
    _, wide = easy
    return estimate_all(wide, **FAST).set_index("method")


class TestRecovery:
    @pytest.mark.parametrize("method", ["unconf", "did", "did_lags"])
    def test_scenario_a_recovers_truth(self, easy, table, method):
        res, _ = easy
        row = table.loc[method]
        assert abs(row["estimate"] - res.true_att()) < 3 * row["se"]

    def test_table_columns(self, table):
        assert {"estimate", "se", "ci_low", "ci_high", "n_treated"} <= set(table)
        assert "overlap_ess_controls" in table


class TestFailureModes:
    def test_did_overstates_in_c(self):
        res = simulate(scenario_config("c", n_users=8_000, seed=5))
        wide = to_wide(res.panel)
        did = dml_did(did_design(wide), **FAST)
        unc = dml_unconf(unconf_design(wide), **FAST)
        assert did.estimate - res.true_att() > 3 * did.se
        assert abs(unc.estimate - res.true_att()) < 3 * unc.se

    def test_unconf_understates_in_b(self):
        res = simulate(scenario_config("b", n_users=8_000, seed=5))
        wide = to_wide(res.panel)
        did = dml_did(did_design(wide), **FAST)
        unc = dml_unconf(unconf_design(wide), **FAST)
        assert res.true_att() - unc.estimate > 3 * unc.se
        assert abs(did.estimate - res.true_att()) < 3 * did.se
        nai = naive(unconf_design(wide))
        assert res.true_att() - nai.estimate > 10 * nai.se


class TestNaive:
    def test_matches_difference_in_means(self):
        idx = pd.RangeIndex(4)
        d = pd.Series([1, 1, 0, 0], index=idx)
        y = pd.Series([3.0, 5.0, 1.0, 2.0], index=idx)

        est = naive(Design("toy", y, d, pd.DataFrame(index=idx)))
        assert est.estimate == pytest.approx(2.5)
        assert est.se == pytest.approx(np.sqrt(2.0 / 2 + 0.5 / 2))


class TestOverlap:
    def test_summary_values(self):
        ps = np.array([0.9, 0.8, 0.5, 0.995, 0.2])
        d = np.array([1, 1, 0, 0, 0])
        out = overlap_summary(ps, d, clip=0.01)
        assert out["share_clipped"] == pytest.approx(1 / 3)
        w = np.array([1.0, 99.0, 0.25])
        assert out["ess_controls"] == pytest.approx(w.sum() ** 2 / (w**2).sum())
        assert out["max_weight_share"] == pytest.approx(99 / w.sum())


class TestPanel:
    def test_two_rows_per_user(self, easy):
        _, wide = easy
        design = did_design(wide)
        long = two_period_panel(design)
        assert len(long) == 2 * len(wide)
        post = long[long["_t"] == 1].set_index("_id")["_y"]
        assert np.allclose(post, design.pre + design.y)
        assert set(long.loc[long["_g"] != 1, "_g"]) == {np.inf}

    def test_missing_pre_raises(self, easy):
        _, wide = easy
        with pytest.raises(ValueError, match="pre"):
            dml_did(unconf_design(wide), **FAST)


class TestLearners:
    def test_hgb_runs(self, easy):
        _, wide = easy
        est = dml_unconf(unconf_design(wide), kind="hgb", fast=True, n_folds=2)
        assert np.isfinite(est.estimate)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            learners("forest")

    def test_same_seed_same_estimate(self, easy):
        _, wide = easy
        a = dml_did(did_design(wide), **FAST).estimate
        b = dml_did(did_design(wide), **FAST).estimate
        assert a == b


class TestEstimate:
    def test_ci_is_symmetric(self):
        e = Estimate("x", 1.0, 0.5, 10, 5)
        lo, hi = e.ci
        assert hi - 1.0 == pytest.approx(1.0 - lo)
        assert hi - lo == pytest.approx(2 * 1.959964 * 0.5, rel=1e-5)
