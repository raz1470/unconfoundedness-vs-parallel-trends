"""Tests for diagnostics.py and plots.py."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from uvpt import scenario_config, simulate
from uvpt.diagnostics import (
    att_weights,
    balance_table,
    event_time_gaps,
    event_time_means,
    fit_propensity,
    placebo_event_time,
    placebo_test,
    pretrend_event_time,
    propensity_features,
    sensitivity,
    weighted_mean_se,
)
from uvpt.estimators import dml_did, naive
from uvpt.features import did_design, to_wide, unconf_design, ycols
from uvpt.plots import (
    plot_balance,
    plot_estimates,
    plot_event_means,
    plot_gaps,
    plot_overlap,
    use_style,
)


def _wide(scenario, n=6_000, seed=5):
    return to_wide(simulate(scenario_config(scenario, n_users=n, seed=seed)).panel)


@pytest.fixture(scope="module")
def wide_a():
    return _wide("a")


@pytest.fixture(scope="module")
def wide_b():
    return _wide("b")


@pytest.fixture(scope="module")
def wide_c():
    return _wide("c")


class TestWeightedMean:
    def test_equal_weights_match_plain_mean(self):
        y = np.array([1.0, 2.0, 4.0, 7.0])
        m, se = weighted_mean_se(y, np.ones(4))
        assert m == pytest.approx(y.mean())
        assert se == pytest.approx(y.std(ddof=0) / np.sqrt(4))

    def test_att_weights(self):
        ps = pd.Series([0.5, 0.2, 0.999])
        d = pd.Series([1, 0, 0])
        w = att_weights(ps, d, clip=0.01)
        assert w.tolist() == pytest.approx([1.0, 0.25, 99.0])


class TestEventTime:
    def test_unweighted_means_match_groupby(self, wide_a):
        m = event_time_means(wide_a)
        got = m[(m.treated == 1) & (m.event_time == -3)]["mean"].item()
        assert got == pytest.approx(wide_a.loc[wide_a.treated == 1, "y_m3"].mean())
        assert len(m) == 2 * 13

    def test_gaps(self, wide_a):
        m = event_time_means(wide_a)
        g = event_time_gaps(m).set_index("event_time")
        t = m[(m.treated == 1) & (m.event_time == 0)]["mean"].item()
        c = m[(m.treated == 0) & (m.event_time == 0)]["mean"].item()
        assert g.loc[0, "gap"] == pytest.approx(t - c)
        assert (g["ci_high"] > g["ci_low"]).all()


class TestBalance:
    def test_weighting_is_mechanical(self, wide_c):
        x = propensity_features(wide_c, cutoff=-1)
        ps = fit_propensity(x, wide_c["treated"])
        bal = balance_table(x, wide_c["treated"], att_weights(ps, wide_c["treated"]))
        assert bal["smd_raw"].abs().max() > 0.5
        assert bal["smd_weighted"].abs().max() < 0.1

    def test_smd_toy(self):
        x = pd.DataFrame({"a": [1.0, 3.0, 0.0, 2.0]})
        d = pd.Series([1, 1, 0, 0])
        bal = balance_table(x, d)
        assert bal.loc["a", "smd_raw"] == pytest.approx(1.0 / np.sqrt(2.0))


class TestPropensity:
    def test_features_respect_cutoff(self, wide_a):
        x = propensity_features(wide_a, cutoff=-5)
        assert not set(ycols(-4, 4)) & set(x.columns)

    def test_cross_fitted_scores_in_unit_interval(self, wide_a):
        ps = fit_propensity(propensity_features(wide_a), wide_a["treated"])
        assert ps.between(0, 1).all()
        assert ps.index.equals(wide_a.index)


class TestPlacebo:
    def test_passes_in_a(self, wide_a):
        out = placebo_test(wide_a, fast=True)
        assert not out["rejects_zero"].any()

    def test_did_placebo_passes_unconf_fails_in_b(self, wide_b):
        out = placebo_test(wide_b, fast=True).set_index("method")
        assert not out.loc["did_placebo", "rejects_zero"]
        assert out.loc["unconf_placebo", "estimate"] < 0
        assert out.loc["unconf_placebo", "rejects_zero"]

    def test_did_placebo_fails_in_c(self, wide_c):
        out = placebo_test(wide_c, fast=True).set_index("method")
        assert out.loc["did_placebo", "rejects_zero"]
        assert out.loc["did_placebo", "estimate"] < 0

    def test_event_time_version(self, wide_c):
        out = placebo_event_time(wide_c)
        assert out["placebo_mean_gap"] < 0
        assert out["placebo_max_abs_z"] > 3
        assert set(out) >= {"propensity", "weights", "gaps"}


class TestPretrend:
    def test_flat_in_b(self, wide_b):
        out = pretrend_event_time(wide_b)
        assert out["pretrend_max_abs_z"] < 3

    def test_dip_in_c(self, wide_c):
        out = pretrend_event_time(wide_c)
        assert out["pretrend_mean_gap"] < 0
        assert out["pretrend_max_abs_z"] > 3

    def test_reference_window_is_zero_on_average(self, wide_a):
        g = pretrend_event_time(wide_a)["gaps"].set_index("event_time")
        assert g.loc[-8:-5, "gap"].mean() == pytest.approx(0, abs=1e-10)


class TestSensitivity:
    def test_bounds_bracket_estimate(self, wide_a):
        est = dml_did(did_design(wide_a), fast=True)
        s = sensitivity(est, cf_y=0.03, cf_d=0.03)
        assert s["theta_lower"] < est.estimate < s["theta_upper"]
        assert s["ci_lower"] < s["theta_lower"]
        assert 0 < s["rv"] < 1

    def test_needs_dml_model(self, wide_a):
        with pytest.raises(ValueError, match="DML"):
            sensitivity(naive(unconf_design(wide_a)))


class TestPlots:
    def test_all_plots_draw(self, wide_a):
        use_style()
        fig, axes = plt.subplots(1, 5)
        m = event_time_means(wide_a)
        plot_event_means(m, ax=axes[0], title="means")
        plot_gaps(event_time_gaps(m), ax=axes[1])
        x = propensity_features(wide_a)
        ps = fit_propensity(x, wide_a["treated"])
        plot_overlap(ps, wide_a["treated"], ax=axes[2])
        plot_balance(balance_table(x, wide_a["treated"]), ax=axes[3])
        table = pd.DataFrame(
            {
                "method": ["a", "b"],
                "estimate": [1.0, 2.0],
                "ci_low": [0.5, 1.5],
                "ci_high": [1.5, 2.5],
            }
        )
        plot_estimates(table, truth=2.0, ax=axes[4], order=["b", "a"])
        assert all(ax.has_data() for ax in axes)
        plt.close(fig)

    def test_creates_axes_when_missing(self, wide_a):
        ax = plot_gaps(event_time_gaps(event_time_means(wide_a)))
        assert ax.has_data()
        plt.close(ax.figure)
