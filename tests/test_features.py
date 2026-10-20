"""Tests for features.py, including no-leakage checks on feature timing."""

import numpy as np
import pandas as pd
import pytest

from uvpt import scenario_config, simulate
from uvpt.features import (
    Design,
    apply_transform,
    did_design,
    history_features,
    placebo_designs,
    to_wide,
    unconf_design,
    window_mean,
    winsoriser,
    ycol,
    ycols,
)


@pytest.fixture(scope="module")
def res():
    return simulate(scenario_config("c", n_users=1_000, seed=11))


@pytest.fixture(scope="module")
def wide(res):
    return to_wide(res.panel)


def _blank_from(wide, k):
    """Copy of ``wide`` with every outcome at event time ``k`` or later set to NaN."""
    out = wide.copy()
    out[ycols(k, 4)] = np.nan
    return out


class TestNames:
    def test_ycol(self):
        assert [ycol(-8), ycol(-1), ycol(0), ycol(3)] == [
            "y_m8",
            "y_m1",
            "y_p0",
            "y_p3",
        ]


class TestWide:
    def test_one_row_per_user(self, res, wide):
        assert len(wide) == res.panel["user_id"].nunique()
        assert set(ycols(-8, 4)) <= set(wide.columns)

    def test_index_week_is_week_at_t0(self, res, wide):
        p = res.panel
        t0 = p[p["event_time"] == 0].set_index("user_id")["week"]
        assert wide["index_week"].equals(t0)

    def test_values_round_trip(self, res, wide):
        p = res.panel
        row = p[(p["user_id"] == wide.index[0]) & (p["event_time"] == -3)]
        assert wide.iloc[0]["y_m3"] == row["y"].item()

    def test_without_week(self, res):
        w = to_wide(res.panel.drop(columns="week"), week=None)
        assert "index_week" not in w
        assert "season_sin" not in unconf_design(w).x


class TestNoLeakage:
    def test_unconf_uses_nothing_from_t0(self, wide):
        a = unconf_design(wide).x
        b = unconf_design(_blank_from(wide, 0)).x
        pd.testing.assert_frame_equal(a, b)
        assert np.isfinite(a.to_numpy()).all()

    def test_did_uses_no_outcomes(self, wide):
        a = did_design(wide).x
        b = did_design(_blank_from(wide, -8)).x
        pd.testing.assert_frame_equal(a, b)
        assert not any(c.startswith("y_") for c in a.columns)

    def test_did_early_lags_stop_at_t_minus_5(self, wide):
        a = did_design(wide, include_early_lags=True).x
        b = did_design(_blank_from(wide, -4), include_early_lags=True).x
        pd.testing.assert_frame_equal(a, b)
        assert set(ycols(-8, -5)) <= set(a.columns)

    def test_placebo_features_stop_at_t_minus_5(self, wide):
        a = placebo_designs(wide)
        b = placebo_designs(_blank_from(wide, -4))
        for key in a:
            pd.testing.assert_frame_equal(a[key].x, b[key].x)

    def test_placebo_outcome_is_pre_treatment(self, wide):
        a = placebo_designs(wide)
        b = placebo_designs(_blank_from(wide, 0))
        for key in a:
            pd.testing.assert_series_equal(a[key].y, b[key].y)


class TestOutcomes:
    def test_unconf_outcome_is_post_mean(self, wide):
        y = unconf_design(wide).y
        assert np.allclose(y, wide[ycols(0, 3)].mean(axis=1))

    def test_did_outcome_is_post_minus_pre(self, wide):
        y = did_design(wide).y
        expected = wide[ycols(0, 3)].mean(axis=1) - wide[ycols(-4, -1)].mean(axis=1)
        assert np.allclose(y, expected)

    def test_window_mean_bounds_inclusive(self, wide):
        assert np.allclose(window_mean(wide, (-2, -2)), wide["y_m2"])

    def test_winsoriser_caps(self, res, wide):
        f = winsoriser(res.panel, q=0.9)
        cap = res.panel["y"].quantile(0.9)
        assert f(wide[ycols(0, 3)].to_numpy()).max() <= cap


class TestHistory:
    def test_slope_of_linear_series(self):
        w = pd.DataFrame({ycol(k): [2.0 * k + 5] for k in range(-8, 0)})
        h = history_features(w)
        assert h["slope"].item() == pytest.approx(2.0)
        assert h["recent_change"].item() == pytest.approx(2.0 * (-1.5 + 6.5))

    def test_short_history_for_placebo(self, wide):
        h = history_features(wide, cutoff=-5)
        assert [c for c in h.columns if c.startswith("y_")] == ycols(-8, -5)


class TestDesign:
    def test_index_mismatch_raises(self, wide):
        d = unconf_design(wide)
        with pytest.raises(ValueError, match="index"):
            Design("bad", d.y.iloc[1:], d.d, d.x)

    def test_names(self, wide):
        assert unconf_design(wide).name == "unconf"
        assert did_design(wide).name == "did"
        assert did_design(wide, include_early_lags=True).name == "did_lags"


class TestTransform:
    def test_applies_to_outcomes_only(self, wide):
        out = apply_transform(wide, np.log1p)
        assert np.allclose(out["y_m1"], np.log1p(wide["y_m1"]))
        assert out["x1"].equals(wide["x1"])

    def test_none_is_identity(self, wide):
        assert apply_transform(wide, None) is wide
