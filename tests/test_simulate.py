"""Tests for simulate.py."""

import numpy as np
import pytest

from uvpt.simulate import (
    COVARIATES,
    EVENT_TIMES,
    SCENARIOS,
    SimConfig,
    scenario_config,
    simulate,
)

SMALL = {"n_users": 2_000, "seed": 3}

# simulate(scenario_config("a", n_users=200, seed=7)): first three rows of y.
GOLDEN_Y = [38.1037341031447, 37.428171495438086, 36.82154423336742]
GOLDEN_N_TREATED = 43


@pytest.fixture(scope="module", params=SCENARIOS)
def result(request):
    return simulate(scenario_config(request.param, **SMALL))


class TestShapes:
    def test_one_row_per_user_and_event_time(self, result):
        panel = result.panel
        n = panel["user_id"].nunique()
        assert len(panel) == n * len(EVENT_TIMES)
        counts = panel.groupby("user_id")["event_time"].apply(list)
        assert all(c == list(EVENT_TIMES) for c in counts)

    def test_columns(self, result):
        expected = {"user_id", "event_time", "week", "treated", "y", *COVARIATES}
        assert set(result.panel.columns) == expected

    def test_oracle_aligned(self, result):
        cols = ["user_id", "event_time"]
        assert result.oracle[cols].equals(result.panel[cols])

    def test_weeks_inside_calendar(self, result):
        week = result.panel["week"]
        assert week.min() >= 0
        assert week.max() <= result.config.n_weeks - 1

    def test_treated_share(self, result):
        share = result.users["treated"].mean()
        assert share == pytest.approx(result.config.treated_share, abs=0.03)


class TestNoOracleLeak:
    def test_panel_has_no_potential_outcomes_or_trait(self, result):
        leaked = {"y0", "y1", "trait", "dip", "p_select"} & set(result.panel.columns)
        assert not leaked


class TestPotentialOutcomes:
    def test_observed_matches_assignment(self, result):
        p, o = result.panel, result.oracle
        treated = p["treated"] == 1
        assert np.allclose(p.loc[treated, "y"], o.loc[treated, "y1"])
        assert np.allclose(p.loc[~treated, "y"], o.loc[~treated, "y0"])

    def test_no_effect_before_index(self, result):
        o = result.oracle
        pre = o["event_time"] < 0
        assert np.allclose(o.loc[pre, "y1"], o.loc[pre, "y0"])

    def test_true_att_equals_tau_without_clipping(self, result):
        assert result.true_att() == pytest.approx(result.config.tau, abs=0.01)

    def test_true_att_on_log_scale_differs(self):
        res = simulate(scenario_config("a", **SMALL))
        assert res.true_att(np.log1p) < res.true_att()


class TestSelection:
    def _corr(self, res, col):
        u = res.users
        return np.corrcoef(u["treated"], u[col])[0, 1]

    def test_trait_ignored_in_a(self):
        res = simulate(scenario_config("a", **SMALL))
        assert abs(self._corr(res, "trait")) < 0.06

    def test_selection_in_c_is_on_observables_only(self):
        res = simulate(scenario_config("c", **SMALL))
        u = res.users
        logit = np.log(u["p_select"] / (1 - u["p_select"]))
        design = np.column_stack([np.ones(len(u)), u[["x1", "x2", "x3", "dip"]]])
        fitted = design @ np.linalg.lstsq(design, logit, rcond=None)[0]
        assert np.allclose(fitted, logit)

    def test_low_trait_selects_in_b(self):
        res = simulate(scenario_config("b", **SMALL))
        assert self._corr(res, "trait") < -0.15

    def test_low_recent_engagement_selects_in_c(self):
        res = simulate(scenario_config("c", **SMALL))
        assert self._corr(res, "dip") < -0.15

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError, match="scenario"):
            scenario_config("z")


class TestEligibility:
    def test_every_user_meets_rule(self):
        res = simulate(scenario_config("a", activity_threshold=25.0, **SMALL))
        p = res.panel
        pre = p[p["event_time"].between(-4, -1)]
        active = (pre["y"] > 25.0).groupby(pre["user_id"]).sum()
        assert (active >= res.config.min_active_weeks).all()
        assert res.users["eligible"].mean() < 1.0

    def test_same_rule_for_both_groups(self):
        res = simulate(scenario_config("a", activity_threshold=25.0, **SMALL))
        kept = res.users[res.users["eligible"]]
        assert set(kept["treated"]) == {0, 1}


class TestIndexDates:
    def test_resample_draws_from_treated_weeks(self):
        res = simulate(scenario_config("a", control_index="resample", **SMALL))
        u = res.users
        treated_weeks = set(u.loc[u["treated"] == 1, "index_week"])
        assert set(u.loc[u["treated"] == 0, "index_week"]) <= treated_weeks

    def test_bad_control_index_raises(self):
        with pytest.raises(ValueError, match="control_index"):
            simulate(SimConfig(control_index="random", n_users=50))


class TestHeavyTail:
    def test_right_skewed_and_positive(self):
        res = simulate(scenario_config("a", heavy_tail=True, **SMALL))
        y = res.panel["y"]
        assert (y >= 0).all()
        assert y.mean() > y.median() * 1.2


class TestDeterminism:
    def test_same_seed_same_panel(self):
        a = simulate(scenario_config("b", **SMALL)).panel
        b = simulate(scenario_config("b", **SMALL)).panel
        assert a.equals(b)

    def test_golden_values(self):
        res = simulate(scenario_config("a", n_users=200, seed=7))
        assert res.panel["y"].head(3).tolist() == pytest.approx(GOLDEN_Y)
        assert res.users["treated"].sum() == GOLDEN_N_TREATED
