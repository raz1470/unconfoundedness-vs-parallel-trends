# Unconfoundedness vs Parallel Trends

[![CI](https://github.com/raz1470/unconfoundedness-vs-parallel-trends/actions/workflows/ci.yml/badge.svg)](https://github.com/raz1470/unconfoundedness-vs-parallel-trends/actions/workflows/ci.yml)

**Which assumption fits your self-selected treatment? Simulations and a pre-period diagnostic to help you choose.**

- The estimator (DML, doubly robust) is a separate choice from the identifying assumption. Picking the assumption is the decision that matters.
- Unconfoundedness with lagged outcomes fails when an unobserved stable trait drives selection. DiD fails when users select in after a transitory dip, because it credits the rebound to the treatment.
- The pre-period data can tell you which failure you face. When it cannot, the two estimates bound the truth.

![Top: the DiD pre-trend check by scenario. Bottom: Monte Carlo spread of the unconfoundedness and DiD estimates against the true effect.](https://raw.githubusercontent.com/raz1470/unconfoundedness-vs-parallel-trends/main/assets/hero.png)

Companion repo to the article *Unconfoundedness vs Parallel Trends: Which Assumption Fits Your Self-Selected Treatment?* The full walkthrough is in [the notebook](notebooks/unconfoundedness_vs_parallel_trends.ipynb).

## Quick start

```bash
git clone https://github.com/raz1470/unconfoundedness-vs-parallel-trends
cd unconfoundedness-vs-parallel-trends
uv sync
uv run pytest
```

Simulate a scenario, estimate both designs and run the diagnostic:

```python
from uvpt import scenario_config, simulate
from uvpt.diagnostics import diagnose
from uvpt.estimators import estimate_all
from uvpt.features import to_wide

res = simulate(scenario_config("c", seed=1))  # selection on a transitory dip
wide = to_wide(res.panel)

estimate_all(wide)           # naive, unconfoundedness DML, DiD DML, DiD with lags
diagnose(wide)["reading"]    # which assumption the pre-period data supports
res.true_att()               # 2.0
```

On your own data, pass a long panel with one row per user and event time:

```python
wide = to_wide(panel, user="user_id", time="event_time", treated="treated",
               outcome="sessions", covariates=["tenure", "platform"], week="week")
```

## The two designs

| | Unconfoundedness | Conditional DiD |
|---|---|---|
| Outcome | mean engagement, t0 to t+3 | post mean minus pre mean (t-4 to t-1) |
| Covariates | engagement t-8 to t-1, slope, recent change, baseline covariates | baseline covariates only |
| Estimator | `DoubleMLIRM`, ATTE score | `DoubleMLDIDBinary`, two-period panel |
| Fails when | a stable unobserved trait drives selection | selection follows a transitory dip |

Do not add outcome lags to the DiD. Matching on pre-period outcomes induces regression to the mean (Daw and Hatfield 2018). The repo's `did_lags` variant shows the cost.

## Decision rule

`diagnose` runs two pre-period checks. Each flags when any week in t-4 to t-1 has |z| > 2.5.

- **DiD pre-trend check.** Weight controls on baseline covariates only and compare each user's change since t-8 to t-5.
- **Unconfoundedness placebo.** Fit the propensity on the history through t-5 and compare t-4 to t-1.

| Pre-trend check | Placebo | Use |
|---|---|---|
| flat | flat | either design (they should agree) |
| flat | drifts | DiD |
| dips | either | unconfoundedness, with DiD reported as the other bound |

Weighted balance on the features the propensity was fitted on is not a test. It is close to perfect by construction.

## Simulation results

200 replications per scenario, 10k users each, true ATT 2.0.

| Scenario | Unconfoundedness bias | DiD bias | DiD + lags bias | Coverage (unconf / DiD) | Pre-trend check flags |
|---|---|---|---|---|---|
| a. Observables only | +0.01 | +0.00 | +0.00 | 94% / 93% | 3% |
| b. Stable trait | **-0.70** | +0.00 | **-0.68** | 0% / 94% | 3% |
| c. Transitory dip | -0.02 | **+1.01** | +0.24 | 94% / 0% | 100% |
| d. Both | **-0.73** | **+0.81** | -0.39 | 1% / 0% | 100% |

In scenario d the truth sits between the two designs in every replication.

Regenerate with `uv run python tools/run_montecarlo.py --diagnostics` (about two hours on two cores) or `--fast-mode` for a quick check.

## Repo layout

- `src/uvpt/simulate.py`: weekly panel with four selection scenarios and known potential outcomes.
- `src/uvpt/features.py`: wide frame, feature sets and outcomes for each design.
- `src/uvpt/estimators.py`: naive, unconfoundedness DML and DiD DML, with overlap reporting.
- `src/uvpt/diagnostics.py`: event-time plots, balance, pre-trend check, placebo, sensitivity, `diagnose`.
- `src/uvpt/montecarlo.py`: resumable Monte Carlo with bias, RMSE and coverage.
- `notebooks/`: the executed walkthrough. Built from `tools/build_notebook.py`.
- `results/`: Monte Carlo output behind section 8 of the notebook.

## Licence

MIT
