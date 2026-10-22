# Unconfoundedness vs Parallel Trends

[![CI](https://github.com/raz1470/unconfoundedness-vs-parallel-trends/actions/workflows/ci.yml/badge.svg)](https://github.com/raz1470/unconfoundedness-vs-parallel-trends/actions/workflows/ci.yml)

**Which assumption fits your self-selected treatment? Simulations and a pre-period diagnostic to help you choose.**

- The estimator (DML, doubly robust) is a separate choice from the identifying assumption. Picking the assumption is the decision that matters.
- Lagged-outcome regression (unconfoundedness given past outcomes) fails when an unobserved time-invariant confounder drives selection. DiD (parallel trends) fails when users select in after a temporary dip, because it credits the rebound to the treatment.
- A pre-trend check tells you whether DiD is safe. When it is not, report lagged-outcome regression with DiD as the other bound.

![Top: the DiD pre-trend check by scenario. Bottom: Monte Carlo spread of the lagged-outcome and DiD estimates against the true effect.](https://raw.githubusercontent.com/raz1470/unconfoundedness-vs-parallel-trends/main/assets/hero.png)

Companion repo to the article [*Difference-in-Differences vs Lagged-Outcome Regression*](https://raz1470.github.io/unconfoundedness-vs-parallel-trends/article.html). The full walkthrough is in [the notebook](notebooks/unconfoundedness_vs_parallel_trends.ipynb).

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

res = simulate(scenario_config("c", seed=1))  # selection on a temporary dip
wide = to_wide(res.panel)

estimate_all(wide)           # naive, lagged-outcome DML, DiD DML, DiD with lags
diagnose(wide)["reading"]    # which assumption the pre-period data supports
res.true_att()               # 2.0
```

On your own data, pass a long panel with one row per user and event time:

```python
wide = to_wide(panel, user="user_id", time="event_time", treated="treated",
               outcome="sessions", covariates=["tenure", "platform"], week="week")
```

## The two designs

The code calls lagged-outcome regression `unconf`, after its assumption.

| | Lagged-outcome regression | Conditional DiD |
|---|---|---|
| Outcome | mean engagement, t0 to t+3 | post mean minus pre mean (t-4 to t-1) |
| Covariates | engagement t-8 to t-1, slope, recent change, baseline covariates | baseline covariates only |
| Estimator | `DoubleMLIRM`, ATTE score | `DoubleMLDIDBinary`, two-period panel |
| Assumes | unconfoundedness given past outcomes and covariates | parallel trends given covariates |
| Fails when | a time-invariant confounder drives selection | selection follows a temporary dip |

Do not add outcome lags to the DiD. Matching on pre-period outcomes induces regression to the mean (Daw and Hatfield 2018). The repo's `did_lags` variant shows the cost.

## Decision rule

- **Flat pre-trend → DiD.** It copes with a time-invariant confounder, which lagged-outcome regression cannot.
- **Pre-trend dips → lagged-outcome regression, with DiD as the other bound.** The dip rules out DiD, but nothing in the pre-period rules out a time-invariant confounder as well.

The pre-trend check weights controls on baseline covariates only and compares each user's change since t-8 to t-5 over t-4 to t-1. It flags when any of those weeks has |z| > 2.5. A flat pre-trend is evidence for parallel trends, not proof (Roth 2022).

`diagnose` also runs a placebo for lagged-outcome regression (propensity fitted on the history through t-5, compared over t-4 to t-1). In the simulation it flagged b, c and d alike, so it does not change the decision. Its `reading` still reports both checks.

Weighted balance on the features the propensity was fitted on is not a test. It is close to perfect by construction.

## Simulation results

200 replications per scenario, 10k users each, true ATT 2.0.

| Scenario | Lagged-outcome bias | DiD bias | DiD + lags bias | Coverage (lagged-outcome / DiD) | Pre-trend check flags |
|---|---|---|---|---|---|
| a. Observables only | +0.01 | +0.00 | +0.00 | 94% / 93% | 3% |
| b. Time-invariant confounder | **-0.70** | +0.00 | **-0.68** | 0% / 94% | 3% |
| c. Temporary dip | -0.02 | **+1.01** | +0.24 | 94% / 0% | 100% |
| d. Both | **-0.73** | **+0.81** | -0.39 | 1% / 0% | 100% |

In scenario d the truth sits between the two designs in every replication. That is a property of this simulation: Ding and Li (2019) prove the bracket only when one of the two assumptions holds.

Regenerate with `uv run python tools/run_montecarlo.py --diagnostics` (about two hours on two cores) or `--fast-mode` for a quick check.

## Repo layout

- `src/uvpt/simulate.py`: weekly panel with four selection scenarios and known potential outcomes.
- `src/uvpt/features.py`: wide frame, feature sets and outcomes for each design.
- `src/uvpt/estimators.py`: naive, lagged-outcome DML and DiD DML, with overlap reporting.
- `src/uvpt/diagnostics.py`: event-time plots, balance, pre-trend check, placebo, sensitivity, `diagnose`.
- `src/uvpt/montecarlo.py`: resumable Monte Carlo with bias, RMSE and coverage.
- `notebooks/`: the executed walkthrough. Built from `tools/build_notebook.py`.
- `docs/article.html`: the article, served by GitHub Pages. Data from `tools/article_data.py`, then `tools/bake_article.py` writes the numbers and charts into the HTML.
- `results/`: Monte Carlo output behind section 8 of the notebook.

## Licence

MIT
