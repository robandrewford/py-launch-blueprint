# PFNs Blueprint for Bayesian Marketing Analysis

This repository is a Python project template, not the PFNs codebase itself. The right way to use it for your goal is to treat it as the **control plane** around a PFNs-style model project:

- synthetic prior generation for marketing measurement tasks,
- evaluation harnesses against Bayesian baselines such as `pymc-marketing`, and
- continuous experimentation loops that generate hypotheses, run jobs, score results, and promote better configurations.

## What you are actually building

You want a foundation model that learns to do **Bayesian marketing analysis in context**. Concretely, that means training on batches sampled from priors that resemble common marketing measurement problems:

- media mix modeling (MMM),
- incrementality / lift estimation,
- budget allocation,
- marketing response curves with saturation and adstock,
- heterogeneous treatment effects across geographies or cohorts,
- seasonality, trend, and promotional shocks,
- noisy attribution and delayed conversion windows.

Instead of directly replacing `pymc-marketing`, the foundation model should learn to **approximate posterior-relevant predictions** or downstream decisions from many synthetic tasks generated from those priors.

## Recommended repository shape inside this template

If you evolve this template into a PFNs-style project, add modules like the following:

```text
py_launch_blueprint/
├── experiments/
│   ├── runners.py               # launch train/eval sweeps
│   ├── registry.py              # named experiments and checkpoints
│   └── scoring.py               # metric aggregation and promotion rules
├── marketing/
│   ├── schema.py                # canonical dataset definitions
│   ├── transforms.py            # adstock, saturation, lag features
│   ├── simulators.py            # synthetic marketing task generators
│   ├── priors.py                # PFNs-compatible get_batch functions
│   └── benchmarks.py            # compare PFNs outputs to PyMC baselines
├── agents/
│   ├── loop.py                  # continuous research / experiment loop
│   ├── planner.py               # propose next experiment or ablation
│   └── reporter.py              # summarize wins/regressions
└── projects.py                  # existing CLI example from the template
```

## PFNs-specific adaptation strategy

### 1. Define a canonical marketing task schema

Before writing priors, normalize every synthetic or real dataset into a single schema. A useful minimum schema is:

- `x_context`: historical observations available to the model,
- `y_context`: observed business outcome or latent target proxy,
- `x_query`: rows for which the model must predict,
- `metadata`: channel names, geo ids, calendars, spend units,
- `task_type`: MMM, lift, allocation, elasticity, forecasting, etc.

Recommended feature groups:

- media spend per channel,
- lagged spend windows,
- adstock-transformed features,
- saturation-transformed features,
- promotions and price,
- holidays and seasonal terms,
- geo / cohort identifiers,
- macro controls,
- measurement-noise indicators.

### 2. Build PFNs priors around real marketing structure

Your synthetic priors should mirror the latent assumptions in Bayesian marketing models. Start with three high-value priors.

#### A. MMM response prior

Generate datasets with latent parameters for:

- channel-specific adstock decay,
- channel-specific saturation,
- baseline trend and seasonality,
- observation noise,
- optional hierarchical pooling by geography.

The model input should include raw and transformed media features. Targets can include:

- posterior mean sales / conversions,
- predictive intervals,
- incremental contribution by channel,
- elasticity or ROI ranking.

#### B. Incrementality experiment prior

Generate synthetic test/control studies with:

- pre-period imbalance,
- staggered intervention timing,
- heterogeneous treatment effects,
- spillover and carryover,
- small-sample noise.

This teaches the model to reason about causal lift instead of only curve fitting.

#### C. Budget allocation prior

Sample latent response curves for multiple channels and train the model to predict:

- expected return under candidate spend vectors,
- regret relative to the true optimum,
- recommended budget ordering,
- uncertainty-aware allocation scores.

### 3. Use `pymc-marketing` as the teacher and benchmark

For many tasks, the best path is **teacher-student distillation** rather than trying to learn directly from raw historical labels alone.

A practical pattern:

1. Sample synthetic marketing datasets.
2. Fit a reference Bayesian model with `pymc-marketing` or a similar PyMC stack.
3. Extract supervision targets such as posterior means, predictive quantiles, lift distributions, or optimal budget recommendations.
4. Train the PFNs-style model to reproduce those outputs in context.

This gives you:

- grounded supervision from a strong Bayesian estimator,
- explicit control over the training distribution,
- better interpretability during debugging because PFNs errors can be compared to teacher posteriors.

### 4. Keep evaluation split by decision type

Do not rely on one scalar metric. Keep separate suites for:

- **posterior approximation**: NLL, calibration, interval coverage,
- **causal estimation**: lift error, sign accuracy, treatment ranking,
- **allocation**: regret, top-k budget recommendations, ROI ordering,
- **robustness**: performance under missingness, drift, sparse channels, delayed effects,
- **transfer**: synthetic-to-real generalization.

## Continuous experimentation loop (Karpathy-style research agent)

If you want an `autoresearch-macos`-style loop, the safest architecture is a constrained offline research loop rather than a fully autonomous online learner.

### Core loop

1. **Observe**
   - Collect latest experiment metrics, failures, and dataset drift summaries.
2. **Hypothesize**
   - Propose one change: prior mix, architecture knob, loss weighting, evaluator, or prompt/tooling change.
3. **Run**
   - Launch a bounded training or evaluation job.
4. **Score**
   - Compare against champion baselines with explicit promotion thresholds.
5. **Report**
   - Persist a markdown summary of what changed and why.
6. **Promote or reject**
   - Only bless a new checkpoint if it clears quality gates.

### Guardrails you should enforce

- fixed experiment budget per loop,
- immutable datasets and seeds for benchmark suites,
- promotion based on multiple metrics rather than a single win,
- mandatory human review before deployment to production workflows,
- artifact logging for data, configs, checkpoints, and reports.

### Good first automated actions

Your autonomous loop should initially mutate only:

- prior hyperparameter ranges,
- mixture weights across priors,
- sequence length / feature grouping,
- distillation target choice,
- loss weights across evaluation families.

Avoid allowing the loop to rewrite data schemas or deployment code until the benchmark harness is stable.

## Suggested first milestone plan

### Milestone 1: simulator and benchmark scaffold

Build only the simulator + teacher benchmark first.

Deliverables:

- a canonical marketing dataset schema,
- an MMM synthetic generator,
- a lift synthetic generator,
- `pymc-marketing` teacher fitting scripts,
- benchmark reports saved per run.

Success criteria:

- synthetic tasks recover known latent parameters,
- teacher model works end to end on generated tasks,
- metrics are stable across repeated seeds.

### Milestone 2: PFNs-compatible prior layer

Add PFNs-compatible `get_batch()` generators that emit:

- `x`,
- `y`,
- `target_y`,
- optional `style` encodings for hyperparameters such as adstock decay or noise level.

Success criteria:

- batches are shape-consistent,
- priors cover realistic ranges,
- ablations can isolate which prior families help which benchmarks.

### Milestone 3: distillation training

Train a PFNs-style model to imitate teacher outputs.

Success criteria:

- better-than-trivial calibration,
- lower regret than heuristic allocation baselines,
- useful zero-shot or few-shot performance on held-out synthetic tasks.

### Milestone 4: continuous experiment agent

Only after the benchmark harness is reliable should you add the autonomous loop.

Success criteria:

- every run produces a reproducible config and summary,
- proposed changes are small and attributable,
- promotion decisions are deterministic.

## Suggested experiment registry fields

Track each run with structured metadata:

- dataset generator version,
- prior mixture id,
- PFNs architecture hash,
- teacher model version,
- benchmark suite version,
- seed,
- wall-clock time,
- GPU type,
- calibration metrics,
- regret metrics,
- pass/fail promotion decision.

## Practical recommendation for this template

The cleanest near-term use of this repo is:

- keep the existing CLI example as-is,
- use this repository as the starting scaffold for experiment orchestration and documentation,
- add a future package for marketing priors and agent loops when you are ready to turn the research plan into code.

See `examples/pfns_marketing_config.py` for a concrete config sketch showing how a PFNs-style training setup could look at a high level.
