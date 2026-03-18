"""Example PFNs-style configuration for Bayesian marketing research.

This file is intentionally a sketch rather than an executable training script.
It demonstrates how a project bootstrapped from this template could organize
its config objects for PFNs-like training, teacher distillation, and
continuous experimentation.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MarketingPriorConfig:
    """Synthetic marketing prior definition."""

    name: str
    channels: int
    geos: int
    include_adstock: bool = True
    include_saturation: bool = True
    include_lift_effects: bool = False
    parameter_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class DistillationTargetConfig:
    """Teacher outputs used for PFNs-style supervision."""

    posterior_mean: bool = True
    predictive_quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    channel_contribution: bool = True
    budget_regret_target: bool = False


@dataclass(frozen=True)
class ExperimentLoopConfig:
    """Continuous experiment loop settings."""

    max_daily_runs: int = 6
    promote_minimum_calibration_gain: float = 0.01
    promote_maximum_regret: float = 0.05
    mutation_scope: tuple[str, ...] = (
        "prior_mixture",
        "prior_ranges",
        "loss_weights",
        "sequence_length",
    )


@dataclass(frozen=True)
class PFNsMarketingConfig:
    """High-level project configuration sketch."""

    priors: tuple[MarketingPriorConfig, ...]
    teacher_backend: str = "pymc-marketing"
    batch_size: int = 32
    seq_len: int = 128
    num_features: int = 24
    embedding_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    steps_per_epoch: int = 500
    epochs: int = 50
    learning_rate: float = 1e-4
    distillation: DistillationTargetConfig = field(
        default_factory=DistillationTargetConfig
    )
    experiment_loop: ExperimentLoopConfig = field(
        default_factory=ExperimentLoopConfig
    )


EXAMPLE_CONFIG = PFNsMarketingConfig(
    priors=(
        MarketingPriorConfig(
            name="mmm_response",
            channels=8,
            geos=12,
            include_adstock=True,
            include_saturation=True,
            parameter_ranges={
                "adstock_decay": (0.1, 0.95),
                "saturation_alpha": (0.2, 3.0),
                "observation_noise": (0.01, 0.3),
            },
        ),
        MarketingPriorConfig(
            name="incrementality_lift",
            channels=3,
            geos=40,
            include_lift_effects=True,
            parameter_ranges={
                "treatment_effect": (-0.2, 0.8),
                "spillover": (0.0, 0.4),
                "preperiod_bias": (0.0, 0.3),
            },
        ),
    )
)
