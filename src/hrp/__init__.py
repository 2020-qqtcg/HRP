"""Harness Realignment Protocol: statistics, contracts, and selection."""

from src.hrp.core import (
    HarnessRealignment,
    StrongTeacherEstimator,
    centered_star_residual,
    quotient_energy,
    run_experiment,
    smoothed_probabilities,
)
from src.hrp.models import (
    BenchmarkCase,
    CandidateResult,
    DecisionStep,
    Experiment,
    HarnessAdapter,
    HRPConfig,
    HRPResult,
    StateActionEnergy,
    Trajectory,
)

__all__ = [
    "BenchmarkCase",
    "CandidateResult",
    "DecisionStep",
    "Experiment",
    "HRPConfig",
    "HRPResult",
    "HarnessAdapter",
    "HarnessRealignment",
    "StrongTeacherEstimator",
    "StateActionEnergy",
    "Trajectory",
    "centered_star_residual",
    "quotient_energy",
    "run_experiment",
    "smoothed_probabilities",
]
