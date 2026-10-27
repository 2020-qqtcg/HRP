"""Data contracts for the Harness Realignment Protocol."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.hrp.core import StrongTeacherEstimator


@dataclass
class DecisionStep:
    """One executed semantic decision inside a rollout.

    ``state`` is an adapter-defined snapshot of the full semantic state taken
    *before* the decision.  It must contain everything the adapter needs to
    fork the same decision again through ``HarnessAdapter.sample_action``.
    ``action`` is the finite semantic action consumed by HRP.  When the native
    action contains materially more information (for example
    ``bash(command=...)``), ``concrete_action`` keeps that lossless record; the
    semantic action is then an explicit projection rather than a replacement
    for the environment-facing action.
    ``reward`` and ``cost`` follow the shared evaluation specification
    ``r = reward - cost``; the cost is already weighted by the experiment.
    """

    state: Any
    action: str
    reward: float
    cost: float
    available_actions: tuple[str, ...]
    concrete_action: Any | None = None


@dataclass
class Trajectory:
    """An ordered record of the executed semantic decisions of one rollout.

    ``truncated`` marks rollouts stopped by a step budget instead of a natural
    ``submit_answer``/``stop`` termination. HRP reports the corresponding
    unobserved discounted occupancy mass separately from the empirical map.
    """

    steps: list[DecisionStep]
    truncated: bool = False


@dataclass(frozen=True)
class BenchmarkCase:
    """One task, with per-harness native inputs keyed by adapter id."""

    id: str
    inputs: Mapping[str, Any]


class HarnessAdapter(ABC):
    """Uniform semantic interface over one candidate harness framework.

    Adapters expose the shared finite macro-action vocabulary through
    ``semantic_actions`` and translate every framework-native decision into
    one of those actions.  Frameworks whose behavior is not semantically
    equivalent are documented in ``src/adapter/action-diff.md``.
    """

    id: str
    semantic_actions: tuple[str, ...]

    @abstractmethod
    def rollout(self, model: Any, case_input: Any) -> Trajectory:
        """Run one full model+harness trajectory on a task."""

    @abstractmethod
    def sample_action(self, model: Any, case_input: Any, state: Any) -> str:
        """Fork a saved decision state once and return the executed action."""


@dataclass(frozen=True)
class HRPConfig:
    """Pre-registered constants of the protocol.

    ``return_bounds`` must bracket every possible clipped single-trajectory
    regularized return. ``energy_clip`` optionally clips the residual used by
    the empirical score; the action-energy map always retains the complete
    smoothed-frequency readout.
    """

    gamma: float
    tau: float
    return_bounds: tuple[float, float]
    alpha: float = 0.5
    delta: float = 0.05
    n_target: int = 256
    n_teacher: int = 512
    occupancy_rollouts: int = 16
    teacher_return_rollouts: int = 16
    probes_per_trajectory: int | None = None
    energy_clip: float | None = None
    seed: int | None = None
    max_workers: int = 1
    rollout_workers: int | None = None
    teacher_return_sample_workers: int = 1
    reference_sample_workers: int | None = None


@dataclass(frozen=True)
class StateActionEnergy:
    """Action-level energy readout and sampling evidence for one target state.

    ``target_energy`` and ``teacher_energy`` are the gauge-fixed vectors
    ``tau * P_perp * log(policy)``. ``residual`` is their signed difference and
    therefore preserves the identifiable energy-quotient object. The optional
    configured clip is visible only in ``scoring_residual``.

    ``discounted_weight`` is the total Monte Carlo weight assigned to this
    state within its trajectory, including repeated importance-sampling draws.
    ``occupancy_mass`` additionally divides by the number of occupancy
    trajectories. The three contribution fields sum to their corresponding
    candidate-level diagnostics.
    """

    state_id: str
    case_id: str
    trajectory_index: int
    rollout_index: int
    timestep: int
    observed_action: str
    actions: tuple[str, ...]
    probe_draws: int
    discounted_weight: float
    occupancy_mass: float
    reference_action: str
    target_counts: Mapping[str, int]
    teacher_counts: Mapping[str, int]
    target_policy: Mapping[str, float]
    teacher_policy: Mapping[str, float]
    target_energy: Mapping[str, float]
    teacher_energy: Mapping[str, float]
    residual: Mapping[str, float]
    scoring_residual: Mapping[str, float]
    residual_squared_norm: float
    scoring_squared_norm: float
    reverse_kl: float
    teacher_entropy: float
    compat_err_contribution: float
    reverse_kl_gap_contribution: float
    energy_proxy_contribution: float


@dataclass
class CandidateResult:
    """Empirical ranking, scalar diagnostics, and full action-energy map."""

    harness: str
    score: float
    teacher_return_proxy: float
    target_teacher_reverse_kl: float
    target_teacher_energy_proxy: float
    teacher_return_lcb: float
    compat_err: float
    probe_states: int
    teacher_entropy: float
    occupancy_trajectories: int
    teacher_return_trajectories: int
    observed_occupancy_mass: float
    truncated_tail_mass: float
    protocol_config: Mapping[str, Any]
    action_energy_map: list[StateActionEnergy]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidateResult:
        """Reconstruct a typed result from its JSON-compatible dictionary."""
        data = dict(value)
        data["action_energy_map"] = [
            entry
            if isinstance(entry, StateActionEnergy)
            else StateActionEnergy(
                **{
                    **entry,
                    "actions": tuple(entry["actions"]),
                }
            )
            for entry in data["action_energy_map"]
        ]
        protocol_config = dict(data["protocol_config"])
        protocol_config["return_bounds"] = tuple(
            protocol_config["return_bounds"]
        )
        data["protocol_config"] = protocol_config
        return cls(**data)


@dataclass
class HRPResult:
    """Selected harness plus the full per-candidate diagnostic table."""

    selected: str
    candidates: dict[str, CandidateResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "candidates": {
                harness: result.to_dict() for harness, result in self.candidates.items()
            },
        }


@dataclass
class Experiment:
    """Complete experiment specification consumed by the CLI entry point."""

    adapters: Sequence[HarnessAdapter]
    target_models: Mapping[str, Any]
    teacher: StrongTeacherEstimator
    cases: Sequence[BenchmarkCase]
    config: HRPConfig
