"""Framework-independent Harness Realignment statistics and selection."""

from __future__ import annotations

import math
import pickle
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Any, TypeVar

from tqdm import tqdm

from src.hrp.models import (
    BenchmarkCase,
    CandidateResult,
    DecisionStep,
    HarnessAdapter,
    HRPConfig,
    HRPResult,
    StateActionEnergy,
    Trajectory,
)

_T = TypeVar("_T")
_R = TypeVar("_R")


def smoothed_probabilities(
    counts: Mapping[str, int],
    total: int,
    actions: Sequence[str],
    alpha: float,
) -> dict[str, float]:
    """Dirichlet/Laplace smoothed frequencies over the registered action set."""
    denominator = total + len(actions) * alpha
    return {action: (counts.get(action, 0) + alpha) / denominator for action in actions}


def _sample_state_counts(
    adapter: HarnessAdapter,
    model: Any,
    states: Sequence[tuple[int, Any, Any]],
    samples: int,
    max_workers: int,
    request_slots: BoundedSemaphore | None = None,
    on_complete: Callable[[int, Counter[str]], None] | None = None,
    on_sample: Callable[[], None] | None = None,
) -> dict[int, Counter[str]]:
    """Sample states in state-major order without a barrier between states."""
    counts = {index: Counter() for index, _, _ in states}
    remaining = {index: samples for index, _, _ in states}

    def sample(job: tuple[int, Any, Any]) -> tuple[int, str]:
        index, case_input, state = job
        if request_slots is None:
            action = adapter.sample_action(model, case_input, state)
        else:
            with request_slots:
                action = adapter.sample_action(model, case_input, state)
        return index, action

    def collect(result: tuple[int, str]) -> None:
        index, action = result
        counts[index][action] += 1
        remaining[index] -= 1
        if on_sample is not None:
            on_sample()
        if remaining[index] == 0 and on_complete is not None:
            on_complete(index, counts[index])

    jobs = (
        (index, case_input, state)
        for index, case_input, state in states
        for _ in range(samples)
    )
    if max_workers == 1:
        for job in jobs:
            collect(sample(job))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            pending = set()
            for _ in range(max_workers):
                try:
                    pending.add(pool.submit(sample, next(jobs)))
                except StopIteration:
                    break
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    try:
                        pending.add(pool.submit(sample, next(jobs)))
                    except StopIteration:
                        pass
                    collect(result)
    return counts


@dataclass(frozen=True)
class _TeacherStateEstimate:
    """Strong-teacher empirical policy estimate."""

    counts: Mapping[str, int]
    policy: Mapping[str, float]
    entropy: float


class StrongTeacherEstimator:
    """Fixed stronger-teacher proxy for action geometry and teacher performance."""

    def __init__(
        self,
        models: Mapping[str, Any],
        reference_models: Mapping[str, Any] | None = None,
        max_workers: int = 1,
        reference_request_slots: BoundedSemaphore | None = None,
    ) -> None:
        self.models = models
        self.reference_models = reference_models or {}
        self.max_workers = max_workers
        self._request_slots = BoundedSemaphore(max_workers)
        self._reference_request_slots = reference_request_slots

    def _sample_counts(
        self,
        adapter: HarnessAdapter,
        model: Any,
        case_input: Any,
        state: Any,
        samples: int,
        max_workers: int | None = None,
        limit_remote: bool = True,
    ) -> Counter[str]:
        workers = self.max_workers if max_workers is None else max_workers
        return _sample_state_counts(
            adapter,
            model,
            [(0, case_input, state)],
            samples,
            workers,
            self._request_slots if limit_remote else None,
        )[0]

    def rollout_trajectory(
        self,
        adapter: HarnessAdapter,
        case_input: Any,
    ) -> Trajectory:
        """Run one teacher trajectory within the shared remote-call limit."""
        with self._request_slots:
            return adapter.rollout(self.models[adapter.id], case_input)

    def estimate_state(
        self,
        adapter: HarnessAdapter,
        case_input: Any,
        step: DecisionStep,
        config: HRPConfig,
    ) -> _TeacherStateEstimate:
        actions = step.available_actions
        counts = self._sample_counts(
            adapter,
            self.models[adapter.id],
            case_input,
            step.state,
            config.n_teacher,
        )
        probabilities = smoothed_probabilities(
            counts, config.n_teacher, actions, config.alpha
        )
        return _TeacherStateEstimate(
            counts={action: counts.get(action, 0) for action in actions},
            policy=probabilities,
            entropy=-sum(
                probability * math.log(probability)
                for probability in probabilities.values()
            ),
        )

    def teacher_return(
        self,
        adapter: HarnessAdapter,
        case_input: Any,
        config: HRPConfig,
    ) -> float:
        trajectory = self.rollout_trajectory(adapter, case_input)
        return self.regularized_trajectory_return(
            adapter, case_input, trajectory, config
        )

    def regularized_trajectory_return(
        self,
        adapter: HarnessAdapter,
        case_input: Any,
        trajectory: Trajectory,
        config: HRPConfig,
        on_sample: Callable[[], None] | None = None,
    ) -> float:
        """Estimate one fixed teacher trajectory's regularized return."""

        model = self.models[adapter.id]
        reference_model = self.reference_models.get(adapter.id)
        states = [
            (timestep, case_input, step.state)
            for timestep, step in enumerate(trajectory.steps)
        ]

        if reference_model is not None:
            # Each endpoint has one globally bounded, state-major stream. The
            # next state fills slots released by the previous state's tail,
            # while teacher and reference streams overlap with one another.
            reference_workers = min(
                config.reference_sample_workers or config.max_workers,
                config.max_workers,
            )
            with ThreadPoolExecutor(max_workers=2) as endpoint_streams:
                teacher_future = endpoint_streams.submit(
                    _sample_state_counts,
                    adapter,
                    model,
                    states,
                    config.n_teacher,
                    config.teacher_return_sample_workers,
                    self._request_slots,
                    None,
                    on_sample,
                )
                reference_future = endpoint_streams.submit(
                    _sample_state_counts,
                    adapter,
                    reference_model,
                    states,
                    config.n_teacher,
                    reference_workers,
                    self._reference_request_slots,
                    None,
                    on_sample,
                )
                teacher_counts = teacher_future.result()
                reference_counts = reference_future.result()
            sampled_counts = [
                (teacher_counts[timestep], reference_counts[timestep])
                for timestep in range(len(trajectory.steps))
            ]
        else:
            teacher_counts = _sample_state_counts(
                adapter,
                model,
                states,
                config.n_teacher,
                config.teacher_return_sample_workers,
                self._request_slots,
                None,
                on_sample,
            )
            sampled_counts = [
                (teacher_counts[timestep], None)
                for timestep in range(len(trajectory.steps))
            ]

        value = 0.0
        for timestep, (step, counts) in enumerate(
            zip(trajectory.steps, sampled_counts)
        ):
            actions = step.available_actions
            teacher_probs = smoothed_probabilities(
                counts[0], config.n_teacher, actions, config.alpha
            )
            if counts[1] is not None:
                reference_probs = smoothed_probabilities(
                    counts[1], config.n_teacher, actions, config.alpha
                )
            else:
                reference_probs = {
                    action: 1.0 / len(actions) for action in actions
                }
            kl = sum(
                probability * math.log(probability / reference_probs[action])
                for action, probability in teacher_probs.items()
            )
            value += config.gamma**timestep * (
                step.reward - step.cost - config.tau * kl
            )
        return value


def quotient_energy(
    probabilities: Mapping[str, float],
    tau: float,
) -> dict[str, float]:
    """Gauge-fixed representative ``tau * P_perp * log(policy)``."""
    raw = {
        action: tau * math.log(probability)
        for action, probability in probabilities.items()
    }
    center = sum(raw.values()) / len(raw)
    return {action: value - center for action, value in raw.items()}


def centered_star_residual(
    target_probs: Mapping[str, float],
    teacher_probs: Mapping[str, float],
    reference_action: str,
    tau: float,
    clip: float | None,
) -> dict[str, float]:
    """Star-probe residual coordinates projected onto the quotient space.

    For each action ``a`` other than the reference ``a0`` the raw coordinate
    is the clipped target-teacher pairwise log-odds residual
    ``tau*log(p_theta(a)/p_theta(a0)) - tau*log(p_T(a)/p_T(a0))``; the
    reference coordinate is zero.  Applying the centering projection
    ``P_perp = I - 11^T/|A_s|`` yields the gauge-invariant vector ``x_hat``.
    """
    if clip is None:
        target_energy = quotient_energy(target_probs, tau)
        teacher_energy = quotient_energy(teacher_probs, tau)
        return {
            action: target_energy[action] - teacher_energy[action]
            for action in target_probs
        }

    raw: dict[str, float] = {}
    for action in target_probs:
        if action == reference_action:
            raw[action] = 0.0
            continue
        delta_target = tau * math.log(
            target_probs[action] / target_probs[reference_action]
        )
        delta_teacher = tau * math.log(
            teacher_probs[action] / teacher_probs[reference_action]
        )
        residual = delta_target - delta_teacher
        raw[action] = max(-clip, min(clip, residual))
    center = sum(raw.values()) / len(raw)
    return {action: value - center for action, value in raw.items()}


@dataclass
class _StateProbe:
    """Complete empirical action-energy readout for one saved state."""

    actions: tuple[str, ...]
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
    squared_error: float
    reverse_kl: float
    teacher_entropy: float


@dataclass(frozen=True)
class _TargetStateEstimate:
    """Target empirical policy estimate for one saved state."""

    counts: Mapping[str, int]
    policy: Mapping[str, float]


class HarnessRealignment:
    """Strong-teacher harness selection over a fixed candidate set."""

    def __init__(
        self,
        config: HRPConfig,
        target_request_slots: BoundedSemaphore | None = None,
    ) -> None:
        self.config = config
        self._rng = random.Random(config.seed)
        self._target_request_slots = (
            target_request_slots or BoundedSemaphore(config.max_workers)
        )

    def _map(self, fn: Callable[[_T], _R], items: Sequence[_T]) -> list[_R]:
        """Order-preserving map, parallel across adapter calls when configured."""
        if self.config.max_workers <= 1 or len(items) <= 1:
            return [fn(item) for item in items]
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            return list(pool.map(fn, items))

    def _checkpointed_map(
        self,
        fn: Callable[[_T], _R],
        items: Sequence[_T],
        checkpoint: Path | None,
        max_workers: int | None = None,
        description: str = "rollouts",
    ) -> list[_R]:
        workers = self.config.max_workers if max_workers is None else max_workers
        if checkpoint is None:
            with tqdm(
                total=len(items),
                desc=description,
                unit="trajectory",
                dynamic_ncols=True,
                mininterval=1.0,
            ) as progress:
                if workers <= 1 or len(items) <= 1:
                    results = []
                    for item in items:
                        results.append(fn(item))
                        progress.update()
                    return results
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(fn, item): index
                        for index, item in enumerate(items)
                    }
                    completed_results = {}
                    for future in as_completed(futures):
                        completed_results[futures[future]] = future.result()
                        progress.update()
                    return [
                        completed_results[index] for index in range(len(items))
                    ]
        completed = (
            pickle.loads(checkpoint.read_bytes()) if checkpoint.exists() else {}
        )
        missing = [
            (index, item)
            for index, item in enumerate(items)
            if index not in completed
        ]

        progress = tqdm(
            total=len(items),
            initial=len(completed),
            desc=description,
            unit="trajectory",
            dynamic_ncols=True,
            mininterval=1.0,
        )

        def save(index: int, result: _R) -> None:
            completed[index] = result
            temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
            temporary.write_bytes(pickle.dumps(completed))
            temporary.replace(checkpoint)
            progress.update()
            tqdm.write(
                f"[checkpoint] {checkpoint.name}: {len(completed)}/{len(items)}",
            )

        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        try:
            if workers <= 1 or len(missing) <= 1:
                failures: list[tuple[int, Exception]] = []
                for index, item in missing:
                    try:
                        result = fn(item)
                    except Exception as error:
                        failures.append((index, error))
                    else:
                        save(index, result)
            else:
                failures = []
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(fn, item): index for index, item in missing}
                    for future in as_completed(futures):
                        index = futures[future]
                        try:
                            result = future.result()
                        except Exception as error:
                            # Do not let one failed future discard every other
                            # completed result in the same expensive batch.  Save
                            # all successes first; a resumed run then recomputes
                            # only the failed item(s).
                            failures.append((index, error))
                        else:
                            save(index, result)
        finally:
            progress.close()
        if failures:
            _, error = min(failures, key=lambda item: item[0])
            raise error
        return [completed[index] for index in range(len(items))]

    def run(
        self,
        adapters: Sequence[HarnessAdapter],
        target_models: Mapping[str, Any],
        teacher: StrongTeacherEstimator,
        cases: Sequence[BenchmarkCase],
    ) -> HRPResult:
        candidates: dict[str, CandidateResult] = {}
        for adapter in adapters:
            candidates[adapter.id] = self.evaluate_candidate(
                adapter,
                target_models[adapter.id],
                teacher,
                cases,
                num_candidates=len(adapters),
            )
        return HRPResult(selected=self._select(candidates), candidates=candidates)

    def evaluate_candidate(
        self,
        adapter: HarnessAdapter,
        target_model: Any,
        teacher: StrongTeacherEstimator,
        cases: Sequence[BenchmarkCase],
        num_candidates: int = 1,
        checkpoint_dir: Path | None = None,
        teacher_return_checkpoint: Path | None = None,
    ) -> CandidateResult:
        config = self.config
        # Step B: occupancy rollouts of the frozen target model, keeping the
        # discounted weight of every probed decision state.  Rollouts may run
        # in parallel; probe subsampling stays on this thread so the seeded
        # RNG keeps its deterministic order.
        rollout_specs = [
            (case.id, rollout_index, case.inputs[adapter.id])
            for case in cases
            for rollout_index in range(config.occupancy_rollouts)
        ]
        rollout_inputs = [case_input for _, _, case_input in rollout_specs]
        occupancy_checkpoint = (
            checkpoint_dir / "occupancy.pkl" if checkpoint_dir is not None else None
        )
        if occupancy_checkpoint is not None and occupancy_checkpoint.exists():
            occupancy = pickle.loads(occupancy_checkpoint.read_bytes())
            with tqdm(
                total=len(rollout_inputs),
                initial=len(occupancy),
                desc=f"{adapter.id} occupancy",
                unit="trajectory",
                dynamic_ncols=True,
                mininterval=1.0,
            ):
                pass
            tqdm.write(
                f"[checkpoint] resumed {len(occupancy)} occupancy trajectories",
            )
        else:
            def target_rollout(case_input: Any) -> Trajectory:
                with self._target_request_slots:
                    return adapter.rollout(target_model, case_input)

            trajectories = self._checkpointed_map(
                target_rollout,
                rollout_inputs,
                (
                    checkpoint_dir / "occupancy_trajectories.pkl"
                    if checkpoint_dir is not None
                    else None
                ),
                max_workers=min(
                    config.rollout_workers or config.max_workers,
                    config.max_workers,
                ),
                description=f"{adapter.id} occupancy",
            )
            occupancy = [
                (case_input, trajectory, self._probe_weights(trajectory))
                for case_input, trajectory in zip(rollout_inputs, trajectories)
            ]
            if occupancy_checkpoint is not None:
                occupancy_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temporary = occupancy_checkpoint.with_suffix(".pkl.tmp")
                temporary.write_bytes(pickle.dumps(occupancy))
                temporary.replace(occupancy_checkpoint)

        total_rollouts = len(occupancy)
        probe_index: dict[int, tuple[Any, DecisionStep]] = {}
        probe_metadata: dict[int, tuple[str, int, int, int, str]] = {}
        probe_discounted_weights: dict[int, float] = {}
        probe_draws: dict[int, int] = {}
        for trajectory_index, (spec, occupancy_entry) in enumerate(
            zip(rollout_specs, occupancy)
        ):
            case_id, rollout_index, _ = spec
            case_input, trajectory, weighted_steps = occupancy_entry
            timesteps = {
                id(step): timestep
                for timestep, step in enumerate(trajectory.steps)
            }
            for step, weight in weighted_steps:
                key = id(step)
                probe_index.setdefault(key, (case_input, step))
                probe_metadata.setdefault(
                    key,
                    (
                        case_id,
                        trajectory_index,
                        rollout_index,
                        timesteps[key],
                        step.action,
                    ),
                )
                probe_discounted_weights[key] = (
                    probe_discounted_weights.get(key, 0.0) + weight
                )
                probe_draws[key] = probe_draws.get(key, 0) + 1

        # Steps C-F: fork the target and teacher on every unique probe state
        # and compute empirical, smoothed action-energy diagnostics. This path
        # intentionally does not claim a finite-sample support certificate.
        probe_entries = list(probe_index.items())
        probe_checkpoint = (
            checkpoint_dir / "target_teacher_probes.pkl"
            if checkpoint_dir is not None
            else None
        )
        probe_results = self._probe_states(
            adapter,
            target_model,
            teacher,
            probe_entries,
            probe_checkpoint,
        )
        probes: dict[int, _StateProbe] = {
            key: result for (key, _), result in zip(probe_entries, probe_results)
        }

        # Step G: retain the action-level quotient object and derive every
        # scalar mismatch diagnostic from its occupancy-weighted entries.
        action_energy_map: list[StateActionEnergy] = []
        for key, probe in probes.items():
            (
                case_id,
                trajectory_index,
                rollout_index,
                timestep,
                observed_action,
            ) = probe_metadata[key]
            discounted_weight = probe_discounted_weights[key]
            occupancy_mass = discounted_weight / total_rollouts
            compat_err_contribution = occupancy_mass * probe.squared_error
            reverse_kl_gap_contribution = (
                config.tau
                / (1.0 - config.gamma)
                * occupancy_mass
                * probe.reverse_kl
            )
            energy_proxy_contribution = compat_err_contribution / (
                4.0 * config.tau * (1.0 - config.gamma)
            )
            action_energy_map.append(
                StateActionEnergy(
                    state_id=(
                        f"{adapter.id}:{case_id}:trajectory-{trajectory_index}:"
                        f"t{timestep}"
                    ),
                    case_id=case_id,
                    trajectory_index=trajectory_index,
                    rollout_index=rollout_index,
                    timestep=timestep,
                    observed_action=observed_action,
                    actions=probe.actions,
                    probe_draws=probe_draws[key],
                    discounted_weight=discounted_weight,
                    occupancy_mass=occupancy_mass,
                    reference_action=probe.reference_action,
                    target_counts=dict(probe.target_counts),
                    teacher_counts=dict(probe.teacher_counts),
                    target_policy=dict(probe.target_policy),
                    teacher_policy=dict(probe.teacher_policy),
                    target_energy=dict(probe.target_energy),
                    teacher_energy=dict(probe.teacher_energy),
                    residual=dict(probe.residual),
                    scoring_residual=dict(probe.scoring_residual),
                    residual_squared_norm=probe.residual_squared_norm,
                    scoring_squared_norm=probe.squared_error,
                    reverse_kl=probe.reverse_kl,
                    teacher_entropy=probe.teacher_entropy,
                    compat_err_contribution=compat_err_contribution,
                    reverse_kl_gap_contribution=reverse_kl_gap_contribution,
                    energy_proxy_contribution=energy_proxy_contribution,
                )
            )
        action_energy_map.sort(
            key=lambda entry: (entry.trajectory_index, entry.timestep)
        )
        compat_err = sum(
            entry.compat_err_contribution for entry in action_energy_map
        )
        target_teacher_reverse_kl = sum(
            entry.reverse_kl_gap_contribution for entry in action_energy_map
        )
        target_teacher_energy_proxy = sum(
            entry.energy_proxy_contribution for entry in action_energy_map
        )
        observed_occupancy_mass = sum(
            entry.occupancy_mass for entry in action_energy_map
        )
        truncated_tail_mass = sum(
            config.gamma ** len(trajectory.steps)
            for _, trajectory, _ in occupancy
            if trajectory.truncated
        ) / total_rollouts

        # Step J: regularized teacher-performance proxy and its LCB.
        (
            teacher_return_proxy,
            teacher_return_lcb,
            teacher_return_count,
        ) = self._teacher_performance(
            adapter,
            teacher,
            cases,
            num_candidates,
            config.delta,
            teacher_return_checkpoint
            if teacher_return_checkpoint is not None
            else checkpoint_dir / "teacher_return.pkl"
            if checkpoint_dir is not None
            else None,
        )

        probe_count = len(probes)
        teacher_entropy = (
            sum(probe.teacher_entropy for probe in probes.values()) / probe_count
        )

        return CandidateResult(
            harness=adapter.id,
            score=teacher_return_proxy - target_teacher_energy_proxy,
            teacher_return_proxy=teacher_return_proxy,
            target_teacher_reverse_kl=target_teacher_reverse_kl,
            target_teacher_energy_proxy=target_teacher_energy_proxy,
            teacher_return_lcb=teacher_return_lcb,
            compat_err=compat_err,
            probe_states=probe_count,
            teacher_entropy=teacher_entropy,
            occupancy_trajectories=total_rollouts,
            teacher_return_trajectories=teacher_return_count,
            observed_occupancy_mass=observed_occupancy_mass,
            truncated_tail_mass=truncated_tail_mass,
            protocol_config=asdict(config),
            action_energy_map=action_energy_map,
        )

    def _probe_weights(
        self, trajectory: Trajectory
    ) -> list[tuple[DecisionStep, float]]:
        """Decision states of one rollout with their discounted MC weights.

        With no subsampling every state ``s_t`` carries ``(1-gamma)*gamma^t``.
        With ``probes_per_trajectory = m`` set, timesteps are drawn with
        probability proportional to ``gamma^t`` and importance-corrected so
        the discounted sum stays unbiased.
        """
        config = self.config
        steps = trajectory.steps
        discounts = [config.gamma**t for t in range(len(steps))]
        budget = config.probes_per_trajectory
        if budget is None or budget >= len(steps):
            return [
                (step, (1.0 - config.gamma) * discount)
                for step, discount in zip(steps, discounts)
            ]
        total_discount = sum(discounts)
        chosen = self._rng.choices(range(len(steps)), weights=discounts, k=budget)
        weight = (1.0 - config.gamma) * total_discount / budget
        return [(steps[index], weight) for index in chosen]

    def _estimate_target_state(
        self,
        adapter: HarnessAdapter,
        target_model: Any,
        case_input: Any,
        step: DecisionStep,
        max_workers: int,
    ) -> _TargetStateEstimate:
        config = self.config
        actions = step.available_actions
        target_counts = Counter(
            self._map(
                lambda _: adapter.sample_action(target_model, case_input, step.state),
                list(range(config.n_target)),
            )
            if max_workers > 1
            else (
                adapter.sample_action(target_model, case_input, step.state)
                for _ in range(config.n_target)
            )
        )
        target_probs = smoothed_probabilities(
            target_counts, config.n_target, actions, config.alpha
        )
        return _TargetStateEstimate(
            counts={action: target_counts.get(action, 0) for action in actions},
            policy=target_probs,
        )

    def _combine_state_probe(
        self,
        step: DecisionStep,
        target: _TargetStateEstimate,
        teacher_estimate: _TeacherStateEstimate,
    ) -> _StateProbe:
        config = self.config
        actions = step.available_actions

        # Step E reference action: highest joint support under both models.
        reference_action = max(
            sorted(actions),
            key=lambda action: min(
                target.policy[action], teacher_estimate.policy[action]
            ),
        )
        target_energy = quotient_energy(target.policy, config.tau)
        teacher_energy = quotient_energy(teacher_estimate.policy, config.tau)
        residual = {
            action: target_energy[action] - teacher_energy[action]
            for action in actions
        }
        scoring_residual = centered_star_residual(
            target.policy,
            teacher_estimate.policy,
            reference_action,
            config.tau,
            config.energy_clip,
        )
        residual_squared_norm = sum(value * value for value in residual.values())
        squared_error = sum(value * value for value in scoring_residual.values())
        # Direct point estimate on the same Laplace-smoothed action policies.
        reverse_kl = sum(
            probability
            * math.log(probability / teacher_estimate.policy[action])
            for action, probability in target.policy.items()
        )

        return _StateProbe(
            actions=tuple(actions),
            reference_action=reference_action,
            target_counts=target.counts,
            teacher_counts=teacher_estimate.counts,
            target_policy=target.policy,
            teacher_policy=teacher_estimate.policy,
            target_energy=target_energy,
            teacher_energy=teacher_energy,
            residual=residual,
            scoring_residual=scoring_residual,
            residual_squared_norm=residual_squared_norm,
            squared_error=squared_error,
            reverse_kl=reverse_kl,
            teacher_entropy=teacher_estimate.entropy,
        )

    def _probe_states(
        self,
        adapter: HarnessAdapter,
        target_model: Any,
        teacher: StrongTeacherEstimator,
        entries: Sequence[tuple[int, tuple[Any, DecisionStep]]],
        checkpoint: Path | None,
    ) -> list[_StateProbe]:
        """Overlap ordered target and teacher streams across saved states."""
        completed = (
            pickle.loads(checkpoint.read_bytes())
            if checkpoint is not None and checkpoint.exists()
            else {}
        )
        missing = [
            (index, entry)
            for index, entry in enumerate(entries)
            if index not in completed
        ]
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)

        def save(index: int, result: _StateProbe) -> None:
            completed[index] = result
            if checkpoint is not None:
                temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
                temporary.write_bytes(pickle.dumps(completed))
                temporary.replace(checkpoint)
                tqdm.write(
                    f"[checkpoint] {checkpoint.name}: "
                    f"{len(completed)}/{len(entries)}",
                )

        sample_states = [
            (index, entry[1][0], entry[1][1].state)
            for index, entry in missing
        ]
        target_estimates: dict[int, _TargetStateEstimate] = {}
        teacher_estimates: dict[int, _TeacherStateEstimate] = {}
        result_lock = Lock()

        def finalize(index: int) -> None:
            target = target_estimates.get(index)
            teacher_estimate = teacher_estimates.get(index)
            if target is None or teacher_estimate is None:
                return
            step = entries[index][1][1]
            save(
                index,
                self._combine_state_probe(step, target, teacher_estimate),
            )
            del target_estimates[index]
            del teacher_estimates[index]

        def accept_target(index: int, counts: Counter[str]) -> None:
            step = entries[index][1][1]
            probabilities = smoothed_probabilities(
                counts,
                self.config.n_target,
                step.available_actions,
                self.config.alpha,
            )
            estimate = _TargetStateEstimate(
                counts={
                    action: counts.get(action, 0)
                    for action in step.available_actions
                },
                policy=probabilities,
            )
            with result_lock:
                target_estimates[index] = estimate
                finalize(index)

        def accept_teacher(index: int, counts: Counter[str]) -> None:
            step = entries[index][1][1]
            probabilities = smoothed_probabilities(
                counts,
                self.config.n_teacher,
                step.available_actions,
                self.config.alpha,
            )
            estimate = _TeacherStateEstimate(
                counts={
                    action: counts.get(action, 0)
                    for action in step.available_actions
                },
                policy=probabilities,
                entropy=-sum(
                    probability * math.log(probability)
                    for probability in probabilities.values()
                ),
            )
            with result_lock:
                teacher_estimates[index] = estimate
                finalize(index)

        # Keep one bounded stream per endpoint, but remove the per-state
        # barrier: tail slots from state i immediately start state i+1 work.
        completed_samples = len(completed) * (
            self.config.n_target + self.config.n_teacher
        )
        with tqdm(
            total=len(entries) * (self.config.n_target + self.config.n_teacher),
            initial=completed_samples,
            desc=f"{adapter.id} target/teacher probes",
            unit="sample",
            dynamic_ncols=True,
            mininterval=1.0,
        ) as progress:
            progress_lock = Lock()

            def update_progress() -> None:
                with progress_lock:
                    progress.update()

            with ThreadPoolExecutor(max_workers=2) as endpoint_streams:
                target_future = endpoint_streams.submit(
                    _sample_state_counts,
                    adapter,
                    target_model,
                    sample_states,
                    self.config.n_target,
                    self.config.max_workers,
                    self._target_request_slots,
                    accept_target,
                    update_progress,
                )
                teacher_future = endpoint_streams.submit(
                    _sample_state_counts,
                    adapter,
                    teacher.models[adapter.id],
                    sample_states,
                    self.config.n_teacher,
                    teacher.max_workers,
                    teacher._request_slots,
                    accept_teacher,
                    update_progress,
                )
                target_future.result()
                teacher_future.result()

        return [completed[index] for index in range(len(entries))]

    def _teacher_performance(
        self,
        adapter: HarnessAdapter,
        teacher: StrongTeacherEstimator,
        cases: Sequence[BenchmarkCase],
        num_candidates: int,
        delta_teacher_return: float,
        checkpoint: Path | None = None,
    ) -> tuple[float, float, int]:
        """Step J: mean clipped regularized teacher return and its LCB."""
        config = self.config
        lower_bound, upper_bound = config.return_bounds

        def one_trajectory(item: tuple[int, Any]) -> Trajectory:
            _, case_input = item
            return teacher.rollout_trajectory(adapter, case_input)

        return_progress_lock = Lock()

        def update_return_progress() -> None:
            with return_progress_lock:
                return_sample_progress.update()

        def one_return(item: tuple[tuple[int, Any], Trajectory]) -> float:
            (_, case_input), trajectory = item
            value = teacher.regularized_trajectory_return(
                adapter,
                case_input,
                trajectory,
                config,
                update_return_progress,
            )
            return min(max(value, lower_bound), upper_bound)

        rollout_inputs = [
            (rollout_index, case.inputs[adapter.id])
            for case in cases
            for rollout_index in range(config.teacher_return_rollouts)
        ]
        trajectory_checkpoint = (
            checkpoint.with_name("teacher_return_trajectories.pkl")
            if checkpoint is not None
            else None
        )
        trajectories = (
            pickle.loads(trajectory_checkpoint.read_bytes())
            if trajectory_checkpoint is not None and trajectory_checkpoint.exists()
            else {}
        )
        returns = (
            pickle.loads(checkpoint.read_bytes())
            if checkpoint is not None and checkpoint.exists()
            else {}
        )

        def save_result(
            path: Path | None,
            completed: dict[int, Any],
            index: int,
            value: Any,
        ) -> None:
            completed[index] = value
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(pickle.dumps(completed))
            temporary.replace(path)
            tqdm.write(
                f"[checkpoint] {path.name}: "
                f"{len(completed)}/{len(rollout_inputs)}",
            )

        rollout_workers = min(
            config.rollout_workers or config.max_workers,
            teacher.max_workers,
        )
        return_workers = 10
        samples_per_state = config.n_teacher * (
            2 if adapter.id in teacher.reference_models else 1
        )
        known_return_samples = {
            index: len(trajectory.steps) * samples_per_state
            for index, trajectory in trajectories.items()
        }
        completed_return_samples = sum(
            known_return_samples[index] for index in returns
        )
        with (
            tqdm(
                total=len(rollout_inputs),
                initial=len(trajectories),
                desc=f"{adapter.id} teacher rollouts",
                unit="trajectory",
                dynamic_ncols=True,
                mininterval=1.0,
            ) as rollout_progress,
            tqdm(
                total=sum(known_return_samples.values()),
                initial=completed_return_samples,
                desc=f"{adapter.id} teacher/reference return",
                unit="sample",
                dynamic_ncols=True,
                mininterval=1.0,
            ) as return_sample_progress,
            ThreadPoolExecutor(max_workers=rollout_workers) as rollout_pool,
            ThreadPoolExecutor(max_workers=return_workers) as return_pool,
        ):
            rollout_futures = {
                rollout_pool.submit(one_trajectory, item): index
                for index, item in enumerate(rollout_inputs)
                if index not in trajectories
            }
            return_futures = {
                return_pool.submit(
                    one_return, (rollout_inputs[index], trajectory)
                ): index
                for index, trajectory in trajectories.items()
                if index not in returns
            }
            while rollout_futures or return_futures:
                done, _ = wait(
                    set(rollout_futures) | set(return_futures),
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    if future in rollout_futures:
                        index = rollout_futures.pop(future)
                        trajectory = future.result()
                        save_result(
                            trajectory_checkpoint,
                            trajectories,
                            index,
                            trajectory,
                        )
                        rollout_progress.update()
                        sample_count = len(trajectory.steps) * samples_per_state
                        known_return_samples[index] = sample_count
                        with return_progress_lock:
                            return_sample_progress.total += sample_count
                            return_sample_progress.refresh()
                        if index not in returns:
                            return_futures[
                                return_pool.submit(
                                    one_return,
                                    (rollout_inputs[index], trajectory),
                                )
                            ] = index
                    else:
                        index = return_futures.pop(future)
                        save_result(
                            checkpoint,
                            returns,
                            index,
                            future.result(),
                        )

        returns = [returns[index] for index in range(len(rollout_inputs))]
        count = len(returns)
        proxy = sum(returns) / count
        margin = (upper_bound - lower_bound) * math.sqrt(
            math.log(2.0 * num_candidates / delta_teacher_return) / (2.0 * count)
        )
        return proxy, proxy - margin, count

    def _select(self, candidates: dict[str, CandidateResult]) -> str:
        """Select the highest empirical teacher-proxy energy score."""
        ordered = sorted(candidates.values(), key=lambda result: result.harness)
        best = max(
            ordered,
            key=lambda result: (result.score, -result.target_teacher_energy_proxy),
        )
        return best.harness


def run_experiment(
    adapters: Sequence[HarnessAdapter],
    target_models: Mapping[str, Any],
    teacher: StrongTeacherEstimator,
    cases: Sequence[BenchmarkCase],
    config: HRPConfig,
) -> HRPResult:
    """Run the full protocol and return the selection with diagnostics."""
    return HarnessRealignment(config).run(
        adapters,
        target_models,
        teacher,
        cases,
    )
