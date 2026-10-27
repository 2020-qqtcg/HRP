"""Controlled presentation and stochastic execution kernels for Section 5.1.

The presentation study keeps semantic H0 actions fixed.  The execution study
then changes only click reliability and click utility cost.  Its actor prompt is
identical across execution settings and deliberately does not claim that costs
are zero, while also withholding the numerical reliability and cost values.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hrp_miniwob.adapters.miniwob_readout import ReadoutRenderer, RenderedPrompt

NEUTRAL_OLD = "Only exact native completion earns 1; other rewards and explicit costs are zero."
NEUTRAL_NEW = (
    "Only exact native completion counts as task success. The deployment controls click "
    "execution and utility accounting; their settings are not exposed in this prompt."
)


class NeutralReadoutRenderer(ReadoutRenderer):
    """R0--R5 renderer with one execution-neutral system message.

    The inherited renderer still owns the reversible semantic/code mapping and
    JSON transformations.  Replacing this one sentence is intentional: every
    p/c setting receives the same prompt, and the prompt no longer contradicts
    nonzero execution costs.
    """

    def render(self, state) -> RenderedPrompt:
        rendered = super().render(state)
        messages = json.loads(json.dumps(rendered.messages))
        system = messages[0]["content"]
        if system.count(NEUTRAL_OLD) != 1:
            raise AssertionError("The sealed H0 sentence expected by the neutral renderer changed")
        messages[0]["content"] = system.replace(NEUTRAL_OLD, NEUTRAL_NEW)
        payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        return RenderedPrompt(
            messages=messages,
            semantic_codes=rendered.semantic_codes,
            prompt_sha256=hashlib.sha256(payload.encode()).hexdigest(),
            mapping=rendered.mapping,
        )


@dataclass(frozen=True)
class ControlledKernel:
    """Two-branch finite kernel for successful and failed click execution."""

    next_state: NDArray[np.int64]       # [state, action, branch]
    probability: NDArray[np.float64]   # [state, action, branch]
    task_reward: NDArray[np.float64]   # native success reward, before cost
    utility_cost: NDArray[np.float64]  # click-attempt cost
    initial_state: int
    click_mask: NDArray[np.bool_]

    @property
    def reward(self) -> NDArray[np.float64]:
        return self.task_reward - self.utility_cost


@dataclass(frozen=True)
class StochasticSoftSolution:
    q: NDArray[np.float64]
    value: NDArray[np.float64]
    policy: NDArray[np.float64]
    log_policy: NDArray[np.float64]
    energy: NDArray[np.float64]


@dataclass(frozen=True)
class StochasticPolicyEvaluation:
    value: NDArray[np.float64]
    return_value: float
    occupancy: NDArray[np.float64]
    terminal_occupancy_mass: float
    success_probability: float
    discounted_task_reward: float
    discounted_cost: float
    discounted_kl_penalty: float


def controlled_kernel(model, reliability: float, click_cost: float) -> ControlledKernel:
    """Build the exact p/c execution kernel on the native H0 state space.

    A failed click leaves the physical state unchanged, consumes the current
    actor decision, and therefore transitions to the same physical state at the
    next lower remaining-budget level.  Submit actions retain native behavior.
    """

    if not 0 <= reliability <= 1:
        raise ValueError("reliability must lie in [0, 1]")
    if not np.isfinite(click_cost) or click_cost < 0:
        raise ValueError("click_cost must be finite and nonnegative")
    states, actions = model.next_state.shape
    successor = np.full((states, actions, 2), -1, dtype=np.int64)
    probability = np.zeros((states, actions, 2), dtype=np.float64)
    task_reward = np.zeros((states, actions, 2), dtype=np.float64)
    utility_cost = np.zeros((states, actions, 2), dtype=np.float64)
    click_mask = np.asarray([action.startswith("click:") for action in model.task.actions], dtype=bool)

    for state in range(states):
        public = model.state(state)
        for action in range(actions):
            successor[state, action, 0] = model.next_state[state, action]
            task_reward[state, action, 0] = model.task_rewards[state, action]
            if click_mask[action]:
                probability[state, action] = reliability, 1 - reliability
                utility_cost[state, action] = click_cost
                if public.remaining > 1:
                    successor[state, action, 1] = (
                        (public.remaining - 2) * model.task.physical_states + public.physical
                    )
            else:
                probability[state, action, 0] = 1.0

    kernel = ControlledKernel(
        successor,
        probability,
        task_reward,
        utility_cost,
        int(model.initial_state),
        click_mask,
    )
    validate_controlled_kernel(kernel)
    return kernel


def validate_controlled_kernel(kernel: ControlledKernel) -> None:
    shape = kernel.next_state.shape
    if len(shape) != 3 or shape[2] != 2:
        raise ValueError("controlled kernel must have exactly two execution branches")
    for values, name in (
        (kernel.probability, "probability"),
        (kernel.task_reward, "task_reward"),
        (kernel.utility_cost, "utility_cost"),
    ):
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid {name} table")
    if np.any(kernel.probability < 0) or not np.allclose(
        kernel.probability.sum(2), 1.0, rtol=0, atol=1e-14
    ):
        raise ValueError("execution branch probabilities must sum to one")
    rows = np.arange(shape[0])[:, None, None]
    if np.any(kernel.next_state < -1) or np.any(kernel.next_state >= rows):
        raise ValueError("every live successor must precede its source state")
    if not 0 <= kernel.initial_state < shape[0]:
        raise ValueError("initial state is outside the controlled graph")


def controlled_reachable_states(kernel: ControlledKernel) -> NDArray[np.int64]:
    """Reachable nonterminal rows under every positive-probability branch/action."""

    seen, stack = {int(kernel.initial_state)}, [int(kernel.initial_state)]
    while stack:
        state = stack.pop()
        live = kernel.probability[state] > 0
        for successor in kernel.next_state[state][live]:
            successor = int(successor)
            if successor >= 0 and successor not in seen:
                seen.add(successor)
                stack.append(successor)
    return np.asarray(sorted(seen), dtype=np.int64)


def _validate_reference(kernel: ControlledKernel, reference) -> NDArray[np.float64]:
    reference = np.asarray(reference, dtype=np.float64)
    expected = kernel.next_state.shape[:2]
    if reference.shape != expected:
        raise ValueError("reference must have state-by-action shape")
    if np.any(reference <= 0) or not np.all(np.isfinite(reference)):
        raise ValueError("reference probabilities must be finite and positive")
    if not np.allclose(reference.sum(1), 1.0, rtol=0, atol=1e-12):
        raise ValueError("reference rows must sum to one")
    return reference


def solve_controlled(
    kernel: ControlledKernel,
    reference,
    gamma: float = .95,
    tau: float = .1,
) -> StochasticSoftSolution:
    """Exact KL-regularized soft DP for the stochastic two-branch kernel."""

    validate_controlled_kernel(kernel)
    reference = _validate_reference(kernel, reference)
    if not 0 <= gamma < 1 or not np.isfinite(tau) or tau <= 0:
        raise ValueError("require 0 <= gamma < 1 and tau > 0")
    states, actions = reference.shape
    q = np.zeros((states, actions), dtype=np.float64)
    value = np.empty(states, dtype=np.float64)
    energy = np.empty_like(q)
    log_policy = np.empty_like(q)
    reward = kernel.reward
    for state in range(states):
        q[state] = np.sum(kernel.probability[state] * reward[state], axis=1)
        for branch in range(2):
            live = kernel.next_state[state, :, branch] >= 0
            next_rows = kernel.next_state[state, live, branch]
            q[state, live] += (
                gamma * kernel.probability[state, live, branch] * value[next_rows]
            )
        energy[state] = q[state] + tau * np.log(reference[state])
        maximum = float(np.max(energy[state]))
        shifted = (energy[state] - maximum) / tau
        log_normalizer = float(np.log(np.exp(shifted).sum()))
        value[state] = maximum + tau * log_normalizer
        log_policy[state] = shifted - log_normalizer
    return StochasticSoftSolution(q, value, np.exp(log_policy), log_policy, energy)


def evaluate_controlled(
    kernel: ControlledKernel,
    reference,
    policy,
    gamma: float = .95,
    tau: float = .1,
) -> StochasticPolicyEvaluation:
    """Exact fixed-policy value, occupancy, success, and return decomposition."""

    validate_controlled_kernel(kernel)
    reference = _validate_reference(kernel, reference)
    policy = np.asarray(policy, dtype=np.float64)
    if policy.shape != reference.shape or np.any(policy < 0) or not np.all(np.isfinite(policy)):
        raise ValueError("invalid policy table")
    if not np.allclose(policy.sum(1), 1.0, rtol=0, atol=1e-12):
        raise ValueError("policy rows must sum to one")
    log_ratio = np.zeros_like(policy)
    positive = policy > 0
    log_ratio[positive] = np.log(policy[positive]) - np.log(reference[positive])
    expected_reward = np.sum(kernel.probability * kernel.reward, axis=2)
    immediate = np.sum(policy * (expected_reward - tau * log_ratio), axis=1)
    value = immediate.copy()
    for state in range(len(value)):
        for branch in range(2):
            live = kernel.next_state[state, :, branch] >= 0
            successors = kernel.next_state[state, live, branch]
            value[state] += gamma * np.sum(
                policy[state, live]
                * kernel.probability[state, live, branch]
                * value[successors]
            )

    occupancy = np.zeros(len(value), dtype=np.float64)
    occupancy[kernel.initial_state] = 1 - gamma
    exit_mass = 0.0
    for state in range(kernel.initial_state, -1, -1):
        for branch in range(2):
            mass = occupancy[state] * policy[state] * kernel.probability[state, :, branch]
            live = kernel.next_state[state, :, branch] >= 0
            np.add.at(occupancy, kernel.next_state[state, live, branch], gamma * mass[live])
            exit_mass += float(mass[~live].sum())

    visits = occupancy / (1 - gamma)
    expected_task = np.sum(kernel.probability * kernel.task_reward, axis=2)
    expected_cost = np.sum(kernel.probability * kernel.utility_cost, axis=2)
    discounted_task = float(visits @ np.sum(policy * expected_task, axis=1))
    discounted_cost = float(visits @ np.sum(policy * expected_cost, axis=1))
    discounted_kl = float(tau * visits @ np.sum(policy * log_ratio, axis=1))

    success = np.zeros(len(value), dtype=np.float64)
    for state in range(len(value)):
        for action in range(policy.shape[1]):
            for branch in range(2):
                weight = policy[state, action] * kernel.probability[state, action, branch]
                if weight == 0:
                    continue
                success[state] += weight * kernel.task_reward[state, action, branch]
                successor = kernel.next_state[state, action, branch]
                if successor >= 0:
                    success[state] += weight * success[successor]

    return StochasticPolicyEvaluation(
        value=value,
        return_value=float(value[kernel.initial_state]),
        occupancy=occupancy,
        terminal_occupancy_mass=float(gamma * exit_mass / (1 - gamma)),
        success_probability=float(success[kernel.initial_state]),
        discounted_task_reward=discounted_task,
        discounted_cost=discounted_cost,
        discounted_kl_penalty=discounted_kl,
    )


def controlled_metrics(
    kernel: ControlledKernel,
    reference,
    policy,
    gamma: float = .95,
    tau: float = .1,
) -> dict:
    """Return exact independently evaluated C, J, M and companion metrics."""

    policy = np.asarray(policy, dtype=np.float64)
    optimum = solve_controlled(kernel, reference, gamma, tau)
    evaluated = evaluate_controlled(kernel, reference, policy, gamma, tau)
    visits = evaluated.occupancy / (1 - gamma)
    log_policy = np.log(policy)
    mismatch = float(
        tau * visits @ np.sum(policy * (log_policy - optimum.log_policy), axis=1)
    )
    centered_residual = tau * (
        log_policy - optimum.log_policy
        - (log_policy - optimum.log_policy).mean(axis=1, keepdims=True)
    )
    energy_squared = np.sum(centered_residual**2, axis=1)
    ceiling = float(optimum.value[kernel.initial_state])
    return {
        "ceiling": ceiling,
        "regularized_return": evaluated.return_value,
        "mismatch": mismatch,
        "success_probability": evaluated.success_probability,
        "discounted_task_reward": evaluated.discounted_task_reward,
        "discounted_cost": evaluated.discounted_cost,
        "discounted_kl_penalty": evaluated.discounted_kl_penalty,
        "return_decomposition_error": abs(
            evaluated.return_value
            - evaluated.discounted_task_reward
            + evaluated.discounted_cost
            + evaluated.discounted_kl_penalty
        ),
        "gap_identity_error": abs(ceiling - evaluated.return_value - mismatch),
        "nonterminal_occupancy_mass": float(evaluated.occupancy.sum()),
        "terminal_occupancy_mass": evaluated.terminal_occupancy_mass,
        "occupancy_energy_squared": float(evaluated.occupancy @ energy_squared),
        "energy_gap_upper_bound": float(
            evaluated.occupancy @ energy_squared / (4 * tau * (1 - gamma))
        ),
    }
