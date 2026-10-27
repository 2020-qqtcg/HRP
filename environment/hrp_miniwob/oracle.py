"""Exact KL-regularized planning for a finite, deterministic task graph.

Rows are topologically ordered: every successor is either an earlier row or
``-1`` (termination). Including the remaining action budget in the state makes
the controlled MiniWoB graph acyclic. Termination has zero future reward and a
single absorbing action, so its regularization cost is also zero.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SoftSolution:
    q: NDArray[np.float64]
    value: NDArray[np.float64]
    policy: NDArray[np.float64]
    log_policy: NDArray[np.float64]
    energy: NDArray[np.float64]


@dataclass(frozen=True)
class PolicyEvaluation:
    value: NDArray[np.float64]
    return_value: float
    occupancy: NDArray[np.float64]
    terminal_occupancy_mass: float


def _validate_mdp(next_state, rewards, reference, gamma, tau):
    next_state = np.asarray(next_state)
    rewards = np.asarray(rewards, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if next_state.ndim != 2 or 0 in next_state.shape:
        raise ValueError("next_state must be a nonempty state-by-action matrix")
    if next_state.dtype.kind not in "iu":
        raise ValueError("next_state must contain integer state indices")
    if rewards.shape != next_state.shape or reference.shape != next_state.shape:
        raise ValueError("rewards and reference must have the next_state shape")
    if np.any(next_state < -1) or np.any(
        next_state >= np.arange(len(next_state))[:, None]
    ):
        raise ValueError("successors must be -1 or precede their source state")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("rewards must be finite")
    if not np.all(np.isfinite(reference)) or np.any(reference <= 0):
        raise ValueError("reference must have finite, strictly positive probabilities")
    if not np.allclose(reference.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("each reference row must sum to one")
    if not 0 <= gamma < 1 or not np.isfinite(tau) or tau <= 0:
        raise ValueError("require 0 <= gamma < 1 and finite tau > 0")
    return next_state, rewards, reference


def solve(next_state, rewards, reference, gamma, tau) -> SoftSolution:
    """Compute Q*, V*, pi*, log pi*, and F*=Q*+tau log(reference).

    ``rewards`` already includes execution costs. A single ``tau`` and semantic
    ``reference`` define the regularized objective throughout the episode.
    ``log_policy`` remains useful when exponentiation underflows to zero.
    """
    next_state, rewards, reference = _validate_mdp(
        next_state, rewards, reference, gamma, tau
    )
    q = rewards.copy()
    value = np.empty(len(next_state))
    energy = np.empty_like(rewards)
    log_policy = np.empty_like(rewards)
    log_reference = np.log(reference)
    for state, successors in enumerate(next_state):
        live = successors >= 0
        q[state, live] += gamma * value[successors[live]]
        energy[state] = q[state] + tau * log_reference[state]
        maximum = energy[state].max()
        shifted = (energy[state] - maximum) / tau
        log_normalizer = np.log(np.exp(shifted).sum())
        value[state] = maximum + tau * log_normalizer
        log_policy[state] = shifted - log_normalizer
    return SoftSolution(q, value, np.exp(log_policy), log_policy, energy)


def evaluate(
    next_state, rewards, reference, policy, initial_state, gamma, tau
) -> PolicyEvaluation:
    """Evaluate a fixed policy and its discounted occupancy exactly.

    Zero policy probabilities are allowed, using 0*log(0)=0. Energy diagnostics
    involving log(policy) separately require strictly positive probabilities.
    Occupancy includes the (1-gamma) factor and is NOT renormalized over the
    nonterminal states. ``terminal_occupancy_mass`` includes the infinite
    zero-reward absorbing continuation, so both masses together sum to one.
    """
    next_state, rewards, reference = _validate_mdp(
        next_state, rewards, reference, gamma, tau
    )
    policy = np.asarray(policy, dtype=float)
    if policy.shape != next_state.shape:
        raise ValueError("policy must have the next_state shape")
    if not np.all(np.isfinite(policy)) or np.any(policy < 0):
        raise ValueError("policy probabilities must be finite and nonnegative")
    if not np.allclose(policy.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("each policy row must sum to one")
    if not isinstance(initial_state, (int, np.integer)) or not (
        0 <= initial_state < len(next_state)
    ):
        raise ValueError("initial_state must index a nonterminal state")

    log_ratio = np.zeros_like(policy)
    positive = policy > 0
    log_ratio[positive] = np.log(policy[positive]) - np.log(reference[positive])
    immediate = np.sum(policy * (rewards - tau * log_ratio), axis=1)
    value = immediate.copy()
    for state, successors in enumerate(next_state):
        live = successors >= 0
        value[state] += gamma * np.dot(policy[state, live], value[successors[live]])

    occupancy = np.zeros(len(next_state))
    occupancy[initial_state] = 1 - gamma
    exit_mass = 0.0
    for state in range(initial_state, -1, -1):
        successors = next_state[state]
        live = successors >= 0
        mass = occupancy[state] * policy[state]
        np.add.at(occupancy, successors[live], gamma * mass[live])
        exit_mass += mass[~live].sum()
    terminal_mass = gamma * exit_mass / (1 - gamma)
    return PolicyEvaluation(value, float(value[initial_state]), occupancy, float(terminal_mass))
