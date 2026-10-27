"""Undiscounted, finite-horizon control on acyclic MiniWoB task graphs.

The remaining decision budget is part of the state. Visitation is a count of
visits, with no (1-gamma) normalization or infinite absorbing continuation.
"""

import numpy as np


def self_reference_metrics(next_state, rewards, log_policy, initial_state, tau=.1):
    """Solve the optimum and independently evaluate a frozen self-reference actor.

Rewards are binary terminal success, with zero intermediate reward/cost.
The reference is the SAME complete semantic distribution as the evaluated
actor. M is computed from visits and KL to the optimum, never from C-J.
"""
    nxt = np.asarray(next_state)
    reward = np.asarray(rewards, dtype=np.float64)
    lp = np.asarray(log_policy, dtype=np.float64)
    if nxt.ndim != 2 or 0 in nxt.shape or nxt.dtype.kind not in "iu":
        raise ValueError("A nonempty integer state-action transition table is required")
    if reward.shape != nxt.shape or lp.shape != nxt.shape:
        raise ValueError("Rewards and log probabilities must match transitions")
    if np.any(nxt < -1) or np.any(nxt >= np.arange(len(nxt))[:, None]):
        raise ValueError("Finite-horizon successors must precede their source state")
    if not 0 <= initial_state < len(nxt) or not np.isfinite(tau) or tau <= 0:
        raise ValueError("Invalid initial state or regularization coefficient")
    if not np.isin(reward, (0., 1.)).all() or np.any((reward != 0) & (nxt >= 0)):
        raise ValueError("Only binary terminal rewards are supported")
    policy = np.exp(lp)
    if not np.isfinite(lp).all() or np.any(policy <= 0):
        raise ValueError("Full-support, finite semantic probabilities are required")
    if not np.allclose(policy.sum(1), 1., atol=1e-12, rtol=0):
        raise ValueError("Semantic policy must be normalized")

    optimal_value = np.zeros(len(nxt))
    log_optimal = np.empty_like(lp)
    success_value = np.sum(policy * reward, axis=1)
    for state, successors in enumerate(nxt):
        live = successors >= 0
        q = reward[state].copy()
        q[live] += optimal_value[successors[live]]
        logits = lp[state] + q / tau
        normalizer = np.logaddexp.reduce(logits)
        optimal_value[state] = tau * normalizer
        log_optimal[state] = logits - normalizer
        success_value[state] += policy[state, live] @ success_value[successors[live]]

    visits = np.zeros(len(nxt))
    visits[initial_state] = 1.
    terminal_mass = 0.
    for state in range(initial_state, -1, -1):
        live = nxt[state] >= 0
        mass = visits[state] * policy[state]
        np.add.at(visits, nxt[state, live], mass[live])
        terminal_mass += mass[~live].sum()
    # Forward reward accumulation is independent of backward success evaluation.
    actual_return = float(visits @ np.sum(policy * reward, axis=1))
    mismatch = float(tau * visits @ np.sum(policy * (lp - log_optimal), axis=1))
    ceiling = float(optimal_value[initial_state])
    success = float(success_value[initial_state])
    metrics = {
        "ceiling": ceiling, "mismatch": mismatch, "J": actual_return,
        "C_minus_M": ceiling - mismatch, "success_exact": success,
        "reference_kl_penalty": 0., "expected_actor_decisions": float(visits.sum()),
        "terminal_probability": float(terminal_mass),
        "gap_identity_error": abs(ceiling - mismatch - actual_return),
        "success_identity_error": abs(actual_return - success),
    }
    if (metrics["gap_identity_error"] > 1e-10
            or metrics["success_identity_error"] > 1e-10
            or abs(terminal_mass - 1.) > 1e-10
            or mismatch < -1e-10 or not -1e-10 <= ceiling <= 1. + 1e-10):
        raise AssertionError(f"Finite-horizon consistency check failed: {metrics}")
    return metrics, {"visits": visits, "log_optimal": log_optimal,
                     "optimal_value": optimal_value, "success_value": success_value}


def independent_success_rollouts(next_state, rewards, log_policy, initial_state,
                                 horizon, episodes, seed):
    """Independent categorical rollouts on the audited native execution graph."""
    rng = np.random.default_rng(seed)
    cdf = np.exp(log_policy).cumsum(axis=1)
    cdf[:, -1] = 1.
    states = np.full(episodes, initial_state, dtype=np.int64)
    returns = np.zeros(episodes)
    lengths = np.zeros(episodes, dtype=np.int64)
    for _ in range(horizon):
        live = np.flatnonzero(states >= 0)
        if not len(live):
            break
        current = states[live]
        actions = (rng.random(len(live))[:, None] > cdf[current]).sum(axis=1)
        returns[live] += rewards[current, actions]
        lengths[live] += 1
        states[live] = next_state[current, actions]
    if np.any(states >= 0) or not np.isin(returns, (0., 1.)).all():
        raise AssertionError("Rollout violated the finite binary-success protocol")
    return returns, lengths
