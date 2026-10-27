"""Policy measurements on the independently browser-validated finite kernel.

All policies are state-by-action probability tables. Energy and policy-gradient
measurements require strictly positive policies, as produced by the LM softmax.
Rollout results are compiled-kernel simulations, not fresh Chromium executions.
Pilot prediction and return evaluation use separate random-number streams even
when the caller supplies the same integer seed.
"""

import numpy as np

from .model import TaskModel
from .oracle import evaluate, solve
from .suite import reachable_states


def readout_drift(log_policy_a, log_policy_b, tau: float) -> np.ndarray:
    """Signed centered energy difference per state/action, A minus B.

    Action columns must already be aligned by their complete semantic meaning.
    Centering removes arbitrary state-dependent logit offsets. This function
    returns the residual array; callers choose a fixed bank or occupancy measure.
    """
    difference = tau * (np.asarray(log_policy_a) - np.asarray(log_policy_b))
    return difference - difference.mean(axis=1, keepdims=True)


def policy_metrics(model: TaskModel, policy, costs, reference,
                   gamma: float = 0.95, tau: float = 0.1) -> dict:
    """JSON-compatible exact values, success, and oracle energy diagnostics.

    ``fixed_bank_energy_mse`` averages over reachable states AND actions;
    ``fixed_bank_energy_squared`` averages the action-L2 squared residual over
    reachable states. ``occupancy_energy_squared`` uses d^pi=(1-gamma) sum_t
    gamma^t Pr(s_t=s), without renormalizing its nonterminal mass.
    ``kl_mismatch`` is computed from occupancy and KL(pi || pi*), independently
    of ``gap=ceiling-regularized_return``.
    """
    policy = np.asarray(policy, dtype=float)
    if np.any(policy <= 0):
        raise ValueError("energy diagnostics require strictly positive policies")
    costs = np.broadcast_to(costs, model.task_rewards.shape)
    result = evaluate(model.next_state, model.task_rewards - costs, reference,
                      policy, model.initial_state, gamma, tau)
    optimum = solve(model.next_state, model.task_rewards - costs, reference, gamma, tau)
    log_policy = np.log(policy)
    log_ratio = log_policy - np.log(reference)
    visitation = result.occupancy / (1 - gamma)
    task_reward = float(visitation @ np.sum(policy * model.task_rewards, axis=1))
    cost = float(visitation @ np.sum(policy * costs, axis=1))
    kl_penalty = float(tau * visitation @ np.sum(policy * log_ratio, axis=1))

    # Undiscounted probability of the task's single terminal success event.
    success = np.sum(policy * model.task_rewards, axis=1)
    for state, successors in enumerate(model.next_state):
        live = successors >= 0
        success[state] += policy[state, live] @ success[successors[live]]

    residual = readout_drift(log_policy, optimum.log_policy, tau)
    squared = np.sum(residual**2, axis=1)
    bank = reachable_states(model)
    occupancy_energy = float(result.occupancy @ squared)
    mismatch = float(tau * visitation @ np.sum(
        policy * (log_policy - optimum.log_policy), axis=1))
    ceiling = float(optimum.value[model.initial_state])
    gap = ceiling - result.return_value
    return {
        "regularized_return": result.return_value,
        "success_probability": float(success[model.initial_state]),
        "discounted_task_reward": task_reward,
        "discounted_cost": cost,
        "discounted_kl_penalty": kl_penalty,
        "return_decomposition_error": abs(result.return_value - task_reward + cost + kl_penalty),
        "ceiling": ceiling,
        "gap": gap,
        "kl_mismatch": mismatch,
        "gap_identity_error": abs(gap - mismatch),
        "reachable_state_count": int(len(bank)),
        "fixed_bank_energy_mse": float(np.mean(residual[bank]**2)),
        "fixed_bank_energy_squared": float(np.mean(squared[bank])),
        "occupancy_energy_squared": occupancy_energy,
        "energy_gap_upper_bound": occupancy_energy / (4 * tau * (1 - gamma)),
        "nonterminal_occupancy_mass": float(result.occupancy.sum()),
        "terminal_occupancy_mass": result.terminal_occupancy_mass,
    }


def policy_gradient_weights(model: TaskModel, policy, costs, reference,
                            gamma: float = 0.95, tau: float = 0.1):
    """Return discounted visitation[S] and current-policy advantage[S,A].

    No optimal-policy solver is used. For logits z and pi=softmax(z), the exact
    gradient of J is weight[:,None] * pi * advantage. When training an LM, the
    weights and advantages must be detached before a weighted log-pi objective.
    """
    policy = np.asarray(policy, dtype=float)
    if np.any(policy <= 0):
        raise ValueError("policy gradients require strictly positive policies")
    rewards = model.task_rewards - np.asarray(costs)
    result = evaluate(model.next_state, rewards, reference, policy,
                      model.initial_state, gamma, tau)
    advantage = rewards - tau * (np.log(policy) - np.log(reference))
    for state, successors in enumerate(model.next_state):
        live = successors >= 0
        advantage[state, live] += gamma * result.value[successors[live]]
    advantage -= result.value[:, None]
    return result.occupancy / (1 - gamma), advantage


def sample_rollouts(model: TaskModel, policy, costs, reference,
                    gamma: float = 0.95, tau: float = 0.1,
                    episodes: int = 256, seed: int = 0) -> dict:
    """Sample fresh evaluation trajectories from the compiled task kernel.

    ``trajectories`` records each regularized return, binary success, number of
    actions, and state/action indices. ``discounted_occupancy`` is the Monte Carlo
    mean of (1-gamma) sum_t gamma^t 1{s_t=s}. Standard errors describe rollout
    noise for this fixed policy/task, not uncertainty across task or training seeds.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive")
    policy = np.asarray(policy, dtype=float)
    costs = np.broadcast_to(costs, model.task_rewards.shape)
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0,)))
    occupancy = np.zeros(len(model.next_state))
    trajectories = []
    for _ in range(episodes):
        state = model.initial_state
        discount, total, success = 1.0, 0.0, 0.0
        states, actions = [], []
        while state >= 0:
            action = int(rng.choice(policy.shape[1], p=policy[state]))
            occupancy[state] += (1 - gamma) * discount
            reward = model.task_rewards[state, action]
            regularized = (reward - costs[state, action]
                           - tau * np.log(policy[state, action] / reference[state, action]))
            total += discount * regularized
            success += reward
            states.append(int(state))
            actions.append(action)
            state = int(model.next_state[state, action])
            discount *= gamma
        trajectories.append({
            "regularized_return": float(total), "success": int(success),
            "steps": len(actions), "states": states, "actions": actions,
        })
    returns = np.array([x["regularized_return"] for x in trajectories])
    successes = np.array([x["success"] for x in trajectories])
    return {
        "backend": "compiled-kernel-rollout",
        "rng_stream": "evaluation-spawn-key-0",
        "seed": seed,
        "episodes": episodes,
        "trajectories": trajectories,
        "mean_return": float(returns.mean()),
        "return_standard_error": float(returns.std(ddof=1) / np.sqrt(episodes)) if episodes > 1 else None,
        "success_rate": float(successes.mean()),
        "success_standard_error": float(successes.std(ddof=1) / np.sqrt(episodes)) if episodes > 1 else None,
        "mean_steps": float(np.mean([x["steps"] for x in trajectories])),
        "discounted_occupancy": (occupancy / episodes).tolist(),
    }


def pilot_prediction(model: TaskModel, policy, costs, reference,
                     gamma: float = 0.95, tau: float = 0.1,
                     episodes: int = 256, seed: int = 0) -> dict:
    """Predict J = C - M using oracle C and pilot-trajectory state KL costs.

    Rewards are read only by the oracle construction. Pilot sampling reads the
    policy and transition kernel, and accumulates tau*gamma^t KL(pi||pi*) at each
    visited state. It never observes an evaluation return or computes M as C-J.
    ``trajectory_mismatch`` and ``discounted_occupancy`` permit direct auditing.
    Prediction SE is the pilot sample-mean SE, conditional on the exact oracle
    and action probability table; it excludes training/task/readout uncertainty.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive")
    policy = np.asarray(policy, dtype=float)
    optimum = solve(model.next_state, model.task_rewards - np.asarray(costs),
                    reference, gamma, tau)
    log_ratio = np.zeros_like(policy)
    positive = policy > 0
    log_ratio[positive] = np.log(policy[positive]) - optimum.log_policy[positive]
    state_kl = np.sum(policy * log_ratio, axis=1)
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
    occupancy = np.zeros(len(model.next_state))
    mismatch = np.zeros(episodes)
    for episode in range(episodes):
        state, discount = model.initial_state, 1.0
        while state >= 0:
            occupancy[state] += (1 - gamma) * discount
            mismatch[episode] += tau * discount * state_kl[state]
            action = int(rng.choice(policy.shape[1], p=policy[state]))
            state = int(model.next_state[state, action])
            discount *= gamma
    ceiling = float(optimum.value[model.initial_state])
    return {
        "backend": "compiled-kernel-pilot",
        "rng_stream": "pilot-spawn-key-1",
        "seed": seed,
        "episodes": episodes,
        "ceiling": ceiling,
        "pilot_mismatch": float(mismatch.mean()),
        "predicted_return": float(ceiling - mismatch.mean()),
        "prediction_standard_error": float(mismatch.std(ddof=1) / np.sqrt(episodes)) if episodes > 1 else None,
        "trajectory_mismatch": mismatch.tolist(),
        "discounted_occupancy": (occupancy / episodes).tolist(),
    }
