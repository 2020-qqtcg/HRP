"""Reproducible numerical diagnostics, using synthetic tabular target policies.

These checks test the finite MDP and its oracle. They do not train an LM or
establish that an actual harness satisfies the model's assumptions.
"""

import numpy as np

from .oracle import evaluate, solve


def validate_oracle(
    next_state,
    task_rewards,
    costs,
    reference,
    initial_state,
    gamma,
    tau,
    seed=42,
    n_policies=32,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Check exact identities and return the data needed to reproduce them.

    ``task_rewards`` pays one only on successful termination; ``costs`` can be
    a vector of per-action costs or a full state-by-action array. Generated
    policies perturb oracle log probabilities with N(0, 1.5**2) noise. The
    energy checks require their probabilities to remain representably positive.
    """
    next_state = np.asarray(next_state)
    task_rewards = np.asarray(task_rewards, dtype=float)
    costs = np.broadcast_to(np.asarray(costs, dtype=float), task_rewards.shape)
    reference = np.asarray(reference, dtype=float)
    rewards = task_rewards - costs
    oracle = solve(next_state, rewards, reference, gamma, tau)
    if n_policies < 1:
        raise ValueError("n_policies must be positive")
    tolerance = 1e-10
    rng = np.random.default_rng(seed)

    def summarize(policy, log_policy, evaluation):
        # Forward occupancy gives a separate check of the backward value solve.
        weight = evaluation.occupancy / (1 - gamma)
        discounted_task_reward = float(weight @ np.sum(policy * task_rewards, axis=1))
        discounted_cost = float(weight @ np.sum(policy * costs, axis=1))
        discounted_kl_penalty = float(
            tau * weight @ np.sum(policy * (log_policy - np.log(reference)), axis=1)
        )
        reach_probability = np.zeros(len(next_state))
        reach_probability[initial_state] = 1.0
        for state in range(initial_state, -1, -1):
            successors = next_state[state]
            live = successors >= 0
            np.add.at(
                reach_probability,
                successors[live],
                reach_probability[state] * policy[state, live],
            )
        success_probability = float(
            reach_probability @ np.sum(policy * task_rewards, axis=1)
        )
        decomposition = discounted_task_reward - discounted_cost - discounted_kl_penalty
        return {
            "regularized_return": evaluation.return_value,
            "success_probability": success_probability,
            "discounted_task_reward": discounted_task_reward,
            "discounted_cost": discounted_cost,
            "discounted_kl_penalty": discounted_kl_penalty,
            "decomposition_error": abs(decomposition - evaluation.return_value),
            "nonterminal_occupancy_mass": float(evaluation.occupancy.sum()),
            "terminal_occupancy_mass": evaluation.terminal_occupancy_mass,
        }

    optimum = evaluate(next_state, rewards, reference, oracle.policy, initial_state, gamma, tau)
    reference_evaluation = evaluate(
        next_state, rewards, reference, reference, initial_state, gamma, tau
    )
    baselines = {
        "optimum": summarize(oracle.policy, oracle.log_policy, optimum),
        "reference": summarize(reference, np.log(reference), reference_evaluation),
    }
    logits = oracle.log_policy[None, :, :] + rng.normal(
        0.0, 1.5, size=(n_policies, *reference.shape)
    )
    logits -= logits.max(axis=2, keepdims=True)
    target_log_policies = logits - np.log(np.exp(logits).sum(axis=2, keepdims=True))
    target_policies = np.exp(target_log_policies)
    if np.any(target_policies == 0):
        raise ValueError("generated policies underflowed; energy checks require positive support")

    target_returns = np.empty(n_policies)
    target_mismatches = np.empty(n_policies)
    target_upper_bounds = np.empty(n_policies)
    target_occupancies = np.empty((n_policies, len(next_state)))
    target_terminal_masses = np.empty(n_policies)
    wrong_direction = np.empty(n_policies)
    wrong_occupancy = np.empty(n_policies)
    for index, (policy, log_policy) in enumerate(zip(target_policies, target_log_policies)):
        result = evaluate(next_state, rewards, reference, policy, initial_state, gamma, tau)
        target_returns[index] = result.return_value
        target_occupancies[index] = result.occupancy
        target_terminal_masses[index] = result.terminal_occupancy_mass
        kl = np.sum(policy * (log_policy - oracle.log_policy), axis=1)
        target_mismatches[index] = tau / (1 - gamma) * (result.occupancy @ kl)
        residual = tau * log_policy - oracle.energy
        residual -= residual.mean(axis=1, keepdims=True)
        target_upper_bounds[index] = (
            result.occupancy @ np.sum(residual**2, axis=1) / (4 * tau * (1 - gamma))
        )
        reverse_kl = np.sum(oracle.policy * (oracle.log_policy - log_policy), axis=1)
        wrong_direction[index] = tau / (1 - gamma) * (result.occupancy @ reverse_kl)
        wrong_occupancy[index] = tau / (1 - gamma) * (optimum.occupancy @ kl)

    target_gaps = oracle.value[initial_state] - target_returns
    centered_energy = oracle.energy - oracle.energy.mean(axis=1, keepdims=True)
    centered_log_policy = tau * (
        oracle.log_policy - oracle.log_policy.mean(axis=1, keepdims=True)
    )
    mass_error = np.abs(target_occupancies.sum(axis=1) + target_terminal_masses - 1)
    baseline_mass_error = max(
        abs(item["nonterminal_occupancy_mass"] + item["terminal_occupancy_mass"] - 1)
        for item in baselines.values()
    )
    checks = {
        "tolerance": tolerance,
        "max_optimal_value_evaluation_error": float(np.max(np.abs(optimum.value - oracle.value))),
        "max_gap_identity_error": float(np.max(np.abs(target_gaps - target_mismatches))),
        "max_quotient_energy_error": float(np.max(np.abs(centered_energy - centered_log_policy))),
        "max_energy_upper_bound_violation": float(max(0.0, np.max(target_gaps - target_upper_bounds))),
        "max_occupancy_mass_error": float(max(np.max(mass_error), baseline_mass_error)),
        "max_return_decomposition_error": max(item["decomposition_error"] for item in baselines.values()),
        "max_optimality_violation": float(max(0.0, -np.min(target_gaps))),
    }
    for name, error in checks.items():
        if name != "tolerance" and error > tolerance:
            raise AssertionError(f"{name}={error:.6g} exceeds tolerance {tolerance}")
    report = {
        "configuration": {
            "gamma": gamma,
            "tau": tau,
            "seed": seed,
            "n_policies": n_policies,
            "num_states": len(next_state),
            "num_actions": next_state.shape[1],
            "initial_state": int(initial_state),
            "target_policy_source": "oracle log probabilities plus independent N(0, 1.5^2) noise",
        },
        "baselines": baselines,
        "checks": checks,
        "negative_controls": {
            "max_wrong_kl_direction_gap_error": float(np.max(np.abs(target_gaps - wrong_direction))),
            "max_wrong_oracle_occupancy_gap_error": float(np.max(np.abs(target_gaps - wrong_occupancy))),
            "note": "Reported discrepancies may be zero for degenerate tasks; no nonzero requirement.",
        },
    }
    arrays = {
        "oracle_q": oracle.q,
        "oracle_value": oracle.value,
        "oracle_policy": oracle.policy,
        "oracle_log_policy": oracle.log_policy,
        "oracle_energy": oracle.energy,
        "oracle_occupancy": optimum.occupancy,
        "reference_occupancy": reference_evaluation.occupancy,
        "generated_target_policies": target_policies,
        "generated_target_log_policies": target_log_policies,
        "target_returns": target_returns,
        "target_gaps": target_gaps,
        "target_mismatches": target_mismatches,
        "target_energy_upper_bounds": target_upper_bounds,
        "target_occupancies": target_occupancies,
        "target_terminal_occupancy_masses": target_terminal_masses,
        "wrong_kl_direction_mismatches": wrong_direction,
        "wrong_oracle_occupancy_mismatches": wrong_occupancy,
    }
    return report, arrays
