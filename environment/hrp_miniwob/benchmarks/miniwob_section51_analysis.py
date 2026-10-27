"""Finite-sample diagnostics, sealed predictions, exact scoring, and reporting.

Prediction functions only receive categorical sampling and transition methods;
they do not receive exact actor occupancy, exact mismatch, return, or success.
Exact scores are produced in separate commands after prediction files have been
hashed and sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from hrp_miniwob.adapters.miniwob_section51 import (
    ControlledKernel,
    controlled_kernel,
    controlled_metrics,
    controlled_reachable_states,
    solve_controlled,
)
from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_section51 as study
from hrp_miniwob.benchmarks.miniwob_meta_harness import digest
from hrp_miniwob.oracle import evaluate, solve
from hrp_miniwob.suite import reachable_states

MS = (4, 16, 64)
NS = (16, 64, 256)
ALPHAS = (.1, .5, 1.0)
MAIN_ALPHA = .5
DIRECT_LOG_SAMPLES = 4
CALIBRATION_REPEATS = 100
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 5205


def centered(values):
    values = np.asarray(values, dtype=np.float64)
    return values - values.mean(axis=-1, keepdims=True)


def stable_seed(base: int, *parts) -> int:
    payload = json.dumps([base, *parts], sort_keys=True, ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def deterministic_exact(next_state, rewards, reference, policy, initial_state):
    optimum = solve(next_state, rewards, reference, study.GAMMA, study.TAU)
    result = evaluate(next_state, rewards, reference, policy, initial_state, study.GAMMA, study.TAU)
    visits = result.occupancy / (1 - study.GAMMA)
    log_policy = np.log(policy)
    mismatch = float(
        study.TAU * visits @ np.sum(policy * (log_policy - optimum.log_policy), axis=1)
    )
    success = np.sum(policy * rewards, axis=1)
    for state, successors in enumerate(next_state):
        live = successors >= 0
        success[state] += policy[state, live] @ success[successors[live]]
    log_reference = np.log(reference)
    discounted_reward = float(visits @ np.sum(policy * rewards, axis=1))
    discounted_kl = float(study.TAU * visits @ np.sum(policy * (log_policy - log_reference), axis=1))
    residual = study.TAU * centered(log_policy - optimum.log_policy)
    squared = np.sum(residual**2, axis=1)
    ceiling = float(optimum.value[initial_state])
    return {
        "ceiling": ceiling,
        "regularized_return": result.return_value,
        "mismatch": mismatch,
        "success_probability": float(success[initial_state]),
        "discounted_task_reward": discounted_reward,
        "discounted_cost": 0.0,
        "discounted_kl_penalty": discounted_kl,
        "occupancy": result.occupancy,
        "terminal_occupancy_mass": result.terminal_occupancy_mass,
        "optimal_policy": optimum.policy,
        "optimal_log_policy": optimum.log_policy,
        "energy_residual": residual,
        "energy_upper_bound": float(result.occupancy @ squared / (4 * study.TAU * (1 - study.GAMMA))),
        "gap_identity_error": abs(ceiling - result.return_value - mismatch),
        "return_decomposition_error": abs(result.return_value - discounted_reward + discounted_kl),
    }


@dataclass(frozen=True)
class SamplingEnvironment:
    """The only deployment information exposed to finite-sample estimators."""

    policy: np.ndarray
    initial_state: int
    next_state: np.ndarray | None = None
    controlled: ControlledKernel | None = None

    def __post_init__(self):
        if (self.next_state is None) == (self.controlled is None):
            raise ValueError("Provide exactly one deterministic or controlled transition kernel")
        if np.any(self.policy <= 0) or not np.allclose(self.policy.sum(1), 1, atol=1e-12, rtol=0):
            raise ValueError("Sampling service requires a positive normalized policy")

    def action(self, state: int, rng: np.random.Generator) -> int:
        return int(rng.choice(self.policy.shape[1], p=self.policy[state]))

    def successor(self, state: int, action: int, rng: np.random.Generator) -> int:
        if self.next_state is not None:
            return int(self.next_state[state, action])
        probabilities = self.controlled.probability[state, action]
        branch = int(rng.choice(2, p=probabilities))
        return int(self.controlled.next_state[state, action, branch])

    def action_counts(self, state: int, increments: tuple[int, ...], rng: np.random.Generator):
        counts = np.zeros(self.policy.shape[1], dtype=np.int64)
        rows = []
        for count in increments:
            counts = counts + rng.multinomial(count, self.policy[state])
            rows.append(counts.copy())
        return rows


def geometric_states(environment: SamplingEnvironment, count: int, rng: np.random.Generator):
    """Independent exact samples from discounted occupancy including absorption."""

    states = np.full(count, -1, dtype=np.int64)
    pilot_decisions = np.zeros(count, dtype=np.int64)
    times = rng.geometric(1 - study.GAMMA, size=count) - 1
    for row, target_time in enumerate(times):
        state = environment.initial_state
        for _ in range(int(target_time)):
            if state < 0:
                break
            action = environment.action(state, rng)
            state = environment.successor(state, action, rng)
            pilot_decisions[row] += 1
        states[row] = state
    return states, pilot_decisions


def smoothed_policy(counts, alpha):
    counts = np.asarray(counts, dtype=np.float64)
    return (counts + alpha) / (counts.sum(axis=-1, keepdims=True) + alpha * counts.shape[-1])


def state_diagnostic(estimated_policy, optimal_log_policy, true_log_policy=None):
    log_estimated = np.log(estimated_policy)
    kl = float(np.sum(estimated_policy * (log_estimated - optimal_log_policy)))
    estimated_energy = study.TAU * centered(log_estimated)
    target_energy = study.TAU * centered(optimal_log_policy)
    residual = estimated_energy - target_energy
    result = {
        "kl": kl,
        "energy_squared": float(np.sum(residual**2)),
        "energy_mse": float(np.mean(residual**2)),
    }
    if true_log_policy is not None:
        readout_error = estimated_energy - study.TAU * centered(true_log_policy)
        result["readout_mse"] = float(np.mean(readout_error**2))
    return result


def nested_probe(
    environment: SamplingEnvironment,
    optimal_log_policy,
    rng: np.random.Generator,
    ms=MS,
    ns=NS,
    alpha=MAIN_ALPHA,
    independent_noise=True,
):
    """Return all nested m/N points from one finite-sampling pool."""

    states, pilot = geometric_states(environment, max(ms), rng)
    increments = (ns[0], ns[1] - ns[0], ns[2] - ns[1])
    first, second = {}, {}
    for row, state in enumerate(states):
        if state < 0:
            continue
        first[row] = environment.action_counts(int(state), increments, rng)
        if independent_noise:
            second[row] = environment.action_counts(int(state), increments, rng)
    results = {}
    factor = study.TAU / (1 - study.GAMMA)
    for m in ms:
        for n_index, n in enumerate(ns):
            kl, energy, readout, noise = [], [], [], []
            zeros, nonterminal = 0, 0
            for row, state in enumerate(states[:m]):
                if state < 0:
                    kl.append(0.0)
                    energy.append(0.0)
                    continue
                counts = first[row][n_index]
                estimated = smoothed_policy(counts, alpha)
                metrics = state_diagnostic(
                    estimated,
                    optimal_log_policy[state],
                    np.log(environment.policy[state]),
                )
                kl.append(metrics["kl"])
                energy.append(metrics["energy_squared"])
                readout.append(metrics["readout_mse"])
                zeros += int(np.sum(counts == 0))
                nonterminal += counts.size
                if independent_noise:
                    other = smoothed_policy(second[row][n_index], alpha)
                    delta = study.TAU * centered(np.log(estimated) - np.log(other))
                    noise.append(float(np.mean(delta**2)))
            results[m, n] = {
                "mismatch": factor * float(np.mean(kl)),
                "energy_upper_bound": float(np.mean(energy)) / (4 * study.TAU * (1 - study.GAMMA)),
                "readout_mse": float(np.mean(readout)) if readout else 0.0,
                "sampling_noise_mse": float(np.mean(noise)) if noise else 0.0,
                "zero_count_fraction": zeros / nonterminal if nonterminal else 0.0,
                "absorbing_fraction": float(np.mean(states[:m] < 0)),
                "pilot_decisions": int(pilot[:m].sum()),
                "action_samples": int(np.sum(states[:m] >= 0) * n),
                "nominal_action_samples": int(m * n),
                "states": states[:m].copy(),
            }
    return results


def uniform_probe(
    environment: SamplingEnvironment,
    bank,
    occupancy,
    optimal_log_policy,
    rng,
    ms=MS,
    ns=NS,
    alpha=MAIN_ALPHA,
):
    indices = rng.integers(0, len(bank), size=max(ms))
    states = np.asarray(bank)[indices]
    increments = (ns[0], ns[1] - ns[0], ns[2] - ns[1])
    tables = [environment.action_counts(int(state), increments, rng) for state in states]
    results = {}
    factor = study.TAU / (1 - study.GAMMA)
    for m in ms:
        for n_index, n in enumerate(ns):
            kl = []
            weighted = []
            for row, state in enumerate(states[:m]):
                estimated = smoothed_policy(tables[row][n_index], alpha)
                value = state_diagnostic(estimated, optimal_log_policy[state])["kl"]
                kl.append(value)
                weighted.append(len(bank) * occupancy[state] * value)
            results[m, n] = {
                "uniform_naive_mismatch": factor * float(np.mean(kl)),
                "uniform_importance_mismatch": factor * float(np.mean(weighted)),
            }
    return results


def finite_certificate(estimated_rms, m, n, action_count, p0, radius, delta=.05):
    """Numerically instantiate Proposition finite-sample with occupancy sampling."""

    b_n = math.sqrt(math.log(4 * m * action_count / delta) / (2 * n)) + 1 / (2 * n)
    applicable = bool(b_n <= p0 / 2)
    if not applicable:
        return {"applicable": False, "b_N": b_n, "p0": p0, "reason": "b_N > p0/2"}
    # Star graph: q=n-1 and lambda_2=1. Oracle target gives q_*=0;
    # exact discounted occupancy sampling gives W=1.
    kappa = 4 * study.TAU * b_n / p0
    k_graph = math.sqrt(action_count - 1) * kappa
    bound = (
        (estimated_rms + k_graph) ** 2
        + radius**2 * math.sqrt(math.log(2 / delta) / (2 * m))
    ) / (4 * study.TAU * (1 - study.GAMMA))
    return {
        "applicable": True,
        "b_N": b_n,
        "p0": p0,
        "R": radius,
        "W": 1.0,
        "q_star": 0.0,
        "graph": "star",
        "K_G": k_graph,
        "bound": bound,
    }


def graph_reconstruction(counts, alpha=MAIN_ALPHA):
    """Reconstruct the same centered log frequency through four action graphs."""

    probabilities = smoothed_policy(np.asarray(counts), alpha)
    target = centered(study.TAU * np.log(probabilities))
    n = len(probabilities)

    def incidence(edges):
        matrix = np.zeros((len(edges), n), dtype=np.float64)
        for row, (left, right) in enumerate(edges):
            matrix[row, left], matrix[row, right] = 1, -1
        return matrix

    graphs = {
        "star": [(0, i) for i in range(1, n)],
        "chain": [(i, i + 1) for i in range(n - 1)],
        "complete": [(i, j) for i in range(n) for j in range(i + 1, n)],
        "disconnected": [(i, i + 1) for i in range(max(0, n // 2 - 1))]
                        + [(i, i + 1) for i in range(n // 2, n - 1)],
    }
    result = {}
    for name, edges in graphs.items():
        matrix = incidence(edges)
        laplacian = matrix.T @ matrix
        reconstruction = np.linalg.pinv(laplacian) @ matrix.T @ (matrix @ target)
        eigenvalues = np.linalg.eigvalsh(laplacian)
        components = int(np.sum(eigenvalues < 1e-10))
        result[name] = {
            "edges": len(edges),
            "components": components,
            "max_centered_error": float(np.max(np.abs(centered(reconstruction) - target))),
            "identifies_full_centered_coordinate": components == 1,
            "frequency_table_reused": True,
        }
    return result


def family_macro(values, families, reducer=np.mean):
    return float(np.mean([reducer(np.asarray(values)[np.asarray(families) == family]) for family in study.FAMILIES]))


def _b_task_inputs(experiment, model, rendering):
    reference_table = study.load_table(experiment, "calibration", "B_base_R0", model)
    actor_table = study.load_table(experiment, "calibration", f"B_h0_final_{rendering}", model)
    reference = np.exp(study.full_log_policy(reference_table))
    policy = np.exp(study.full_log_policy(actor_table))
    exact = deterministic_exact(
        model.next_state,
        model.task_rewards,
        reference,
        policy,
        model.initial_state,
    )
    return reference, policy, exact


def analyze_b_calibration(experiment=study.EXPERIMENT):
    experiment = Path(experiment).resolve()
    output = experiment / "calibration/B"
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "registered_before_sampling_analysis",
        "grid": {"m": list(MS), "N": list(NS)},
        "alpha": MAIN_ALPHA,
        "repeats": CALIBRATION_REPEATS,
        "main_budget_candidates": [[4, 256], [16, 64], [64, 16]],
        "selection": "lowest family-macro M RMSE across R0/R1/R3; exact tie chooses smaller m then smaller N",
        "state_sampling": "one independent geometric-time state from each independent pilot trajectory; absorption contributes zero",
        "action_sampling": "nested multinomial increments representing prefixes 16,64,256",
        "registered_at_utc": datetime.now(UTC).isoformat(),
    }
    protocol_path = experiment / "calibration/B_protocol.json"
    if protocol_path.exists():
        old = study.read(protocol_path)
        protocol["registered_at_utc"] = old["registered_at_utc"]
        if old != protocol:
            raise ValueError("B calibration protocol changed")
    else:
        write_json(protocol_path, protocol)

    models = study.load_models(experiment, "calibration")
    shape = (len(models), len(study.B_RENDERINGS), len(MS), len(NS), CALIBRATION_REPEATS)
    fields = {
        name: np.full(shape, np.nan, dtype=np.float64)
        for name in (
            "mismatch", "energy_upper_bound", "readout_mse", "sampling_noise_mse",
            "zero_count_fraction", "absorbing_fraction", "uniform_naive_mismatch",
            "uniform_importance_mismatch", "certificate_bound",
        )
    }
    exact_mismatch = np.empty((len(models), len(study.B_RENDERINGS)), dtype=np.float64)
    exact_upper = np.empty_like(exact_mismatch)
    certificate_applicable = np.zeros(shape, dtype=bool)
    p0 = np.empty_like(exact_mismatch)
    radius = np.empty_like(exact_mismatch)
    started = time.monotonic()
    example_graphs = None
    alpha_rows = []

    for task_index, model in enumerate(models):
        bank = reachable_states(model)
        for rendering_index, rendering in enumerate(study.B_RENDERINGS):
            _, policy, exact = _b_task_inputs(experiment, model, rendering)
            environment = SamplingEnvironment(policy, model.initial_state, next_state=model.next_state)
            exact_mismatch[task_index, rendering_index] = exact["mismatch"]
            exact_upper[task_index, rendering_index] = exact["energy_upper_bound"]
            p0[task_index, rendering_index] = float(np.min(policy[bank]))
            radius[task_index, rendering_index] = float(
                np.max(np.linalg.norm(exact["energy_residual"][bank], axis=1))
            )
            for repeat in range(CALIBRATION_REPEATS):
                rng = np.random.default_rng(stable_seed(
                    5101, model.task.fingerprint, rendering, repeat, "occupancy"
                ))
                pool = nested_probe(environment, exact["optimal_log_policy"], rng)
                uniform = uniform_probe(
                    environment,
                    bank,
                    exact["occupancy"],
                    exact["optimal_log_policy"],
                    np.random.default_rng(stable_seed(
                        5101, model.task.fingerprint, rendering, repeat, "uniform"
                    )),
                )
                for mi, m in enumerate(MS):
                    for ni, n in enumerate(NS):
                        row = pool[m, n] | uniform[m, n]
                        for field in (
                            "mismatch", "energy_upper_bound", "readout_mse", "sampling_noise_mse",
                            "zero_count_fraction", "absorbing_fraction", "uniform_naive_mismatch",
                            "uniform_importance_mismatch",
                        ):
                            fields[field][task_index, rendering_index, mi, ni, repeat] = row[field]
                        certificate = finite_certificate(
                            math.sqrt(max(0.0, row["energy_upper_bound"] * 4 * study.TAU * (1 - study.GAMMA))),
                            m, n, policy.shape[1], p0[task_index, rendering_index],
                            radius[task_index, rendering_index],
                        )
                        if certificate["applicable"]:
                            certificate_applicable[task_index, rendering_index, mi, ni, repeat] = True
                            fields["certificate_bound"][task_index, rendering_index, mi, ni, repeat] = certificate["bound"]

                if example_graphs is None:
                    sample_state = int(pool[64, 256]["states"][np.flatnonzero(pool[64, 256]["states"] >= 0)[0]])
                    counts = np.random.default_rng(91).multinomial(256, policy[sample_state])
                    example_graphs = graph_reconstruction(counts)

            # Alpha sensitivity uses the same fixed seed and largest pool, once
            # per task/rendering/alpha to avoid treating alphas as new samples.
            for alpha in ALPHAS:
                rng = np.random.default_rng(stable_seed(5101, model.task.fingerprint, rendering, 10_000, alpha))
                row = nested_probe(environment, exact["optimal_log_policy"], rng, alpha=alpha)[64, 256]
                alpha_rows.append({
                    "task_seed": model.task.seed, "family": model.task.family,
                    "rendering": rendering, "alpha": alpha,
                    "mismatch_error": row["mismatch"] - exact["mismatch"],
                    "zero_count_fraction": row["zero_count_fraction"],
                })
        write_json(output / "progress.json", {
            "status": "running", "tasks": task_index + 1, "total": len(models),
            "wall_seconds": time.monotonic() - started,
        })

    raw_path = output / "sampling_arrays.npz"
    np.savez_compressed(
        raw_path,
        **fields,
        exact_mismatch=exact_mismatch,
        exact_energy_upper_bound=exact_upper,
        certificate_applicable=certificate_applicable,
        p0=p0,
        radius=radius,
        task_seed=np.asarray([model.task.seed for model in models]),
        family=np.asarray([model.task.family for model in models]),
        renderings=np.asarray(study.B_RENDERINGS),
        ms=np.asarray(MS),
        ns=np.asarray(NS),
    )
    families = np.asarray([model.task.family for model in models])
    summaries = {}
    for rendering_index, rendering in enumerate(study.B_RENDERINGS):
        summaries[rendering] = {}
        for mi, m in enumerate(MS):
            for ni, n in enumerate(NS):
                estimates = fields["mismatch"][:, rendering_index, mi, ni]
                error = estimates - exact_mismatch[:, rendering_index, None]
                squared_per_task = np.mean(error**2, axis=1)
                bias_per_task = np.mean(error, axis=1)
                cert = fields["certificate_bound"][:, rendering_index, mi, ni]
                applicable = certificate_applicable[:, rendering_index, mi, ni]
                valid = applicable & np.isfinite(cert)
                coverage = np.mean(
                    cert[valid] >= np.broadcast_to(exact_mismatch[:, rendering_index, None], cert.shape)[valid]
                ) if np.any(valid) else None
                summaries[rendering][f"m{m}_N{n}"] = {
                    "family_macro_bias": family_macro(bias_per_task, families),
                    "family_macro_rmse": math.sqrt(family_macro(squared_per_task, families)),
                    "mean_readout_mse": family_macro(
                        fields["readout_mse"][:, rendering_index, mi, ni].mean(1), families
                    ),
                    "mean_sampling_noise_mse": family_macro(
                        fields["sampling_noise_mse"][:, rendering_index, mi, ni].mean(1), families
                    ),
                    "mean_zero_count_fraction": family_macro(
                        fields["zero_count_fraction"][:, rendering_index, mi, ni].mean(1), families
                    ),
                    "energy_bound_to_true_M_mean_ratio": family_macro(
                        fields["energy_upper_bound"][:, rendering_index, mi, ni].mean(1)
                        / np.maximum(exact_mismatch[:, rendering_index], 1e-15),
                        families,
                    ),
                    "uniform_naive_rmse": math.sqrt(family_macro(
                        np.mean((fields["uniform_naive_mismatch"][:, rendering_index, mi, ni]
                                 - exact_mismatch[:, rendering_index, None]) ** 2, axis=1), families
                    )),
                    "uniform_importance_rmse": math.sqrt(family_macro(
                        np.mean((fields["uniform_importance_mismatch"][:, rendering_index, mi, ni]
                                 - exact_mismatch[:, rendering_index, None]) ** 2, axis=1), families
                    )),
                    "certificate_applicable_fraction": float(np.mean(applicable)),
                    "certificate_conditional_coverage": float(coverage) if coverage is not None else None,
                }

    candidates = ((4, 256), (16, 64), (64, 16))
    candidate_scores = {}
    for m, n in candidates:
        mi, ni = MS.index(m), NS.index(n)
        per_task = []
        for task_index in range(len(models)):
            errors = []
            for rendering_index in range(len(study.B_RENDERINGS)):
                delta = fields["mismatch"][task_index, rendering_index, mi, ni] - exact_mismatch[task_index, rendering_index]
                errors.extend(delta.tolist())
            per_task.append(float(np.mean(np.square(errors))))
        candidate_scores[f"m{m}_N{n}"] = math.sqrt(family_macro(per_task, families))
    selected = min(candidates, key=lambda pair: (candidate_scores[f"m{pair[0]}_N{pair[1]}"], pair[0], pair[1]))
    selection = {
        "status": "complete",
        "selected": {"m": selected[0], "N": selected[1], "alpha": MAIN_ALPHA},
        "candidate_family_macro_rmse": candidate_scores,
        "selection_rule": protocol["selection"],
        "all_grid": summaries,
        "graph_reconstruction": example_graphs,
        "alpha_sensitivity": {
            str(alpha): {
                "family_macro_bias": family_macro(
                    [row["mismatch_error"] for row in alpha_rows if row["alpha"] == alpha],
                    [row["family"] for row in alpha_rows if row["alpha"] == alpha],
                ),
                "mean_zero_count_fraction": family_macro(
                    [row["zero_count_fraction"] for row in alpha_rows if row["alpha"] == alpha],
                    [row["family"] for row in alpha_rows if row["alpha"] == alpha],
                ),
            } for alpha in ALPHAS
        },
        "raw_arrays": str(raw_path),
        "raw_arrays_sha256": digest(raw_path),
        "protocol_sha256": digest(protocol_path),
        "wall_seconds": time.monotonic() - started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(output / "selection.json", selection)
    write_json(output / "progress.json", {"status": "complete", "selected": selection["selected"]})
    print(json.dumps({"selected": selection["selected"], "scores": candidate_scores}), flush=True)
    return selection


def execution_configurations(partition=None):
    """Return the 51 preregistered rendering/execution configurations."""

    proposal = study.proposal()["execution_sweeps"]
    rows = []
    for rendering in study.C_RENDERINGS:
        for reliability in proposal["reliability"]:
            rows.append({
                "id": f"p{round(reliability * 10000):05d}_c00100_{rendering}",
                "rendering": rendering,
                "reliability": float(reliability),
                "click_cost": .01,
                "sweep": "reliability",
            })
        for cost in proposal["cost"]:
            if math.isclose(cost, .01):
                continue
            rows.append({
                "id": f"p08000_c{round(cost * 10000):05d}_{rendering}",
                "rendering": rendering,
                "reliability": .8,
                "click_cost": float(cost),
                "sweep": "cost",
            })
    if len(rows) != 51 or len({row["id"] for row in rows}) != 51:
        raise AssertionError("Execution sweep must contain 51 unique configurations")
    for row in rows:
        calibration = (
            math.isclose(row["click_cost"], .01)
            and row["reliability"] in proposal["calibration_reliability"]
        ) or (
            math.isclose(row["reliability"], .8)
            and row["click_cost"] in proposal["calibration_cost"]
        )
        row["partition"] = "calibration" if calibration else "test"
    if sum(row["partition"] == "calibration" for row in rows) != 15:
        raise AssertionError("Expected 15 calibration execution configurations")
    if sum(row["partition"] == "test" for row in rows) != 36:
        raise AssertionError("Expected 36 held-out execution configurations")
    return [row for row in rows if partition is None or row["partition"] == partition]


def _c_tables(experiment, split, model, model_id, rendering):
    reference_table = study.load_table(experiment, split, "C_base_R0", model)
    actor_table = study.load_table(experiment, split, f"C_{model_id}_{rendering}", model)
    reference = np.exp(study.full_log_policy(reference_table))
    policy = np.exp(study.full_log_policy(actor_table))
    if not np.array_equal(reference_table["next_state"], model.next_state):
        raise AssertionError("C reference table uses the wrong semantic graph")
    if not np.array_equal(actor_table["next_state"], model.next_state):
        raise AssertionError("C actor table uses the wrong semantic graph")
    return reference, policy


def _controlled_oracle(model, reference, reliability, click_cost):
    kernel = controlled_kernel(model, reliability, click_cost)
    optimum = solve_controlled(kernel, reference, study.GAMMA, study.TAU)
    return kernel, optimum


def single_probe(environment, optimal_log_policy, m, n, alpha, rng, uniform_bank=None):
    states, pilot = geometric_states(environment, m, rng)
    factor = study.TAU / (1 - study.GAMMA)
    kls, energy, zeros, total_actions = [], [], 0, 0
    effective = []
    for state in states:
        if state < 0:
            kls.append(0.0)
            energy.append(0.0)
            continue
        counts = rng.multinomial(n, environment.policy[state])
        estimated = smoothed_policy(counts, alpha)
        metrics = state_diagnostic(estimated, optimal_log_policy[state])
        kls.append(metrics["kl"])
        energy.append(metrics["energy_squared"])
        zeros += int(np.sum(counts == 0))
        total_actions += counts.size
        if environment.controlled is not None:
            action_execution = np.where(environment.controlled.click_mask, 0.0, 1.0)
            # Reliability is the probability of branch zero for clicks.
            action_execution[environment.controlled.click_mask] = environment.controlled.probability[
                state, environment.controlled.click_mask, 0
            ]
            effective.append(float(estimated @ action_execution))
    result = {
        "mismatch": factor * float(np.mean(kls)),
        "energy_upper_bound": float(np.mean(energy)) / (4 * study.TAU * (1 - study.GAMMA)),
        "zero_count_fraction": zeros / total_actions if total_actions else 0.0,
        "absorbing_fraction": float(np.mean(states < 0)),
        "pilot_decisions": int(pilot.sum()),
        "action_samples": int(np.sum(states >= 0) * n),
        "nominal_action_samples": int(m * n),
        "effective_execution_rate": float(np.mean(effective)) if effective else 1.0,
    }
    if uniform_bank is not None:
        selected = np.asarray(uniform_bank)[rng.integers(0, len(uniform_bank), size=m)]
        uniform_kls = []
        for state in selected:
            counts = rng.multinomial(n, environment.policy[state])
            estimated = smoothed_policy(counts, alpha)
            uniform_kls.append(state_diagnostic(estimated, optimal_log_policy[state])["kl"])
        result["uniform_mismatch_score"] = factor * float(np.mean(uniform_kls))
        result["uniform_action_samples"] = int(m * n)
    return result


def rollout_predictor(
    environment,
    kernel,
    reference,
    query_budget,
    rng,
    log_samples=DIRECT_LOG_SAMPLES,
    privileged_log_probability=False,
):
    """Direct rollout baseline constrained by an action-query budget.

    The ordinary regularized baseline estimates the chosen action probability
    using ``log_samples`` categorical queries.  The privileged variant reads the
    native full-distribution log probability and is reported separately.
    """

    returns, rewards, successes = [], [], []
    queries, attempted_episodes, discarded_partial_queries = 0, 0, 0
    while queries < query_budget:
        state, discount = environment.initial_state, 1.0
        total, raw_total, success = 0.0, 0.0, 0.0
        episode_queries = 0
        complete = True
        attempted_episodes += 1
        while state >= 0:
            cost = 1 if privileged_log_probability else 1 + log_samples
            if queries + episode_queries + cost > query_budget:
                complete = False
                break
            action = environment.action(state, rng)
            episode_queries += 1
            if privileged_log_probability:
                log_probability = math.log(environment.policy[state, action])
            else:
                counts = rng.multinomial(log_samples, environment.policy[state])
                episode_queries += log_samples
                log_probability = math.log(smoothed_policy(counts, MAIN_ALPHA)[action])
            branch = int(rng.choice(2, p=kernel.probability[state, action]))
            task_reward = float(kernel.task_reward[state, action, branch])
            utility_cost = float(kernel.utility_cost[state, action, branch])
            regularized = task_reward - utility_cost - study.TAU * (
                log_probability - math.log(reference[state, action])
            )
            total += discount * regularized
            raw_total += discount * task_reward
            success += task_reward
            state = int(kernel.next_state[state, action, branch])
            discount *= study.GAMMA
        if not complete:
            queries += episode_queries
            discarded_partial_queries += episode_queries
            break
        queries += episode_queries
        returns.append(total)
        rewards.append(raw_total)
        successes.append(success)
        if attempted_episodes > 100_000:
            raise RuntimeError("Direct-rollout budget loop did not terminate")
    if not returns:
        raise RuntimeError(
            "The matched direct-rollout budget did not complete one episode; "
            "reduce the preregistered chosen-action probability query count"
        )
    return {
        "regularized_return": float(np.mean(returns)),
        "discounted_task_reward": float(np.mean(rewards)),
        "success": float(np.mean(successes)),
        "episodes": len(returns),
        "queries": queries,
        "discarded_partial_queries": discarded_partial_queries,
        "query_budget": int(query_budget),
        "log_probability_access": "native" if privileged_log_probability else f"frequency_N={log_samples}",
    }


def _c_measurement(
    model,
    policy,
    reference,
    config,
    m,
    n,
    seed,
    include_rollout=True,
):
    kernel, optimum = _controlled_oracle(
        model, reference, config["reliability"], config["click_cost"]
    )
    environment = SamplingEnvironment(policy, model.initial_state, controlled=kernel)
    rng = np.random.default_rng(seed)
    probe = single_probe(
        environment,
        optimum.log_policy,
        m,
        n,
        MAIN_ALPHA,
        rng,
        uniform_bank=controlled_reachable_states(kernel),
    )
    result = {
        "ceiling": float(optimum.value[kernel.initial_state]),
        "probe": probe,
        "query_accounting": {
            "pilot_decisions": probe["pilot_decisions"],
            "occupancy_action_samples": probe["action_samples"],
            "uniform_action_samples": probe["uniform_action_samples"],
        },
    }
    if include_rollout:
        diagnostic_budget = probe["pilot_decisions"] + probe["action_samples"]
        result["direct_rollout_frequency"] = rollout_predictor(
            environment,
            kernel,
            reference,
            diagnostic_budget,
            np.random.default_rng(seed ^ 0xA51C),
            privileged_log_probability=False,
        )
        result["direct_rollout_native_logprob"] = rollout_predictor(
            environment,
            kernel,
            reference,
            diagnostic_budget,
            np.random.default_rng(seed ^ 0xB51C),
            privileged_log_probability=True,
        )
    return result


def fit_affine(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    design = np.stack([np.ones(np.sum(valid)), x[valid]], axis=1)
    intercept, slope = np.linalg.lstsq(design, y[valid], rcond=None)[0]
    prediction = intercept + slope * x[valid]
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "calibration_rmse": float(np.sqrt(np.mean((prediction - y[valid]) ** 2))),
        "rows": int(np.sum(valid)),
    }


def apply_affine(mapping, value):
    return mapping["intercept"] + mapping["slope"] * value


def calibrate_c(experiment=study.EXPERIMENT):
    experiment = Path(experiment).resolve()
    b = study.read(experiment / "calibration/B/selection.json")
    m, n = b["selected"]["m"], b["selected"]["N"]
    output = experiment / "calibration/C"
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "registered_before_C_calibration_truth",
        "measurement_budget": {"m": m, "N": n, "alpha": MAIN_ALPHA},
        "direct_rollout_chosen_action_samples": DIRECT_LOG_SAMPLES,
        "configurations": [row["id"] for row in execution_configurations("calibration")],
        "baseline": {"reliability": .8, "click_cost": .01, "rendering": "R0"},
        "simple_mapping": "ordinary least squares affine map from the calibration score to exact delta J, separately by frozen checkpoint",
        "tie_rule": "family-macro RMSE of the KL/Bregman point predictor on calibration configurations",
        "test_exclusion": "held-out configurations and all test tasks unavailable to fitting",
        "registered_at_utc": datetime.now(UTC).isoformat(),
    }
    path = experiment / "calibration/C_protocol.json"
    if path.exists():
        old = study.read(path)
        protocol["registered_at_utc"] = old["registered_at_utc"]
        if old != protocol:
            raise ValueError("C calibration protocol changed")
    else:
        write_json(path, protocol)

    models = study.load_models(experiment, "calibration")
    rows = []
    started = time.monotonic()
    for model_id in ("h0_final", "h3_adapt_final"):
        for task_index, model in enumerate(models):
            base_reference, base_policy = _c_tables(experiment, "calibration", model, model_id, "R0")
            baseline_config = {
                "id": "baseline", "rendering": "R0", "reliability": .8, "click_cost": .01
            }
            baseline_measurement = _c_measurement(
                model,
                base_policy,
                base_reference,
                baseline_config,
                m,
                n,
                stable_seed(5102, "calibration", model_id, model.task.fingerprint, "baseline"),
            )
            baseline_kernel = controlled_kernel(model, .8, .01)
            baseline_exact = controlled_metrics(baseline_kernel, base_reference, base_policy)
            for config in execution_configurations("calibration"):
                reference, policy = _c_tables(
                    experiment, "calibration", model, model_id, config["rendering"]
                )
                measurement = _c_measurement(
                    model,
                    policy,
                    reference,
                    config,
                    m,
                    n,
                    stable_seed(5102, "calibration", model_id, model.task.fingerprint, config["id"]),
                )
                exact = controlled_metrics(
                    controlled_kernel(model, config["reliability"], config["click_cost"]),
                    reference,
                    policy,
                )
                delta_c = measurement["ceiling"] - baseline_measurement["ceiling"]
                row = {
                    "model_id": model_id,
                    "task_seed": model.task.seed,
                    "family": model.task.family,
                    "content_sha256": model.task.fingerprint,
                    "config_id": config["id"],
                    "rendering": config["rendering"],
                    "reliability": config["reliability"],
                    "click_cost": config["click_cost"],
                    "delta_ceiling": delta_c,
                    "predicted_delta_kl": delta_c - (
                        measurement["probe"]["mismatch"] - baseline_measurement["probe"]["mismatch"]
                    ),
                    "energy_lower": delta_c - measurement["probe"]["energy_upper_bound"],
                    "energy_upper": delta_c + baseline_measurement["probe"]["energy_upper_bound"],
                    "uniform_score": delta_c - (
                        measurement["probe"]["uniform_mismatch_score"]
                        - baseline_measurement["probe"]["uniform_mismatch_score"]
                    ),
                    "effective_score": (
                        measurement["probe"]["effective_execution_rate"]
                        - baseline_measurement["probe"]["effective_execution_rate"]
                    ),
                    "direct_rollout_frequency": (
                        measurement["direct_rollout_frequency"]["regularized_return"]
                        - baseline_measurement["direct_rollout_frequency"]["regularized_return"]
                    ),
                    "direct_rollout_native_logprob": (
                        measurement["direct_rollout_native_logprob"]["regularized_return"]
                        - baseline_measurement["direct_rollout_native_logprob"]["regularized_return"]
                    ),
                    "raw_reward_score": (
                        measurement["direct_rollout_frequency"]["discounted_task_reward"]
                        - baseline_measurement["direct_rollout_frequency"]["discounted_task_reward"]
                    ),
                    "success_score": (
                        measurement["direct_rollout_frequency"]["success"]
                        - baseline_measurement["direct_rollout_frequency"]["success"]
                    ),
                    "exact_delta_J": exact["regularized_return"] - baseline_exact["regularized_return"],
                    "exact_delta_M": exact["mismatch"] - baseline_exact["mismatch"],
                    "exact_delta_success": exact["success_probability"] - baseline_exact["success_probability"],
                    "probe_queries_config": measurement["probe"]["action_samples"] + measurement["probe"]["pilot_decisions"],
                    "probe_queries_baseline": baseline_measurement["probe"]["action_samples"] + baseline_measurement["probe"]["pilot_decisions"],
                }
                rows.append(row)
            write_json(output / "progress.json", {
                "status": "running", "model_id": model_id, "tasks": task_index + 1,
                "total_tasks": len(models), "rows": len(rows), "wall_seconds": time.monotonic() - started,
            })

    mappings = {}
    mapped_fields = {
        "uniform": "uniform_score",
        "effective_execution": "effective_score",
        "raw_reward_rollout": "raw_reward_score",
        "success_rollout": "success_score",
    }
    for model_id in ("h0_final", "h3_adapt_final"):
        selected = [row for row in rows if row["model_id"] == model_id]
        mappings[model_id] = {
            name: fit_affine(
                [row[field] for row in selected],
                [row["exact_delta_J"] for row in selected],
            ) for name, field in mapped_fields.items()
        }
    per_task_squared = defaultdict(list)
    for row in rows:
        per_task_squared[(row["model_id"], row["task_seed"], row["family"])].append(
            (row["predicted_delta_kl"] - row["exact_delta_J"]) ** 2
        )
    task_mse = [np.mean(values) for values in per_task_squared.values()]
    task_families = [key[2] for key in per_task_squared]
    tie_tolerance = math.sqrt(family_macro(task_mse, task_families))
    raw = output / "rows.jsonl"
    with raw.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    result = {
        "status": "complete",
        "measurement_budget": {"m": m, "N": n, "alpha": MAIN_ALPHA},
        "direct_rollout_chosen_action_samples": DIRECT_LOG_SAMPLES,
        "rows": len(rows),
        "mappings": mappings,
        "tie_tolerance": tie_tolerance,
        "tie_rule": protocol["tie_rule"],
        "KL_family_macro_RMSE": tie_tolerance,
        "rows_sha256": digest(raw),
        "protocol_sha256": digest(path),
        "wall_seconds": time.monotonic() - started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(output / "calibration.json", result)
    write_json(output / "progress.json", {"status": "complete", "rows": len(rows)})
    print(json.dumps({"status": "complete", "rows": len(rows), "tie_tolerance": tie_tolerance}), flush=True)
    return result


def prediction_interval(rows, value_key, seed):
    families = {family: [row for row in rows if row["family"] == family] for family in study.FAMILIES}
    rng = np.random.default_rng(seed)
    samples = []
    for family in study.FAMILIES:
        values = np.asarray([row[value_key] for row in families[family]], dtype=np.float64)
        draws = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
        samples.append(values[draws].mean(1))
    distribution = np.mean(samples, axis=0)
    low, high = np.quantile(distribution, [.025, .975])
    return {
        "mean": float(np.mean([np.mean([row[value_key] for row in families[family]]) for family in study.FAMILIES])),
        "lower": float(low),
        "upper": float(high),
    }


def predict_c_test(experiment=study.EXPERIMENT):
    experiment = Path(experiment).resolve()
    study.verify_seal(experiment)
    audit_path = experiment / "acceptance/test_probability_tables.json"
    if audit_path.exists():
        audit = study.read(audit_path)
        if audit.get("status") != "passed" or audit.get("split") != "test":
            raise RuntimeError("The saved test probability-table audit is invalid")
    else:
        audit = study.audit_probability_tables(experiment, "test")
        write_json(audit_path, audit)
    calibration = study.read(experiment / "calibration/C/calibration.json")
    m, n = calibration["measurement_budget"]["m"], calibration["measurement_budget"]["N"]
    output = experiment / "predictions/C"
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "task_predictions.jsonl"
    if raw.exists():
        raw.unlink()
    models = study.load_models(experiment, "test")
    rows = []
    started = time.monotonic()
    for model_id in ("h0_final", "h3_adapt_final"):
        mapping = calibration["mappings"][model_id]
        for task_index, model in enumerate(models):
            base_reference, base_policy = _c_tables(experiment, "test", model, model_id, "R0")
            baseline_config = {
                "id": "baseline", "rendering": "R0", "reliability": .8, "click_cost": .01
            }
            baseline = _c_measurement(
                model, base_policy, base_reference, baseline_config, m, n,
                stable_seed(5102, "test", model_id, model.task.fingerprint, "baseline"),
            )
            for config in execution_configurations("test"):
                reference, policy = _c_tables(
                    experiment, "test", model, model_id, config["rendering"]
                )
                measurement = _c_measurement(
                    model, policy, reference, config, m, n,
                    stable_seed(5102, "test", model_id, model.task.fingerprint, config["id"]),
                )
                delta_c = measurement["ceiling"] - baseline["ceiling"]
                uniform_score = delta_c - (
                    measurement["probe"]["uniform_mismatch_score"]
                    - baseline["probe"]["uniform_mismatch_score"]
                )
                effective_score = (
                    measurement["probe"]["effective_execution_rate"]
                    - baseline["probe"]["effective_execution_rate"]
                )
                raw_reward_score = (
                    measurement["direct_rollout_frequency"]["discounted_task_reward"]
                    - baseline["direct_rollout_frequency"]["discounted_task_reward"]
                )
                success_score = (
                    measurement["direct_rollout_frequency"]["success"]
                    - baseline["direct_rollout_frequency"]["success"]
                )
                row = {
                    "model_id": model_id,
                    "task_seed": model.task.seed,
                    "family": model.task.family,
                    "content_sha256": model.task.fingerprint,
                    "config_id": config["id"],
                    "rendering": config["rendering"],
                    "reliability": config["reliability"],
                    "click_cost": config["click_cost"],
                    "delta_ceiling": delta_c,
                    "predictions": {
                        "kl_bregman": delta_c - (
                            measurement["probe"]["mismatch"] - baseline["probe"]["mismatch"]
                        ),
                        "ceiling_only": delta_c,
                        "uniform_behavior": apply_affine(mapping["uniform"], uniform_score),
                        "effective_execution": apply_affine(mapping["effective_execution"], effective_score),
                        "direct_rollout_frequency": (
                            measurement["direct_rollout_frequency"]["regularized_return"]
                            - baseline["direct_rollout_frequency"]["regularized_return"]
                        ),
                        "direct_rollout_native_logprob": (
                            measurement["direct_rollout_native_logprob"]["regularized_return"]
                            - baseline["direct_rollout_native_logprob"]["regularized_return"]
                        ),
                        "raw_reward_rollout": apply_affine(mapping["raw_reward_rollout"], raw_reward_score),
                        "success_rollout": apply_affine(mapping["success_rollout"], success_score),
                    },
                    "energy_interval": {
                        "lower": delta_c - measurement["probe"]["energy_upper_bound"],
                        "upper": delta_c + baseline["probe"]["energy_upper_bound"],
                    },
                    "measurement": {
                        "config": measurement["probe"],
                        "baseline": baseline["probe"],
                        "config_direct_frequency": measurement["direct_rollout_frequency"],
                        "baseline_direct_frequency": baseline["direct_rollout_frequency"],
                    },
                    "online_query_budget": {
                        "kl_config": measurement["probe"]["action_samples"] + measurement["probe"]["pilot_decisions"],
                        "kl_baseline": baseline["probe"]["action_samples"] + baseline["probe"]["pilot_decisions"],
                        "uniform_config": measurement["probe"]["uniform_action_samples"],
                        "uniform_baseline": baseline["probe"]["uniform_action_samples"],
                        "direct_frequency_config": measurement["direct_rollout_frequency"]["queries"],
                        "direct_frequency_baseline": baseline["direct_rollout_frequency"]["queries"],
                        "reference_and_oracle": "reported separately; not actor queries",
                    },
                }
                rows.append(row)
                with raw.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
            write_json(output / "progress.json", {
                "status": "running", "model_id": model_id, "tasks": task_index + 1,
                "total_tasks": len(models), "rows": len(rows), "wall_seconds": time.monotonic() - started,
            })

    methods = tuple(rows[0]["predictions"])
    summary = {}
    for model_id in ("h0_final", "h3_adapt_final"):
        summary[model_id] = {}
        for config in execution_configurations("test"):
            selected = [row for row in rows if row["model_id"] == model_id and row["config_id"] == config["id"]]
            summary[model_id][config["id"]] = {
                "configuration": config,
                "methods": {
                    method: prediction_interval(
                        [row | {"value": row["predictions"][method]} for row in selected],
                        "value",
                        stable_seed(5205, "C", model_id, config["id"], method),
                    ) for method in methods
                },
                "energy_interval": {
                    "lower_mean": family_macro(
                        [row["energy_interval"]["lower"] for row in selected],
                        [row["family"] for row in selected],
                    ),
                    "upper_mean": family_macro(
                        [row["energy_interval"]["upper"] for row in selected],
                        [row["family"] for row in selected],
                    ),
                },
            }
    manifest = {
        "status": "predicted_unscored",
        "split": "test",
        "models": ["h0_final", "h3_adapt_final"],
        "configurations_per_model": 36,
        "tasks": len(models),
        "task_rows": len(rows),
        "measurement_budget": {"m": m, "N": n, "alpha": MAIN_ALPHA},
        "direct_rollout_chosen_action_samples": DIRECT_LOG_SAMPLES,
        "methods": list(methods),
        "tie_tolerance": calibration["tie_tolerance"],
        "task_predictions_sha256": digest(raw),
        "config_sha256": digest(experiment / "config.json"),
        "source_note": "finite samples simulated from sealed complete LM probability caches; not claimed as equal-count online API calls",
        "truth_fields_accessed": [],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": time.monotonic() - started,
        "summary": summary,
    }
    write_json(output / "predictions.json", manifest)
    write_json(output / "progress.json", {"status": "complete", "rows": len(rows)})
    print(json.dumps({"status": manifest["status"], "rows": len(rows), "sha256": digest(raw)}), flush=True)
    return manifest


def _controlled_rollouts(kernel, policy, reference, episodes, seed):
    rng = np.random.default_rng(seed)
    returns, successes, rewards, costs, penalties, steps = [], [], [], [], [], []
    for _ in range(episodes):
        state, discount = kernel.initial_state, 1.0
        total = task_total = cost_total = kl_total = success = 0.0
        count = 0
        while state >= 0:
            action = int(rng.choice(policy.shape[1], p=policy[state]))
            branch = int(rng.choice(2, p=kernel.probability[state, action]))
            task_reward = float(kernel.task_reward[state, action, branch])
            utility_cost = float(kernel.utility_cost[state, action, branch])
            kl = study.TAU * math.log(policy[state, action] / reference[state, action])
            total += discount * (task_reward - utility_cost - kl)
            task_total += discount * task_reward
            cost_total += discount * utility_cost
            kl_total += discount * kl
            success += task_reward
            count += 1
            state = int(kernel.next_state[state, action, branch])
            discount *= study.GAMMA
        returns.append(total)
        successes.append(success)
        rewards.append(task_total)
        costs.append(cost_total)
        penalties.append(kl_total)
        steps.append(count)
    return {
        "episodes": episodes,
        "mean_regularized_return": float(np.mean(returns)),
        "mean_success": float(np.mean(successes)),
        "mean_discounted_task_reward": float(np.mean(rewards)),
        "mean_discounted_cost": float(np.mean(costs)),
        "mean_discounted_kl_penalty": float(np.mean(penalties)),
        "mean_steps": float(np.mean(steps)),
        "return_standard_error": float(np.std(returns, ddof=1) / math.sqrt(episodes)),
        "success_standard_error": float(np.std(successes, ddof=1) / math.sqrt(episodes)),
    }


def seal_test_predictions(experiment=study.EXPERIMENT, require=("C", "D")):
    experiment = Path(experiment).resolve()
    study.verify_seal(experiment)
    records = {}
    for name in require:
        path = experiment / f"predictions/{name}/predictions.json"
        raw = experiment / f"predictions/{name}/task_predictions.jsonl"
        if not path.exists() or not raw.exists():
            raise RuntimeError(f"Missing unscored prediction output: {name}")
        record = study.read(path)
        if record.get("status") != "predicted_unscored" or record["task_predictions_sha256"] != digest(raw):
            raise RuntimeError(f"Prediction output is not a complete unscored file: {name}")
        records[name] = {"manifest_sha256": digest(path), "task_predictions_sha256": digest(raw)}
    seal = {
        "status": "sealed_before_scoring",
        "config_sha256": digest(experiment / "config.json"),
        "predictions": records,
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "scoring_outputs_existing_at_seal": {
            name: (experiment / f"scores/{name}/scores.json").exists() for name in require
        },
    }
    if any(seal["scoring_outputs_existing_at_seal"].values()):
        raise RuntimeError("A score output exists before prediction sealing")
    path = experiment / "predictions/seal.json"
    if path.exists() and study.read(path) != seal:
        raise ValueError("Prediction seal changed")
    write_json(path, seal)
    print(json.dumps(seal), flush=True)
    return seal


def verify_prediction_seal(experiment, name):
    seal = study.read(Path(experiment) / "predictions/seal.json")
    if seal.get("status") != "sealed_before_scoring" or name not in seal["predictions"]:
        raise RuntimeError(f"{name} predictions were not sealed before scoring")
    manifest = Path(experiment) / f"predictions/{name}/predictions.json"
    raw = Path(experiment) / f"predictions/{name}/task_predictions.jsonl"
    expected = seal["predictions"][name]
    if digest(manifest) != expected["manifest_sha256"] or digest(raw) != expected["task_predictions_sha256"]:
        raise ValueError(f"Sealed {name} predictions changed")
    return seal


def score_c_test(experiment=study.EXPERIMENT):
    experiment = Path(experiment).resolve()
    study.verify_seal(experiment)
    verify_prediction_seal(experiment, "C")
    output = experiment / "scores/C"
    output.mkdir(parents=True, exist_ok=True)
    models = study.load_models(experiment, "test")
    rows = []
    raw = output / "task_scores.jsonl"
    if raw.exists():
        raw.unlink()
    started = time.monotonic()
    for model_id in ("h0_final", "h3_adapt_final"):
        for task_index, model in enumerate(models):
            base_reference, base_policy = _c_tables(experiment, "test", model, model_id, "R0")
            baseline_kernel = controlled_kernel(model, .8, .01)
            baseline_exact = controlled_metrics(baseline_kernel, base_reference, base_policy)
            baseline_rollout = _controlled_rollouts(
                baseline_kernel,
                base_policy,
                base_reference,
                32,
                stable_seed(5103, "test", model_id, model.task.fingerprint, "baseline"),
            )
            for config in execution_configurations("test"):
                reference, policy = _c_tables(
                    experiment, "test", model, model_id, config["rendering"]
                )
                kernel = controlled_kernel(model, config["reliability"], config["click_cost"])
                exact = controlled_metrics(kernel, reference, policy)
                rollout = _controlled_rollouts(
                    kernel,
                    policy,
                    reference,
                    32,
                    stable_seed(5103, "test", model_id, model.task.fingerprint, config["id"]),
                )
                row = {
                    "model_id": model_id,
                    "task_seed": model.task.seed,
                    "family": model.task.family,
                    "content_sha256": model.task.fingerprint,
                    "config_id": config["id"],
                    "rendering": config["rendering"],
                    "reliability": config["reliability"],
                    "click_cost": config["click_cost"],
                    "exact": {key: value for key, value in exact.items()},
                    "baseline_exact": {key: value for key, value in baseline_exact.items()},
                    "exact_delta_J": exact["regularized_return"] - baseline_exact["regularized_return"],
                    "exact_delta_C": exact["ceiling"] - baseline_exact["ceiling"],
                    "exact_delta_M": exact["mismatch"] - baseline_exact["mismatch"],
                    "exact_delta_success": exact["success_probability"] - baseline_exact["success_probability"],
                    "rollout": rollout,
                    "baseline_rollout": baseline_rollout,
                    "rollout_delta_J": rollout["mean_regularized_return"] - baseline_rollout["mean_regularized_return"],
                    "rollout_delta_success": rollout["mean_success"] - baseline_rollout["mean_success"],
                }
                # Occupancy arrays are not JSON evidence rows; exact scalar
                # metrics already contain the independent M calculation.
                row["exact"].pop("occupancy", None)
                row["baseline_exact"].pop("occupancy", None)
                rows.append(row)
                with raw.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
            write_json(output / "progress.json", {
                "status": "running", "model_id": model_id, "tasks": task_index + 1,
                "total_tasks": len(models), "rows": len(rows), "wall_seconds": time.monotonic() - started,
            })
    maximum_gap = max(row["exact"]["gap_identity_error"] for row in rows)
    maximum_decomposition = max(row["exact"]["return_decomposition_error"] for row in rows)
    result = {
        "status": "complete",
        "tasks": len(models),
        "task_rows": len(rows),
        "rollouts_per_task_configuration": 32,
        "task_scores_sha256": digest(raw),
        "maximum_gap_identity_error": maximum_gap,
        "maximum_return_decomposition_error": maximum_decomposition,
        "prediction_seal_sha256": digest(experiment / "predictions/seal.json"),
        "wall_seconds": time.monotonic() - started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(output / "scores.json", result)
    write_json(output / "progress.json", {"status": "complete", "rows": len(rows)})
    print(json.dumps(result), flush=True)
    return result


def deterministic_as_controlled(next_state, rewards, initial_state):
    next_state = np.asarray(next_state, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float64)
    shape = (*next_state.shape, 2)
    branches = np.full(shape, -1, dtype=np.int64)
    branches[:, :, 0] = next_state
    probability = np.zeros(shape, dtype=np.float64)
    probability[:, :, 0] = 1.0
    task_reward = np.zeros(shape, dtype=np.float64)
    task_reward[:, :, 0] = rewards
    return ControlledKernel(
        next_state=branches,
        probability=probability,
        task_reward=task_reward,
        utility_cost=np.zeros(shape, dtype=np.float64),
        initial_state=int(initial_state),
        click_mask=np.zeros(next_state.shape[1], dtype=bool),
    )


def _d_tables(experiment, split, model, update, harness):
    base = study.load_table(experiment, split, f"D_base_{harness}", model)
    actor = study.load_table(experiment, split, f"D_h3_update_{update:04d}_{harness}", model)
    if not np.array_equal(base["next_state"], actor["next_state"]):
        raise AssertionError("D reference and actor use different deployment graphs")
    if not np.array_equal(base["task_rewards"], actor["task_rewards"]):
        raise AssertionError("D reference and actor use different reward tables")
    return {
        "next_state": base["next_state"],
        "rewards": base["task_rewards"],
        "initial_state": int(base["initial_state"]),
        "reference": np.exp(study.full_log_policy(base)),
        "policy": np.exp(study.full_log_policy(actor)),
        "bank": base["bank"],
        "actions": tuple(str(value) for value in base["semantic_actions"]),
    }


def _d_measurement(inputs, m, n, seed, include_rollout=True):
    optimum = solve(
        inputs["next_state"],
        inputs["rewards"],
        inputs["reference"],
        study.GAMMA,
        study.TAU,
    )
    environment = SamplingEnvironment(
        inputs["policy"], inputs["initial_state"], next_state=inputs["next_state"]
    )
    probe = single_probe(
        environment,
        optimum.log_policy,
        m,
        n,
        MAIN_ALPHA,
        np.random.default_rng(seed),
    )
    result = {"ceiling": float(optimum.value[inputs["initial_state"]]), "probe": probe}
    if include_rollout:
        kernel = deterministic_as_controlled(
            inputs["next_state"], inputs["rewards"], inputs["initial_state"]
        )
        budget = probe["pilot_decisions"] + probe["action_samples"]
        result["direct_rollout_frequency"] = rollout_predictor(
            environment,
            kernel,
            inputs["reference"],
            budget,
            np.random.default_rng(seed ^ 0xD51F),
            privileged_log_probability=False,
        )
        result["direct_rollout_native_logprob"] = rollout_predictor(
            environment,
            kernel,
            inputs["reference"],
            budget,
            np.random.default_rng(seed ^ 0xD51E),
            privileged_log_probability=True,
        )
    return result


def _d_exact(inputs):
    result = deterministic_exact(
        inputs["next_state"],
        inputs["rewards"],
        inputs["reference"],
        inputs["policy"],
        inputs["initial_state"],
    )
    for key in ("occupancy", "optimal_policy", "optimal_log_policy", "energy_residual"):
        result.pop(key)
    return result


def calibrate_d(experiment=study.EXPERIMENT):
    experiment = Path(experiment).resolve()
    b = study.read(experiment / "calibration/B/selection.json")
    c = study.read(experiment / "calibration/C/calibration.json")
    m, n = b["selected"]["m"], b["selected"]["N"]
    output = experiment / "calibration/D"
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "registered_before_D_calibration_truth",
        "measurement_budget": {"m": m, "N": n, "alpha": MAIN_ALPHA},
        "direct_rollout_chosen_action_samples": DIRECT_LOG_SAMPLES,
        "nodes": list(study.D_NODES),
        "deployments": list(study.D_HARNESSES),
        "recovery": "J(update,H3)-J(node0,H0)",
        "ranking": "J(update,H3)-J(update,H0)",
        "initial_H0": "independently finite-measured and charged to every comparison; known-exact auxiliary excluded from primary",
        "tie_tolerance": c["tie_tolerance"],
        "registered_at_utc": datetime.now(UTC).isoformat(),
    }
    path = experiment / "calibration/D_protocol.json"
    if path.exists():
        old = study.read(path)
        protocol["registered_at_utc"] = old["registered_at_utc"]
        if old != protocol:
            raise ValueError("D calibration protocol changed")
    else:
        write_json(path, protocol)

    rows, started = [], time.monotonic()
    models = study.load_models(experiment, "calibration")
    for task_index, model in enumerate(models):
        baseline_inputs = _d_tables(experiment, "calibration", model, 0, "H0")
        baseline_measurement = _d_measurement(
            baseline_inputs,
            m,
            n,
            stable_seed(5104, "calibration", model.task.fingerprint, 0, "H0"),
        )
        baseline_exact = _d_exact(baseline_inputs)
        for update in study.D_NODES:
            conditions = {}
            for harness in study.D_HARNESSES:
                inputs = _d_tables(experiment, "calibration", model, update, harness)
                measurement = _d_measurement(
                    inputs,
                    m,
                    n,
                    stable_seed(5104, "calibration", model.task.fingerprint, update, harness),
                )
                conditions[harness] = {
                    "measurement": measurement,
                    "exact": _d_exact(inputs),
                }
            predicted_recovery = (
                conditions["H3"]["measurement"]["ceiling"]
                - conditions["H3"]["measurement"]["probe"]["mismatch"]
                - baseline_measurement["ceiling"]
                + baseline_measurement["probe"]["mismatch"]
            )
            predicted_ranking = (
                conditions["H3"]["measurement"]["ceiling"]
                - conditions["H0"]["measurement"]["ceiling"]
                - conditions["H3"]["measurement"]["probe"]["mismatch"]
                + conditions["H0"]["measurement"]["probe"]["mismatch"]
            )
            rows.append({
                "task_seed": model.task.seed,
                "family": model.task.family,
                "content_sha256": model.task.fingerprint,
                "update": update,
                "predicted_recovery": predicted_recovery,
                "predicted_ranking": predicted_ranking,
                "exact_recovery": conditions["H3"]["exact"]["regularized_return"] - baseline_exact["regularized_return"],
                "exact_ranking": conditions["H3"]["exact"]["regularized_return"] - conditions["H0"]["exact"]["regularized_return"],
                "energy_recovery_interval": [
                    conditions["H3"]["measurement"]["ceiling"] - baseline_measurement["ceiling"]
                    - conditions["H3"]["measurement"]["probe"]["energy_upper_bound"],
                    conditions["H3"]["measurement"]["ceiling"] - baseline_measurement["ceiling"]
                    + baseline_measurement["probe"]["energy_upper_bound"],
                ],
                "energy_ranking_interval": [
                    conditions["H3"]["measurement"]["ceiling"] - conditions["H0"]["measurement"]["ceiling"]
                    - conditions["H3"]["measurement"]["probe"]["energy_upper_bound"],
                    conditions["H3"]["measurement"]["ceiling"] - conditions["H0"]["measurement"]["ceiling"]
                    + conditions["H0"]["measurement"]["probe"]["energy_upper_bound"],
                ],
            })
        write_json(output / "progress.json", {
            "status": "running", "tasks": task_index + 1, "total": len(models),
            "rows": len(rows), "wall_seconds": time.monotonic() - started,
        })
    raw = output / "rows.jsonl"
    with raw.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    families = [row["family"] for row in rows]
    result = {
        "status": "complete",
        "measurement_budget": {"m": m, "N": n, "alpha": MAIN_ALPHA},
        "rows": len(rows),
        "recovery_RMSE": math.sqrt(family_macro(
            [(row["predicted_recovery"] - row["exact_recovery"]) ** 2 for row in rows], families
        )),
        "ranking_RMSE": math.sqrt(family_macro(
            [(row["predicted_ranking"] - row["exact_ranking"]) ** 2 for row in rows], families
        )),
        "rows_sha256": digest(raw),
        "protocol_sha256": digest(path),
        "wall_seconds": time.monotonic() - started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(output / "calibration.json", result)
    write_json(output / "progress.json", {"status": "complete", "rows": len(rows)})
    print(json.dumps(result), flush=True)
    return result


def freeze_calibration(experiment=study.EXPERIMENT):
    experiment = Path(experiment).resolve()
    sources = {
        "A_readout_audit": study.audit_readout_source(),
        "B": {
            "path": "calibration/B/selection.json",
            "sha256": digest(experiment / "calibration/B/selection.json"),
            "result": study.read(experiment / "calibration/B/selection.json"),
        },
        "C": {
            "path": "calibration/C/calibration.json",
            "sha256": digest(experiment / "calibration/C/calibration.json"),
            "result": study.read(experiment / "calibration/C/calibration.json"),
        },
        "D": {
            "path": "calibration/D/calibration.json",
            "sha256": digest(experiment / "calibration/D/calibration.json"),
            "result": study.read(experiment / "calibration/D/calibration.json"),
        },
    }
    record = {
        "status": "frozen",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "test_tasks_or_reserved_configurations_used": False,
        "choices": {
            "measurement_budget": sources["B"]["result"]["selected"],
            "smoothing_alpha": MAIN_ALPHA,
            "direct_rollout_chosen_action_samples": DIRECT_LOG_SAMPLES,
            "simple_baseline_mappings": sources["C"]["result"]["mappings"],
            "tie_tolerance": sources["C"]["result"]["tie_tolerance"],
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "sources": sources,
    }
    path = experiment / "calibration/frozen_choices.json"
    if path.exists():
        old = study.read(path)
        record["frozen_at_utc"] = old["frozen_at_utc"]
        if old != record:
            raise ValueError("Frozen calibration choices changed")
        return old
    write_json(path, record)
    print(json.dumps({"status": "frozen", "choices": record["choices"]}), flush=True)
    return record


def predict_d_test(experiment=study.EXPERIMENT):
    experiment = Path(experiment).resolve()
    study.verify_seal(experiment)
    audit_path = experiment / "acceptance/test_probability_tables.json"
    if not audit_path.exists() or study.read(audit_path).get("status") != "passed":
        raise RuntimeError("Test probability tables must pass the full source audit before prediction")
    choices = study.read(experiment / "calibration/frozen_choices.json")["choices"]
    m, n = choices["measurement_budget"]["m"], choices["measurement_budget"]["N"]
    output = experiment / "predictions/D"
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "task_predictions.jsonl"
    if raw.exists():
        raw.unlink()
    rows, started = [], time.monotonic()
    models = study.load_models(experiment, "test")
    for task_index, model in enumerate(models):
        baseline_inputs = _d_tables(experiment, "test", model, 0, "H0")
        baseline = _d_measurement(
            baseline_inputs,
            m,
            n,
            stable_seed(5104, "test", model.task.fingerprint, 0, "H0"),
        )
        for update in study.D_NODES:
            conditions = {}
            for harness in study.D_HARNESSES:
                inputs = _d_tables(experiment, "test", model, update, harness)
                conditions[harness] = _d_measurement(
                    inputs,
                    m,
                    n,
                    stable_seed(5104, "test", model.task.fingerprint, update, harness),
                )
            recovery_delta_c = conditions["H3"]["ceiling"] - baseline["ceiling"]
            ranking_delta_c = conditions["H3"]["ceiling"] - conditions["H0"]["ceiling"]
            row = {
                "task_seed": model.task.seed,
                "family": model.task.family,
                "content_sha256": model.task.fingerprint,
                "update": update,
                "predicted_recovery": recovery_delta_c - (
                    conditions["H3"]["probe"]["mismatch"] - baseline["probe"]["mismatch"]
                ),
                "predicted_ranking": ranking_delta_c - (
                    conditions["H3"]["probe"]["mismatch"] - conditions["H0"]["probe"]["mismatch"]
                ),
                "direct_rollout_frequency_recovery": (
                    conditions["H3"]["direct_rollout_frequency"]["regularized_return"]
                    - baseline["direct_rollout_frequency"]["regularized_return"]
                ),
                "direct_rollout_frequency_ranking": (
                    conditions["H3"]["direct_rollout_frequency"]["regularized_return"]
                    - conditions["H0"]["direct_rollout_frequency"]["regularized_return"]
                ),
                "direct_rollout_native_recovery": (
                    conditions["H3"]["direct_rollout_native_logprob"]["regularized_return"]
                    - baseline["direct_rollout_native_logprob"]["regularized_return"]
                ),
                "direct_rollout_native_ranking": (
                    conditions["H3"]["direct_rollout_native_logprob"]["regularized_return"]
                    - conditions["H0"]["direct_rollout_native_logprob"]["regularized_return"]
                ),
                "energy_recovery_interval": {
                    "lower": recovery_delta_c - conditions["H3"]["probe"]["energy_upper_bound"],
                    "upper": recovery_delta_c + baseline["probe"]["energy_upper_bound"],
                },
                "energy_ranking_interval": {
                    "lower": ranking_delta_c - conditions["H3"]["probe"]["energy_upper_bound"],
                    "upper": ranking_delta_c + conditions["H0"]["probe"]["energy_upper_bound"],
                },
                "measurements": {"baseline_H0": baseline, **conditions},
                "query_budget": {
                    "baseline_H0": baseline["probe"]["action_samples"] + baseline["probe"]["pilot_decisions"],
                    "H0": conditions["H0"]["probe"]["action_samples"] + conditions["H0"]["probe"]["pilot_decisions"],
                    "H3": conditions["H3"]["probe"]["action_samples"] + conditions["H3"]["probe"]["pilot_decisions"],
                },
            }
            rows.append(row)
            with raw.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        write_json(output / "progress.json", {
            "status": "running", "tasks": task_index + 1, "total_tasks": len(models),
            "rows": len(rows), "wall_seconds": time.monotonic() - started,
        })
    summary = {}
    for update in study.D_NODES:
        selected = [row for row in rows if row["update"] == update]
        summary[str(update)] = {
            key: prediction_interval(selected, key, stable_seed(5205, "D", update, key))
            for key in (
                "predicted_recovery", "predicted_ranking",
                "direct_rollout_frequency_recovery", "direct_rollout_frequency_ranking",
                "direct_rollout_native_recovery", "direct_rollout_native_ranking",
            )
        }
    manifest = {
        "status": "predicted_unscored",
        "split": "test",
        "nodes": list(study.D_NODES),
        "deployments": list(study.D_HARNESSES),
        "tasks": len(models),
        "task_rows": len(rows),
        "measurement_budget": {"m": m, "N": n, "alpha": MAIN_ALPHA},
        "direct_rollout_chosen_action_samples": DIRECT_LOG_SAMPLES,
        "tie_tolerance": choices["tie_tolerance"],
        "task_predictions_sha256": digest(raw),
        "config_sha256": digest(experiment / "config.json"),
        "truth_fields_accessed": [],
        "summary": summary,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": time.monotonic() - started,
    }
    write_json(output / "predictions.json", manifest)
    write_json(output / "progress.json", {"status": "complete", "rows": len(rows)})
    print(json.dumps({"status": manifest["status"], "rows": len(rows), "sha256": digest(raw)}), flush=True)
    return manifest


def _deterministic_rollouts(inputs, episodes, seed):
    kernel = deterministic_as_controlled(
        inputs["next_state"], inputs["rewards"], inputs["initial_state"]
    )
    return _controlled_rollouts(
        kernel, inputs["policy"], inputs["reference"], episodes, seed
    )


def score_d_test(experiment=study.EXPERIMENT):
    experiment = Path(experiment).resolve()
    study.verify_seal(experiment)
    verify_prediction_seal(experiment, "D")
    output = experiment / "scores/D"
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "task_scores.jsonl"
    if raw.exists():
        raw.unlink()
    rows, started = [], time.monotonic()
    models = study.load_models(experiment, "test")
    for task_index, model in enumerate(models):
        baseline_inputs = _d_tables(experiment, "test", model, 0, "H0")
        baseline_exact = _d_exact(baseline_inputs)
        baseline_rollout = _deterministic_rollouts(
            baseline_inputs,
            32,
            stable_seed(5105, "test", model.task.fingerprint, 0, "H0"),
        )
        for update in study.D_NODES:
            conditions = {}
            for harness in study.D_HARNESSES:
                inputs = _d_tables(experiment, "test", model, update, harness)
                conditions[harness] = {
                    "exact": _d_exact(inputs),
                    "rollout": _deterministic_rollouts(
                        inputs,
                        32,
                        stable_seed(5105, "test", model.task.fingerprint, update, harness),
                    ),
                }
            row = {
                "task_seed": model.task.seed,
                "family": model.task.family,
                "content_sha256": model.task.fingerprint,
                "update": update,
                "exact_recovery": conditions["H3"]["exact"]["regularized_return"] - baseline_exact["regularized_return"],
                "exact_ranking": conditions["H3"]["exact"]["regularized_return"] - conditions["H0"]["exact"]["regularized_return"],
                "exact_delta_ceiling": conditions["H3"]["exact"]["ceiling"] - conditions["H0"]["exact"]["ceiling"],
                "exact_recovery_threshold": conditions["H3"]["exact"]["ceiling"] - baseline_exact["regularized_return"],
                "exact_H3_mismatch": conditions["H3"]["exact"]["mismatch"],
                "exact_H0_mismatch": conditions["H0"]["exact"]["mismatch"],
                "rollout_recovery": conditions["H3"]["rollout"]["mean_regularized_return"] - baseline_rollout["mean_regularized_return"],
                "rollout_ranking": conditions["H3"]["rollout"]["mean_regularized_return"] - conditions["H0"]["rollout"]["mean_regularized_return"],
                "H0": {"exact": conditions["H0"]["exact"], "rollout": conditions["H0"]["rollout"]},
                "H3": {"exact": conditions["H3"]["exact"], "rollout": conditions["H3"]["rollout"]},
                "baseline_H0": {"exact": baseline_exact, "rollout": baseline_rollout},
            }
            for key in ("H0", "H3", "baseline_H0"):
                row[key]["exact"].pop("occupancy", None)
            rows.append(row)
            with raw.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        write_json(output / "progress.json", {
            "status": "running", "tasks": task_index + 1, "total_tasks": len(models),
            "rows": len(rows), "wall_seconds": time.monotonic() - started,
        })
    maximum_gap = max(
        max(row[harness]["exact"]["gap_identity_error"] for harness in ("H0", "H3"))
        for row in rows
    )
    result = {
        "status": "complete",
        "tasks": len(models),
        "task_rows": len(rows),
        "rollouts_per_task_condition": 32,
        "task_scores_sha256": digest(raw),
        "maximum_gap_identity_error": maximum_gap,
        "prediction_seal_sha256": digest(experiment / "predictions/seal.json"),
        "wall_seconds": time.monotonic() - started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(output / "scores.json", result)
    write_json(output / "progress.json", {"status": "complete", "rows": len(rows)})
    print(json.dumps(result), flush=True)
    return result


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def macro_interval(rows, getter, seed):
    rng = np.random.default_rng(seed)
    by_family = {
        family: np.asarray([getter(row) for row in rows if row["family"] == family], dtype=np.float64)
        for family in study.FAMILIES
    }
    draws = []
    for family, values in by_family.items():
        if len(values) == 0:
            raise ValueError(f"No rows for family {family}")
        indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
        draws.append(values[indices].mean(1))
    distribution = np.mean(draws, axis=0)
    low, high = np.quantile(distribution, [.025, .975])
    return {
        "mean": float(np.mean([values.mean() for values in by_family.values()])),
        "lower": float(low),
        "upper": float(high),
        "per_family": {family: float(values.mean()) for family, values in by_family.items()},
    }


def classification(value, tolerance):
    return 1 if value > tolerance else (-1 if value < -tolerance else 0)


def _c_result_summary(experiment):
    predictions = read_jsonl(experiment / "predictions/C/task_predictions.jsonl")
    scores = read_jsonl(experiment / "scores/C/task_scores.jsonl")
    prediction_index = {
        (row["model_id"], row["task_seed"], row["config_id"]): row for row in predictions
    }
    score_index = {
        (row["model_id"], row["task_seed"], row["config_id"]): row for row in scores
    }
    if set(prediction_index) != set(score_index):
        raise AssertionError("C predictions and independent scores do not cover the same frozen cells")
    methods = tuple(predictions[0]["predictions"])
    tie = study.read(experiment / "calibration/frozen_choices.json")["choices"]["tie_tolerance"]
    configurations, joined = {}, []
    for key, prediction in prediction_index.items():
        score = score_index[key]
        joined.append({
            "model_id": key[0], "task_seed": key[1], "config_id": key[2],
            "family": prediction["family"], "predictions": prediction["predictions"],
            "energy_interval": prediction["energy_interval"],
            "truth": score["exact_delta_J"],
            "success": score["exact_delta_success"],
            "rollout_truth": score["rollout_delta_J"],
        })
    for model_id in ("h0_final", "h3_adapt_final"):
        configurations[model_id] = {}
        for config in execution_configurations("test"):
            rows = [row for row in joined if row["model_id"] == model_id and row["config_id"] == config["id"]]
            true_interval = macro_interval(
                rows, lambda row: row["truth"], stable_seed(5205, "C-score", model_id, config["id"])
            )
            rollout_interval = macro_interval(
                rows, lambda row: row["rollout_truth"], stable_seed(5205, "C-rollout", model_id, config["id"])
            )
            success_interval = macro_interval(
                rows, lambda row: row["success"], stable_seed(5205, "C-success", model_id, config["id"])
            )
            method_intervals = {
                method: macro_interval(
                    rows,
                    lambda row, method=method: row["predictions"][method],
                    stable_seed(5205, "C-predict", model_id, config["id"], method),
                ) for method in methods
            }
            configurations[model_id][config["id"]] = {
                "configuration": config,
                "observed_delta_J": true_interval,
                "observed_delta_success": success_interval,
                "independent_32_rollout_delta_J": rollout_interval,
                "predictions": method_intervals,
                "energy_sufficient_interval": {
                    "lower": family_macro(
                        [row["energy_interval"]["lower"] for row in rows],
                        [row["family"] for row in rows],
                    ),
                    "upper": family_macro(
                        [row["energy_interval"]["upper"] for row in rows],
                        [row["family"] for row in rows],
                    ),
                },
            }

    method_metrics = {}
    for method in methods:
        errors = [row["predictions"][method] - row["truth"] for row in joined]
        families = [row["family"] for row in joined]
        config_points = [
            (entry["predictions"][method]["mean"], entry["observed_delta_J"]["mean"], entry["predictions"][method])
            for model in configurations.values() for entry in model.values()
        ]
        config_errors = np.asarray([predicted - observed for predicted, observed, _ in config_points])
        all_correct = [
            classification(predicted, tie) == classification(observed, tie)
            for predicted, observed, _ in config_points
        ]
        non_tie = [
            np.sign(predicted) == np.sign(observed)
            for predicted, observed, _ in config_points if abs(observed) > tie
        ]
        coverage = [
            interval["lower"] <= observed <= interval["upper"]
            for _, observed, interval in config_points
        ]
        pair_correct, pair_total = 0, 0
        for model_id, model in configurations.items():
            ids = sorted(model)
            for left in range(len(ids)):
                for right in range(left + 1, len(ids)):
                    a, b = model[ids[left]], model[ids[right]]
                    observed = a["observed_delta_J"]["mean"] - b["observed_delta_J"]["mean"]
                    if abs(observed) <= tie:
                        continue
                    predicted = a["predictions"][method]["mean"] - b["predictions"][method]["mean"]
                    pair_correct += int(np.sign(observed) == np.sign(predicted))
                    pair_total += 1
        method_metrics[method] = {
            "task_family_macro_MAE": family_macro(np.abs(errors), families),
            "task_family_macro_RMSE": math.sqrt(family_macro(np.square(errors), families)),
            "configuration_MAE": float(np.mean(np.abs(config_errors))),
            "configuration_RMSE": float(np.sqrt(np.mean(config_errors**2))),
            "three_way_sign_accuracy": float(np.mean(all_correct)),
            "non_tie_sign_accuracy": float(np.mean(non_tie)) if non_tie else None,
            "pairwise_ranking_accuracy_non_tie": pair_correct / pair_total if pair_total else None,
            "pairwise_non_tie_comparisons": pair_total,
            "prediction_interval_coverage": float(np.mean(coverage)),
        }

    energy_points = [entry for model in configurations.values() for entry in model.values()]
    energy_decisions = []
    for entry in energy_points:
        interval = entry["energy_sufficient_interval"]
        decision = 1 if interval["lower"] > 0 else (-1 if interval["upper"] < 0 else 0)
        truth = classification(entry["observed_delta_J"]["mean"], tie)
        energy_decisions.append((decision, truth))
    conclusive = [(prediction, truth) for prediction, truth in energy_decisions if prediction]
    energy_summary = {
        "conclusive_fraction": len(conclusive) / len(energy_decisions),
        "conclusive_accuracy": float(np.mean([prediction == truth for prediction, truth in conclusive])) if conclusive else None,
        "rule": "positive if DeltaC-U_config>0; negative if DeltaC+U_baseline<0; otherwise inconclusive",
    }

    reversals = {}
    for model_id, model in configurations.items():
        reversals[model_id] = {}
        for rendering in study.C_RENDERINGS:
            for sweep in ("reliability", "cost"):
                points = [entry for entry in model.values()
                          if entry["configuration"]["rendering"] == rendering
                          and entry["configuration"]["sweep"] == sweep]
                parameter = "reliability" if sweep == "reliability" else "click_cost"
                points.sort(key=lambda entry: entry["configuration"][parameter])
                signs = [classification(entry["observed_delta_J"]["mean"], tie) for entry in points]
                crossings = [
                    [points[i - 1]["configuration"][parameter], points[i]["configuration"][parameter]]
                    for i in range(1, len(points)) if signs[i] and signs[i - 1] and signs[i] != signs[i - 1]
                ]
                reversals[model_id][f"{rendering}_{sweep}"] = {
                    "parameter": parameter,
                    "observed_signs": signs,
                    "measured_brackets": crossings,
                    "unique_crossing": len(crossings) == 1,
                    "interpolation_used": False,
                }
    rollout_errors = [row["rollout_truth"] - row["truth"] for row in joined]
    return {
        "tie_tolerance": tie,
        "configurations": configurations,
        "method_metrics": method_metrics,
        "energy_sufficient_bound": energy_summary,
        "reversal_observations": reversals,
        "independent_32_rollout": {
            "task_family_macro_MAE_vs_exact": family_macro(np.abs(rollout_errors), [row["family"] for row in joined]),
            "task_family_macro_RMSE_vs_exact": math.sqrt(family_macro(np.square(rollout_errors), [row["family"] for row in joined])),
        },
        "rows": len(joined),
        "identity": study.read(experiment / "scores/C/scores.json"),
    }


def _d_result_summary(experiment):
    predictions = read_jsonl(experiment / "predictions/D/task_predictions.jsonl")
    scores = read_jsonl(experiment / "scores/D/task_scores.jsonl")
    prediction_index = {(row["task_seed"], row["update"]): row for row in predictions}
    score_index = {(row["task_seed"], row["update"]): row for row in scores}
    if set(prediction_index) != set(score_index):
        raise AssertionError("D predictions and independent scores have different cells")
    tie = study.read(experiment / "calibration/frozen_choices.json")["choices"]["tie_tolerance"]
    nodes, joined = {}, []
    for key, prediction in prediction_index.items():
        score = score_index[key]
        joined.append({
            "task_seed": key[0], "update": key[1], "family": prediction["family"],
            "predicted_recovery": prediction["predicted_recovery"],
            "predicted_ranking": prediction["predicted_ranking"],
            "direct_frequency_recovery": prediction["direct_rollout_frequency_recovery"],
            "direct_frequency_ranking": prediction["direct_rollout_frequency_ranking"],
            "direct_native_recovery": prediction["direct_rollout_native_recovery"],
            "direct_native_ranking": prediction["direct_rollout_native_ranking"],
            "energy_recovery_interval": prediction["energy_recovery_interval"],
            "energy_ranking_interval": prediction["energy_ranking_interval"],
            "exact_recovery": score["exact_recovery"],
            "exact_ranking": score["exact_ranking"],
            "rollout_recovery": score["rollout_recovery"],
            "rollout_ranking": score["rollout_ranking"],
            "exact_delta_ceiling": score["exact_delta_ceiling"],
            "threshold_margin": score["exact_recovery_threshold"] - score["exact_H3_mismatch"],
        })
    fields = (
        "predicted_recovery", "predicted_ranking", "direct_frequency_recovery",
        "direct_frequency_ranking", "direct_native_recovery", "direct_native_ranking",
        "exact_recovery", "exact_ranking", "rollout_recovery", "rollout_ranking",
        "exact_delta_ceiling", "threshold_margin",
    )
    for update in study.D_NODES:
        rows = [row for row in joined if row["update"] == update]
        nodes[str(update)] = {
            field: macro_interval(rows, lambda row, field=field: row[field], stable_seed(5205, "D-score", update, field))
            for field in fields
        }
        nodes[str(update)]["energy_recovery_interval"] = {
            "lower": family_macro([row["energy_recovery_interval"]["lower"] for row in rows], [row["family"] for row in rows]),
            "upper": family_macro([row["energy_recovery_interval"]["upper"] for row in rows], [row["family"] for row in rows]),
        }
        nodes[str(update)]["energy_ranking_interval"] = {
            "lower": family_macro([row["energy_ranking_interval"]["lower"] for row in rows], [row["family"] for row in rows]),
            "upper": family_macro([row["energy_ranking_interval"]["upper"] for row in rows], [row["family"] for row in rows]),
        }

    def decision_metrics(predicted, observed):
        pairs = []
        for node in nodes.values():
            prediction = classification(node[predicted]["mean"], tie)
            truth = classification(node[observed]["mean"], tie)
            pairs.append((prediction, truth))
        return {
            "three_way_accuracy": float(np.mean([a == b for a, b in pairs])),
            "false_positives": sum(a == 1 and b != 1 for a, b in pairs),
            "false_negatives": sum(a != 1 and b == 1 for a, b in pairs),
        }

    families = [row["family"] for row in joined]

    def diagnostic_metrics(recovery_field, ranking_field):
        recovery_errors = [row[recovery_field] - row["exact_recovery"] for row in joined]
        ranking_errors = [row[ranking_field] - row["exact_ranking"] for row in joined]
        return {
            "recovery_task_family_macro_MAE": family_macro(np.abs(recovery_errors), families),
            "recovery_task_family_macro_RMSE": math.sqrt(
                family_macro(np.square(recovery_errors), families)
            ),
            "ranking_task_family_macro_MAE": family_macro(np.abs(ranking_errors), families),
            "ranking_task_family_macro_RMSE": math.sqrt(
                family_macro(np.square(ranking_errors), families)
            ),
            "recovery_decisions": decision_metrics(recovery_field, "exact_recovery"),
            "ranking_decisions": decision_metrics(ranking_field, "exact_ranking"),
        }

    def first_node(field, criterion="mean"):
        for update in study.D_NODES:
            if nodes[str(update)][field][criterion] > 0:
                return update
        return None

    threshold_errors = [
        abs(node["exact_recovery"]["mean"] - node["threshold_margin"]["mean"])
        for node in nodes.values()
    ]
    return {
        "tie_tolerance": tie,
        "nodes": nodes,
        "method_metrics": {
            "kl_bregman": diagnostic_metrics("predicted_recovery", "predicted_ranking"),
            "direct_rollout_frequency": diagnostic_metrics(
                "direct_frequency_recovery", "direct_frequency_ranking"
            ),
            "direct_rollout_native_logprob": diagnostic_metrics(
                "direct_native_recovery", "direct_native_ranking"
            ),
            "independent_32_rollout": diagnostic_metrics(
                "rollout_recovery", "rollout_ranking"
            ),
        },
        "crossings": {
            "first_observed_recovery_mean_positive": first_node("exact_recovery", "mean"),
            "first_observed_recovery_CI_positive": first_node("exact_recovery", "lower"),
            "first_predicted_recovery_mean_positive": first_node("predicted_recovery", "mean"),
            "first_observed_ranking_mean_positive": first_node("exact_ranking", "mean"),
            "first_observed_ranking_CI_positive": first_node("exact_ranking", "lower"),
            "first_predicted_ranking_mean_positive": first_node("predicted_ranking", "mean"),
            "interpolation_used": False,
        },
        "general_recovery_identity_max_aggregate_error": max(threshold_errors),
        "higher_ceiling_special_case_supported": all(
            nodes[str(update)]["exact_delta_ceiling"]["lower"] > 0 for update in study.D_NODES
        ),
        "rows": len(joined),
        "identity": study.read(experiment / "scores/D/scores.json"),
    }


def _readout_summary():
    data = study.read(study.READOUT / "analysis/readout_data.json")
    test = data["splits"]["test"]
    result = {
        "source": str(study.READOUT / "analysis/readout_data.json"),
        "source_sha256": digest(study.READOUT / "analysis/readout_data.json"),
        "repeat_controls": {
            model: record["maxima"] for model, record in data["repeat_controls"].items()
        },
        "models": {},
        "interpretation": "same-input repeats measure deterministic numerical stability; B measures finite categorical sampling noise",
    }
    for model_id, record in test["models"].items():
        result["models"][model_id] = {}
        for rendering in ("R0", "R1", "R3"):
            main = record["main"][rendering]
            result["models"][model_id][rendering] = {
                key: main[key] for key in (
                    "target_mse", "readout_mse", "semantic_tv", "argmax_change_rate",
                    "delta_regularized_return_vs_R0", "delta_success_probability_vs_R0",
                )
            }
        result["models"][model_id]["target_positive_control"] = record["positive_control"]
        result["models"][model_id]["validation"] = record["validation"]
        result["models"][model_id]["natural_reference"] = {
            rendering: record["natural_reference"][rendering] for rendering in ("R0", "R1", "R3")
        }
    return result


def _resource_summary(experiment):
    probability = {}
    total_gpu_seconds = 0.0
    for split in ("calibration", "test"):
        probability[split] = {}
        for cell_id, spec in study.cell_specs(split).items():
            accounting = study.read(study.cell_directory(experiment, split, cell_id) / "accounting.json")
            group = probability[split].setdefault(spec["study"], {
                "cells": 0, "tasks": 0, "states": 0, "forward_state_rows": 0,
                "prompt_tokens": 0, "forward_microbatches": 0, "gpu_wall_seconds": 0.0,
            })
            group["cells"] += 1
            for key in ("tasks", "states", "forward_state_rows", "prompt_tokens", "forward_microbatches"):
                group[key] += accounting[key]
            group["gpu_wall_seconds"] += accounting["wall_seconds"]
            total_gpu_seconds += accounting["wall_seconds"]
    c_rows = read_jsonl(experiment / "predictions/C/task_predictions.jsonl")
    d_rows = read_jsonl(experiment / "predictions/D/task_predictions.jsonl")
    c_config_queries = sum(row["online_query_budget"]["kl_config"] for row in c_rows)
    c_uniform_queries = sum(row["online_query_budget"]["uniform_config"] for row in c_rows)
    c_direct_queries = sum(row["online_query_budget"]["direct_frequency_config"] for row in c_rows)
    c_baseline, c_uniform_baseline, c_direct_baseline = {}, {}, {}
    for row in c_rows:
        key = (row["model_id"], row["task_seed"])
        c_baseline.setdefault(key, row["online_query_budget"]["kl_baseline"])
        c_uniform_baseline.setdefault(key, row["online_query_budget"]["uniform_baseline"])
        c_direct_baseline.setdefault(key, row["online_query_budget"]["direct_frequency_baseline"])
    d_baseline, d_condition = {}, {}
    for row in d_rows:
        d_baseline.setdefault(row["task_seed"], row["query_budget"]["baseline_H0"])
        for harness in ("H0", "H3"):
            d_condition.setdefault((row["task_seed"], row["update"], harness), row["query_budget"][harness])
    return {
        "probability_measurement": probability,
        "probability_measurement_total_gpu_hours": total_gpu_seconds / 3600,
        "equivalent_online_actor_queries": {
            "C_KL_configurations": c_config_queries,
            "C_KL_shared_baselines": sum(c_baseline.values()),
            "C_KL_total": c_config_queries + sum(c_baseline.values()),
            "C_uniform_total": c_uniform_queries + sum(c_uniform_baseline.values()),
            "C_direct_frequency_total": c_direct_queries + sum(c_direct_baseline.values()),
            "D_conditions": sum(d_condition.values()),
            "D_shared_initial_H0": sum(d_baseline.values()),
            "D_total": sum(d_condition.values()) + sum(d_baseline.values()),
            "definition": "categorical action samples plus pilot actor decisions; reference/oracle construction excluded and reported separately",
        },
        "cached_CPU_wall_seconds": {
            "C_prediction": study.read(experiment / "predictions/C/predictions.json")["wall_seconds"],
            "D_prediction": study.read(experiment / "predictions/D/predictions.json")["wall_seconds"],
            "C_score": study.read(experiment / "scores/C/scores.json")["wall_seconds"],
            "D_score": study.read(experiment / "scores/D/scores.json")["wall_seconds"],
        },
        "reference_and_oracle": "full frozen-base reference forwards are included in GPU cells; exact DP is CPU and is not counted as actor queries",
    }


def _format_interval(record, digits=4):
    return f'{record["mean"]:.{digits}f} [{record["lower"]:.{digits}f}, {record["upper"]:.{digits}f}]'


def write_report(experiment, data):
    experiment = Path(experiment)
    study.DOCS.mkdir(parents=True, exist_ok=True)
    b = data["B_sampling"]
    c = data["C_prediction"]
    d = data["D_adaptation"]
    lines = [
        "# 第五章 5.1：有限读数、回报预测与适应诊断",
        "",
        "状态：**complete**。A–D 固定清单均已完成；模型训练与 H3 适应都只有 seed 42，本实验没有新增训练。",
        "",
        "## 设计与证据范围",
        "",
        "使用三个 MiniWoB++ 任务族，新增 calibration 72 与 test 576 个 IID 任务。每个任务已通过原生浏览器成功路径重放，并按 seed 和公开内容排除所有保留数据。模型为 Qwen3.5-2B 基础模型、E1 H0 update 1024 与 Figure 3 H3-adapt update 1024；checkpoint 条件不能解释为训练重复。",
        "",
        "B/C/D 的有限动作读数由已保存的真实 LM 完整菜单概率进行有限分类抽样模拟。它严格使用同一 prompt、decoder 和完整菜单，但不声称实际发出了等量黑盒 API 请求。test 预测文件先保存并哈希，随后才运行独立精确评分和 32 次新随机流 rollout。oracle 只提供 ceiling 与被查询状态的最优目标。",
        "",
        "## A. 目标等价与 readout drift",
        "",
        "A 直接复用并核验已完成的 readout-equivalence 实验。共同 semantic reference 下 R0/R1/R3 的最优目标等价误差在 FP64 容差内为零；相同输入五次前向的 log probability、TV 与 centered readout 最大差均为零。这是数值稳定性基线，不是有限动作采样噪声。选择后改名保持策略和回报不变；只改目标的 oracle 阳性控制得到 g≠0、eta=0。",
        "",
        "| 冻结模型 | 呈现 | readout MSE | ΔJ vs R0 | Δsuccess vs R0 |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {"base": "base", "h0_final": "H0 final", "h3_adapt_final": "H3 adapt final"}
    for model_id, model in data["A_readout"]["models"].items():
        for rendering in ("R1", "R3"):
            row = model[rendering]
            lines.append(
                f'| {labels[model_id]} | {rendering} | {row["readout_mse"]["mean"]:.6f} | '
                f'{_format_interval(row["delta_regularized_return_vs_R0"], 4)} | '
                f'{_format_interval(row["delta_success_probability_vs_R0"], 4)} |'
            )
    selected = b["selected"]
    lines += [
        "",
        "共同目标并没有保证冻结模型的实际偏好或部署回报相同。自然 reference 附加分析单独保留，因为它会同时改变 g 与 eta。",
        "",
        "## B. 有限采样误差",
        "",
        f'主预算只用 calibration 选择为 **m={selected["m"]}, N={selected["N"]}, alpha={selected["alpha"]}**。三组等动作查询预算候选的 family-macro M RMSE 为：'
        + ", ".join(f"{key}={value:.5f}" for key, value in b["candidate_family_macro_rmse"].items()) + "。",
        "",
        "| 呈现 | m,N | M bias | M RMSE | readout MSE | zero-count | energy bound / true M |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    point = f'm{selected["m"]}_N{selected["N"]}'
    for rendering in study.B_RENDERINGS:
        row = b["all_grid"][rendering][point]
        lines.append(
            f'| {rendering} | {selected["m"]},{selected["N"]} | {row["family_macro_bias"]:+.5f} | '
            f'{row["family_macro_rmse"]:.5f} | {row["mean_readout_mse"]:.6f} | '
            f'{row["mean_zero_count_fraction"]:.3f} | {row["energy_bound_to_true_M_mean_ratio"]:.2f}× |'
        )
    lines += [
        "",
        "occupancy 采样保留吸收态的零贡献，没有把非终止状态重新归一化。均匀状态库的 naïve 分数与使用真实 density ratio 的 importance 版本均完整报告。star、chain、complete graph 复用同一频率表并恢复同一个 centered log-frequency；断连图不能识别组件间偏移。有限样本 certificate 只有在 b_N≤p0/2 时计为适用，未通过条件的点没有强行给出证书。",
        "",
        "## C. 未揭示 test 配置的回报与排序预测",
        "",
        f'近似并列阈值由 calibration KL 预测 RMSE 冻结为 {c["tie_tolerance"]:.6f}。下表以 72 个模型×配置聚合点评分；truth 是完整概率的独立 occupancy-KL/策略评估，32 rollout 是另一条新随机流。',
        "",
        "| 方法 | config MAE | config RMSE | 非并列正负准确率 | 非并列 pairwise 排序 | 预测区间覆盖 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, row in c["method_metrics"].items():
        lines.append(
            f'| {method} | {row["configuration_MAE"]:.5f} | {row["configuration_RMSE"]:.5f} | '
            f'{row["non_tie_sign_accuracy"]:.3f} | {row["pairwise_ranking_accuracy_non_tie"]:.3f} | '
            f'{row["prediction_interval_coverage"]:.3f} |'
        )
    lines += [
        "",
        (
            f'energy sufficient interval 在 {100*c["energy_sufficient_bound"]["conclusive_fraction"]:.1f}% 的配置上给出方向；'
            f'其有结论点准确率为 {c["energy_sufficient_bound"]["conclusive_accuracy"] if c["energy_sufficient_bound"]["conclusive_accuracy"] is not None else "N/A"}。它是充分界，不作为点预测。'
        ),
        "",
        f"完整菜单下 parser validity 为结构性 1，有效格式率没有区分力。frequency direct rollout 每个所选动作使用 {DIRECT_LOG_SAMPLES} 次分类抽样估计 log probability，并把这些查询计入预算；native-logprob rollout 单列为拥有额外访问权限的基线。raw reward 与 success rollout 经过 calibration 映射后才与 regularized J 比较。",
        "",
        "## D. H3 适应 checkpoint 的恢复与接口排序",
        "",
        "| update | 预测恢复 ΔJ | 实测恢复 ΔJ | 预测 H3−H0 | 实测 H3−H0 | ΔC(H3−H0) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for update in study.D_NODES:
        row = d["nodes"][str(update)]
        lines.append(
            f'| {update} | {_format_interval(row["predicted_recovery"])} | {_format_interval(row["exact_recovery"])} | '
            f'{_format_interval(row["predicted_ranking"])} | {_format_interval(row["exact_ranking"])} | '
            f'{_format_interval(row["exact_delta_ceiling"])} |'
        )
    lines += [
        "",
        "| D 诊断方法 | 恢复 RMSE | 恢复三分类准确率 | 排序 RMSE | 排序三分类准确率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, row in d["method_metrics"].items():
        lines.append(
            f'| {method} | {row["recovery_task_family_macro_RMSE"]:.5f} | '
            f'{row["recovery_decisions"]["three_way_accuracy"]:.3f} | '
            f'{row["ranking_task_family_macro_RMSE"]:.5f} | '
            f'{row["ranking_decisions"]["three_way_accuracy"]:.3f} |'
        )
    crossings = d["crossings"]
    lines += [
        "",
        f'首个实测恢复均值为正的保存节点：{crossings["first_observed_recovery_mean_positive"]}；95% 区间下界为正：{crossings["first_observed_recovery_CI_positive"]}；有限诊断预测均值首次为正：{crossings["first_predicted_recovery_mean_positive"]}。',
        f'首个实测接口排序 H3>H0 的保存节点：{crossings["first_observed_ranking_mean_positive"]}；95% 区间下界为正：{crossings["first_observed_ranking_CI_positive"]}；有限诊断预测均值首次为正：{crossings["first_predicted_ranking_mean_positive"]}。没有在稀疏 checkpoint 之间插值。',
        "",
        f'一般恢复条件的聚合数值残差最大为 {d["general_recovery_identity_max_aggregate_error"]:.3e}。本轮 higher-ceiling 特例是否成立：{d["higher_ceiling_special_case_supported"]}；若不成立，结论仅按一般 J=C−M 诊断表述。恢复与同 checkpoint 接口排序分别评分，未把二者混为同一事件。',
        "",
        "## 计算、可追溯性与限制",
        "",
        "所有概率单元保存完整语义动作表、prompt hash、模型/checkpoint hash、状态与 token 计数。GPU forward 与 CPU 缓存抽样的实际时间、等价在线 actor 查询预算、reference/oracle 构造分别记录。源训练和适应均只有 seed 42，区间只覆盖配对任务抽样与有限测量随机性，不含训练 seed 不确定性。",
        "",
        (
            f'完整概率缓存累计 {data["resources"]["probability_measurement_total_gpu_hours"]:.3f} GPU 小时；'
            f'C 主 KL 诊断的等价在线 actor 查询为 {data["resources"]["equivalent_online_actor_queries"]["C_KL_total"]:,}，'
            f'D 为 {data["resources"]["equivalent_online_actor_queries"]["D_total"]:,}。这些查询预算不把离线缓存节省解释为在线效率。'
        ),
        "",
        f'- 固定配置：`{experiment / "config.json"}`',
        f'- test 预测封存：`{experiment / "predictions/seal.json"}`',
        f'- 完整机器可读汇总：`{experiment / "analysis/diagnostics_data.json"}`',
        f'- B 原始数组：`{experiment / "calibration/B/sampling_arrays.npz"}`',
        f'- C 逐任务预测/评分：`{experiment / "predictions/C/task_predictions.jsonl"}`、`{experiment / "scores/C/task_scores.jsonl"}`',
        f'- D 逐任务预测/评分：`{experiment / "predictions/D/task_predictions.jsonl"}`、`{experiment / "scores/D/task_scores.jsonl"}`',

        "",
        "受控环境提供精确最优目标，因此这里验证的是测量与预测流程，不声称在未知真实环境中恢复了最优目标。A–D 的负结果、无反转点和不适用证书均按冻结清单保留。",
    ]
    (study.DOCS / "REPORT.md").write_text("\n".join(lines) + "\n")


def analyze_and_report(experiment=study.EXPERIMENT):
    experiment = Path(experiment).resolve()
    study.verify_seal(experiment)
    verify_prediction_seal(experiment, "C")
    verify_prediction_seal(experiment, "D")
    if study.read(experiment / "scores/C/scores.json")["status"] != "complete":
        raise RuntimeError("C score is incomplete")
    if study.read(experiment / "scores/D/scores.json")["status"] != "complete":
        raise RuntimeError("D score is incomplete")
    output = experiment / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    data = {
        "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment": str(experiment),
        "config_sha256": digest(experiment / "config.json"),
        "prediction_seal_sha256": digest(experiment / "predictions/seal.json"),
        "A_readout": _readout_summary(),
        "B_sampling": study.read(experiment / "calibration/B/selection.json"),
        "C_prediction": _c_result_summary(experiment),
        "D_adaptation": _d_result_summary(experiment),
        "resources": _resource_summary(experiment),
        "uncertainty": {
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "unit": "paired native task within family; equal family aggregation",
            "training_seed": 42,
            "training_seed_uncertainty": False,
        },
    }
    path = output / "diagnostics_data.json"
    write_json(path, data)
    write_report(experiment, data)
    completion = {
        "status": "complete",
        "data_sha256": digest(path),
        "report_sha256": digest(study.DOCS / "REPORT.md"),
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(output / "completion.json", completion)
    print(json.dumps(completion), flush=True)
    return data


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "calibrate-b", "calibrate-c", "calibrate-d", "freeze-calibration",
            "predict-c", "predict-d", "seal-predictions", "score-c", "score-d",
            "report",
        ),
    )
    parser.add_argument("--experiment", type=Path, default=study.EXPERIMENT)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    functions = {
        "calibrate-b": analyze_b_calibration,
        "calibrate-c": calibrate_c,
        "calibrate-d": calibrate_d,
        "freeze-calibration": freeze_calibration,
        "predict-c": predict_c_test,
        "predict-d": predict_d_test,
        "seal-predictions": seal_test_predictions,
        "score-c": score_c_test,
        "score-d": score_d_test,
    }
    if args.command == "report":
        analyze_and_report(args.experiment)
    else:
        functions[args.command](args.experiment)


if __name__ == "__main__":
    main()
