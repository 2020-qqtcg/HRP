"""Second finite-measurement study for Section 5.1.

Calibration is deliberately isolated from the previous held-out split.  The
module reuses only the sealed calibration probability tables until a single
budget and smoothing rule passes the preregistered D and C gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np

from hrp_miniwob.adapters.miniwob_section51 import (
    controlled_kernel,
    evaluate_controlled,
    solve_controlled,
)
from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_section51 as study
from hrp_miniwob.benchmarks import miniwob_section51_analysis as old
from hrp_miniwob.benchmarks.miniwob_meta_harness import digest
from hrp_miniwob.suite import reachable_states
from hrp_miniwob.v2 import FAMILIES
from hrp_miniwob.v2.browser import Browser
from hrp_miniwob.v2.model import compile_task
from hrp_miniwob.v2.suite import is_ood, verify_success_path

ROOT = study.ROOT
EXPERIMENT = ROOT / "environment/artifacts/v2/section51_measurement_v2_2026-09-16"
DOCS = ROOT / "environment/docs/v2/section51_measurement_v2_2026-09-16"
SOURCE = study.EXPERIMENT
BRIEF = Path(__file__).resolve()

MS = (64, 256)
NS = (16, 64, 256, 1024)
REPEATS = 32
RULES = (
    ("per_action_alpha_0.1", "per_action", .1),
    ("per_action_alpha_0.5", "per_action", .5),
    ("total_beta_0.5", "total", .5),
    ("total_beta_1.0", "total", 1.0),
)
RULE_ORDER = {name: index for index, (name, _, _) in enumerate(RULES)}
BASE_SEED = 6106
TEST_START_SEED = 61_100_000
TEST_CELLS = tuple(
    ["C_base_R0"]
    + [f"C_{model}_{rendering}" for model in ("h0_final", "h3_adapt_final")
       for rendering in study.C_RENDERINGS]
    + ["D_base_H0", "D_base_H3"]
    + [f"D_h3_update_{update:04d}_{harness}" for update in study.D_NODES
       for harness in study.D_HARNESSES]
)


def read(path):
    return json.loads(Path(path).read_text())


def stable_seed(*parts):
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def alpha_for(rule_id: str, actions: int) -> float:
    name, kind, value = next(rule for rule in RULES if rule[0] == rule_id)
    del name
    return value if kind == "per_action" else value / actions


def candidates():
    return [
        {"id": f"m{m}_N{n}_{rule_id}", "m": m, "N": n, "rule_id": rule_id}
        for m in MS for n in NS for rule_id, _, _ in RULES
    ]


def register_protocol(experiment=EXPERIMENT):
    experiment = Path(experiment).resolve()
    output = experiment / "calibration"
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": 1,
        "task_id": "section51_measurement_v2_2026-09-16",
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "study_source": str(BRIEF),
        "study_source_sha256": digest(BRIEF),
        "source_calibration": str(SOURCE / "probabilities/calibration"),
        "source_manifest_sha256": digest(SOURCE / "suite/manifest.json"),
        "source_seal_sha256": digest(SOURCE / "config.json"),
        "conditions": {
            "harnesses": list(study.D_HARNESSES),
            "adaptation_nodes": list(study.D_NODES),
            "tasks": 72,
            "tasks_per_family": 24,
        },
        "grid": {
            "m": list(MS),
            "N": list(NS),
            "rules": [
                {"id": name, "kind": kind, "value": value}
                for name, kind, value in RULES
            ],
            "candidates": candidates(),
            "measurement_pools": REPEATS,
            "nested_prefix": True,
        },
        "acceptance": {
            "condition_macro_absolute_expected_M_bias_max": .005,
            "family_condition_absolute_expected_M_bias_max": .01,
            "per_node_recovery_aggregate_RMSE_max": .01,
            "per_node_ranking_aggregate_RMSE_max": .01,
            "C_configuration_aggregate_RMSE_max": .03,
        },
        "selection": "lowest mean realized actor-query cost; <=1% cost ties use worst-node return-difference RMSE, then m, N, fixed rule order",
        "test_data": {"start_seed": 61_100_000, "tasks_per_family": 192},
        "test_forbidden_before_acceptance": True,
    }
    path = output / "protocol.json"
    if path.exists():
        previous = read(path)
        protocol["registered_at_utc"] = previous["registered_at_utc"]
        if previous != protocol:
            raise ValueError("Registered measurement-v2 protocol changed")
    else:
        write_json(path, protocol)
    write_json(
        experiment / "progress.json",
        {"status": "calibration_registered", "protocol_sha256": digest(path),
         "updated_at_utc": datetime.now(UTC).isoformat()},
    )
    return protocol


def _nested_counts(policy, states, rng):
    rows = np.flatnonzero(states >= 0)
    result = {n: np.zeros((len(states), policy.shape[1]), dtype=np.int64) for n in NS}
    if not len(rows):
        return result
    increments = (NS[0], *(right - left for left, right in pairwise(NS)))
    for state in np.unique(states[rows]):
        selected = rows[states[rows] == state]
        total = np.zeros((len(selected), policy.shape[1]), dtype=np.int64)
        for n, increment in zip(NS, increments):
            total += rng.multinomial(increment, policy[state], size=len(selected))
            result[n][selected] = total
    return result


def _geometric_states_with_costs(policy, initial_state, next_state, count, rng, controlled=None):
    times = rng.geometric(1 - study.GAMMA, size=count) - 1
    states = np.full(count, int(initial_state), dtype=np.int64)
    decisions = np.zeros(count, dtype=np.int64)
    for step in range(int(times.max(initial=0))):
        rows = np.flatnonzero((times > step) & (states >= 0))
        if not len(rows):
            break
        current = states[rows]
        cumulative = np.cumsum(policy[current], axis=1)
        action = np.sum(rng.random(len(rows))[:, None] > cumulative, axis=1)
        if controlled is None:
            states[rows] = next_state[current, action]
        else:
            probability = controlled.probability[current, action, 0]
            branch = (rng.random(len(rows)) >= probability).astype(np.int64)
            states[rows] = controlled.next_state[current, action, branch]
        decisions[rows] += 1
    return states, decisions


def _all_candidate_estimates(policy, optimal_log_policy, initial_state, next_state, rng, controlled=None):
    states, decisions = _geometric_states_with_costs(
        policy, initial_state, next_state, max(MS), rng, controlled=controlled
    )
    counts = _nested_counts(policy, states, rng)
    factor = study.TAU / (1 - study.GAMMA)
    values, costs, diagnostics = {}, {}, {}
    for m in MS:
        prefix = states[:m]
        live = prefix >= 0
        live_count = int(live.sum())
        for n in NS:
            table = counts[n][:m]
            for rule_id, _, _ in RULES:
                alpha = alpha_for(rule_id, policy.shape[1])
                estimated = (table[live] + alpha) / (n + alpha * policy.shape[1])
                kl = np.zeros(m, dtype=np.float64)
                if live_count:
                    kl[live] = np.sum(
                        estimated * (np.log(estimated) - optimal_log_policy[prefix[live]]), axis=1
                    )
                key = f"m{m}_N{n}_{rule_id}"
                values[key] = factor * float(kl.mean())
                costs[key] = int(decisions[:m].sum() + live_count * n)
                diagnostics[key] = {
                    "zero_count_fraction": float(np.mean(table[live] == 0)) if live_count else 0.0,
                    "absorbing_fraction": float(1 - live_count / m),
                    "smoothing_weight": float(alpha * policy.shape[1] / (n + alpha * policy.shape[1])),
                }
    return values, costs, diagnostics


def _log_choose(n):
    return np.asarray([
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        for k in range(n + 1)
    ])


def exact_expected_mismatch(policy, optimal_log_policy, occupancy, n, rule_id):
    """Exact expectation over categorical counts using Binomial marginals."""

    policy = np.asarray(policy, dtype=np.float64)
    optimal_log_policy = np.asarray(optimal_log_policy, dtype=np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64)
    actions = policy.shape[1]
    alpha = alpha_for(rule_id, actions)
    denominator = n + alpha * actions
    grid = np.arange(n + 1, dtype=np.float64)
    log_choose = _log_choose(n)
    q = (grid + alpha) / denominator
    q_log_q = q * np.log(q)
    flat_p = policy.reshape(-1)
    expected_q_log_q = np.empty_like(flat_p)
    chunk = max(128, 2_000_000 // (n + 1))
    for start in range(0, len(flat_p), chunk):
        p = flat_p[start:start + chunk]
        log_pmf = (
            log_choose[None, :]
            + grid[None, :] * np.log(p)[:, None]
            + (n - grid)[None, :] * np.log1p(-p)[:, None]
        )
        pmf = np.exp(log_pmf)
        pmf /= pmf.sum(axis=1, keepdims=True)
        expected_q_log_q[start:start + chunk] = pmf @ q_log_q
    expected_q_log_q = expected_q_log_q.reshape(policy.shape)
    mean_q = (n * policy + alpha) / denominator
    expected_kl = np.sum(expected_q_log_q - mean_q * optimal_log_policy, axis=1)
    return float(study.TAU / (1 - study.GAMMA) * occupancy @ expected_kl)


def _task_d_calibration(task_index):
    models = study.load_models(SOURCE, "calibration")
    model = models[task_index]
    records = {}
    for update in study.D_NODES:
        for harness in study.D_HARNESSES:
            inputs = old._d_tables(SOURCE, "calibration", model, update, harness)
            exact = old.deterministic_exact(
                inputs["next_state"], inputs["rewards"], inputs["reference"],
                inputs["policy"], inputs["initial_state"],
            )
            condition = f"u{update:04d}_{harness}"
            expected = {}
            for n in NS:
                for rule_id, _, _ in RULES:
                    expected[f"N{n}_{rule_id}"] = exact_expected_mismatch(
                        inputs["policy"], exact["optimal_log_policy"], exact["occupancy"], n, rule_id
                    )
            pools = {}
            for repeat in range(REPEATS):
                rng = np.random.default_rng(stable_seed(
                    BASE_SEED, "D", "calibration", model.task.fingerprint,
                    condition, repeat,
                ))
                pools[str(repeat)] = _all_candidate_estimates(
                    inputs["policy"], exact["optimal_log_policy"], inputs["initial_state"],
                    inputs["next_state"], rng,
                )
            records[condition] = {
                "update": update,
                "harness": harness,
                "true_M": exact["mismatch"],
                "ceiling": exact["ceiling"],
                "J": exact["regularized_return"],
                "actions": len(inputs["actions"]),
                "expected_M": expected,
                "pools": pools,
            }
    # The initial H0 comparison is measured independently and charged separately.
    inputs = old._d_tables(SOURCE, "calibration", model, 0, "H0")
    exact = old.deterministic_exact(
        inputs["next_state"], inputs["rewards"], inputs["reference"],
        inputs["policy"], inputs["initial_state"],
    )
    pools = {}
    for repeat in range(REPEATS):
        rng = np.random.default_rng(stable_seed(
            BASE_SEED, "D", "calibration", model.task.fingerprint, "initial_H0", repeat,
        ))
        pools[str(repeat)] = _all_candidate_estimates(
            inputs["policy"], exact["optimal_log_policy"], inputs["initial_state"],
            inputs["next_state"], rng,
        )
    records["initial_H0"] = {
        "update": 0, "harness": "H0", "true_M": exact["mismatch"],
        "ceiling": exact["ceiling"], "J": exact["regularized_return"],
        "actions": len(inputs["actions"]), "pools": pools,
    }
    return {
        "task_seed": model.task.seed,
        "family": model.task.family,
        "content_sha256": model.task.fingerprint,
        "conditions": records,
    }


def _family_macro(rows, value):
    return float(np.mean([
        np.mean([row[value] for row in rows if row["family"] == family])
        for family in study.FAMILIES
    ]))


def _summarize_d(rows):
    comparison = []
    for candidate in candidates():
        candidate_id = candidate["id"]
        expected_key = f"N{candidate['N']}_{candidate['rule_id']}"
        expected_groups = []
        family_groups = []
        for update in study.D_NODES:
            for harness in study.D_HARNESSES:
                condition = f"u{update:04d}_{harness}"
                selected = []
                for task in rows:
                    row = task["conditions"][condition]
                    selected.append({
                        "family": task["family"],
                        "bias": row["expected_M"][expected_key] - row["true_M"],
                    })
                macro_bias = _family_macro(selected, "bias")
                expected_groups.append({
                    "update": update, "harness": harness, "bias": macro_bias,
                    "absolute_bias": abs(macro_bias),
                })
                for family in study.FAMILIES:
                    bias = float(np.mean([
                        row["bias"] for row in selected if row["family"] == family
                    ]))
                    family_groups.append({
                        "update": update, "harness": harness, "family": family,
                        "bias": bias, "absolute_bias": abs(bias),
                    })
        node_metrics = []
        all_costs = []
        zero_rates = []
        smoothing_weights = []
        for repeat in range(REPEATS):
            for task in rows:
                for condition in task["conditions"].values():
                    if "pools" not in condition:
                        continue
                    _, costs, diagnostic = condition["pools"][str(repeat)]
                    all_costs.append(costs[candidate_id])
                    zero_rates.append(diagnostic[candidate_id]["zero_count_fraction"])
                    smoothing_weights.append(diagnostic[candidate_id]["smoothing_weight"])
        for update in study.D_NODES:
            recovery_errors, ranking_errors = [], []
            for repeat in range(REPEATS):
                recovery_rows, ranking_rows = [], []
                for task in rows:
                    baseline = task["conditions"]["initial_H0"]
                    h0 = task["conditions"][f"u{update:04d}_H0"]
                    h3 = task["conditions"][f"u{update:04d}_H3"]
                    b_hat = baseline["pools"][str(repeat)][0][candidate_id]
                    h0_hat = h0["pools"][str(repeat)][0][candidate_id]
                    h3_hat = h3["pools"][str(repeat)][0][candidate_id]
                    recovery_rows.append({
                        "family": task["family"],
                        "prediction": h3["ceiling"] - baseline["ceiling"] - h3_hat + b_hat,
                        "truth": h3["J"] - baseline["J"],
                    })
                    ranking_rows.append({
                        "family": task["family"],
                        "prediction": h3["ceiling"] - h0["ceiling"] - h3_hat + h0_hat,
                        "truth": h3["J"] - h0["J"],
                    })
                recovery_errors.append(
                    _family_macro(recovery_rows, "prediction") - _family_macro(recovery_rows, "truth")
                )
                ranking_errors.append(
                    _family_macro(ranking_rows, "prediction") - _family_macro(ranking_rows, "truth")
                )
            node_metrics.append({
                "update": update,
                "recovery_RMSE": float(np.sqrt(np.mean(np.square(recovery_errors)))),
                "ranking_RMSE": float(np.sqrt(np.mean(np.square(ranking_errors)))),
                "recovery_bias": float(np.mean(recovery_errors)),
                "ranking_bias": float(np.mean(ranking_errors)),
            })
        max_condition_bias = max(row["absolute_bias"] for row in expected_groups)
        max_family_bias = max(row["absolute_bias"] for row in family_groups)
        max_return_rmse = max(
            max(row["recovery_RMSE"], row["ranking_RMSE"]) for row in node_metrics
        )
        passed = max_condition_bias <= .005 and max_family_bias <= .01 and max_return_rmse <= .01
        comparison.append({
            **candidate,
            "passed_D": passed,
            "max_condition_macro_absolute_expected_M_bias": max_condition_bias,
            "max_family_condition_absolute_expected_M_bias": max_family_bias,
            "max_node_return_difference_RMSE": max_return_rmse,
            "mean_realized_actor_queries": float(np.mean(all_costs)),
            "mean_zero_count_fraction": float(np.mean(zero_rates)),
            "mean_smoothing_weight": float(np.mean(smoothing_weights)),
            "expected_bias_by_condition": expected_groups,
            "expected_bias_by_family_condition": family_groups,
            "node_metrics": node_metrics,
        })
    passing = [row for row in comparison if row["passed_D"]]
    selected = None
    if passing:
        minimum = min(row["mean_realized_actor_queries"] for row in passing)
        pool = [row for row in passing if row["mean_realized_actor_queries"] <= minimum * 1.01]
        selected = min(
            pool,
            key=lambda row: (
                row["max_node_return_difference_RMSE"], row["m"], row["N"],
                RULE_ORDER[row["rule_id"]],
            ),
        )
    return comparison, selected


def calibrate_d(experiment=EXPERIMENT, workers=4):
    experiment = Path(experiment).resolve()
    register_protocol(experiment)
    output = experiment / "calibration/D"
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "task_calibration.jsonl"
    completed = {}
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            row = json.loads(line)
            completed[row["task_seed"]] = row
    models = study.load_models(SOURCE, "calibration")
    pending = [index for index, model in enumerate(models) if model.task.seed not in completed]
    started = time.monotonic()
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for row in pool.map(_task_d_calibration, pending):
                completed[row["task_seed"]] = row
                with rows_path.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                write_json(output / "progress.json", {
                    "status": "running", "completed_tasks": len(completed), "total_tasks": len(models),
                    "wall_seconds": time.monotonic() - started,
                    "updated_at_utc": datetime.now(UTC).isoformat(),
                })
    rows = [completed[model.task.seed] for model in models]
    comparison, selected = _summarize_d(rows)
    result = {
        "status": "passed" if selected else "failed",
        "tasks": len(rows),
        "candidate_count": len(comparison),
        "selected": selected,
        "passing_candidate_ids": [row["id"] for row in comparison if row["passed_D"]],
        "protocol_sha256": digest(experiment / "calibration/protocol.json"),
        "rows_sha256": digest(rows_path),
        "comparison": comparison,
        "wall_seconds_this_invocation": time.monotonic() - started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(output / "acceptance.json", result)
    write_json(output / "progress.json", {"status": result["status"], "completed_tasks": len(rows)})
    write_json(experiment / "progress.json", {
        "status": "D_calibration_passed" if selected else "calibration_failed_D",
        "selected": selected["id"] if selected else None,
        "updated_at_utc": datetime.now(UTC).isoformat(),
    })
    print(json.dumps({
        "status": result["status"], "passing": result["passing_candidate_ids"],
        "selected": selected["id"] if selected else None,
    }))
    return result


def _controlled_exact(model, reference, policy, reliability, click_cost):
    kernel = controlled_kernel(model, reliability, click_cost)
    optimum = solve_controlled(kernel, reference, study.GAMMA, study.TAU)
    evaluated = evaluate_controlled(kernel, reference, policy, study.GAMMA, study.TAU)
    mismatch = float(
        study.TAU / (1 - study.GAMMA)
        * evaluated.occupancy
        @ np.sum(policy * (np.log(policy) - optimum.log_policy), axis=1)
    )
    ceiling = float(optimum.value[kernel.initial_state])
    if abs(ceiling - evaluated.return_value - mismatch) > 1e-10:
        raise AssertionError("Controlled exact return decomposition failed")
    return kernel, optimum, {
        "ceiling": ceiling,
        "J": float(evaluated.return_value),
        "M": mismatch,
    }


def _task_c_calibration(task_index, selected_id):
    models = study.load_models(SOURCE, "calibration")
    model = models[task_index]
    old_id = "m64_N16_per_action_alpha_0.5"
    rows = []
    for model_id in ("h0_final", "h3_adapt_final"):
        reference, baseline_policy = old._c_tables(
            SOURCE, "calibration", model, model_id, "R0"
        )
        baseline_kernel, baseline_optimum, baseline_exact = _controlled_exact(
            model, reference, baseline_policy, .8, .01
        )
        baseline_pools = []
        for repeat in range(REPEATS):
            estimates = _all_candidate_estimates(
                baseline_policy, baseline_optimum.log_policy,
                baseline_kernel.initial_state, None,
                np.random.default_rng(stable_seed(
                    BASE_SEED, "C", "calibration", model_id,
                    model.task.fingerprint, "baseline", repeat,
                )),
                controlled=baseline_kernel,
            )
            baseline_pools.append({
                "selected_M": estimates[0][selected_id],
                "old_M": estimates[0][old_id],
                "selected_queries": estimates[1][selected_id],
                "old_queries": estimates[1][old_id],
            })
        for config in old.execution_configurations("calibration"):
            reference_config, policy = old._c_tables(
                SOURCE, "calibration", model, model_id, config["rendering"]
            )
            if not np.array_equal(reference, reference_config):
                raise AssertionError("C common semantic reference changed across renderings")
            kernel, optimum, exact = _controlled_exact(
                model, reference, policy, config["reliability"], config["click_cost"]
            )
            pools = []
            for repeat in range(REPEATS):
                estimates = _all_candidate_estimates(
                    policy, optimum.log_policy, kernel.initial_state, None,
                    np.random.default_rng(stable_seed(
                        BASE_SEED, "C", "calibration", model_id,
                        model.task.fingerprint, config["id"], repeat,
                    )),
                    controlled=kernel,
                )
                pools.append({
                    "predicted_selected": (
                        exact["ceiling"] - baseline_exact["ceiling"]
                        - estimates[0][selected_id]
                        + baseline_pools[repeat]["selected_M"]
                    ),
                    "predicted_old": (
                        exact["ceiling"] - baseline_exact["ceiling"]
                        - estimates[0][old_id]
                        + baseline_pools[repeat]["old_M"]
                    ),
                    "selected_queries": (
                        estimates[1][selected_id] + baseline_pools[repeat]["selected_queries"]
                    ),
                    "old_queries": estimates[1][old_id] + baseline_pools[repeat]["old_queries"],
                })
            rows.append({
                "model_id": model_id,
                "config_id": config["id"],
                "rendering": config["rendering"],
                "reliability": config["reliability"],
                "click_cost": config["click_cost"],
                "exact_delta_J": exact["J"] - baseline_exact["J"],
                "exact_delta_M": exact["M"] - baseline_exact["M"],
                "delta_ceiling": exact["ceiling"] - baseline_exact["ceiling"],
                "pools": pools,
            })
    return {
        "task_seed": model.task.seed,
        "family": model.task.family,
        "content_sha256": model.task.fingerprint,
        "rows": rows,
    }


def _summarize_c(tasks, selected_id):
    aggregate = {"selected": [], "old": []}
    query_costs = {"selected": [], "old": []}
    for model_id in ("h0_final", "h3_adapt_final"):
        for config in old.execution_configurations("calibration"):
            selected_rows = []
            for task in tasks:
                row = next(
                    value for value in task["rows"]
                    if value["model_id"] == model_id and value["config_id"] == config["id"]
                )
                selected_rows.append((task["family"], row))
            truth = float(np.mean([
                np.mean([row["exact_delta_J"] for family, row in selected_rows if family == name])
                for name in study.FAMILIES
            ]))
            for repeat in range(REPEATS):
                for method, field in (
                    ("selected", "predicted_selected"), ("old", "predicted_old")
                ):
                    prediction = float(np.mean([
                        np.mean([
                            row["pools"][repeat][field]
                            for family, row in selected_rows if family == name
                        ]) for name in study.FAMILIES
                    ]))
                    aggregate[method].append({
                        "model_id": model_id, "config_id": config["id"], "repeat": repeat,
                        "prediction": prediction, "truth": truth, "error": prediction - truth,
                    })
            for family, row in selected_rows:
                del family
                query_costs["selected"].extend(pool["selected_queries"] for pool in row["pools"])
                query_costs["old"].extend(pool["old_queries"] for pool in row["pools"])
    summary = {}
    for method in ("selected", "old"):
        errors = np.asarray([row["error"] for row in aggregate[method]])
        summary[method] = {
            "configuration_aggregate_RMSE": float(np.sqrt(np.mean(errors**2))),
            "bias": float(np.mean(errors)),
            "maximum_absolute_error": float(np.max(np.abs(errors))),
            "mean_realized_actor_queries_per_comparison": float(np.mean(query_costs[method])),
            "points": len(errors),
        }
    summary["selected"]["candidate_id"] = selected_id
    summary["old"]["candidate_id"] = "m64_N16_per_action_alpha_0.5"
    return summary, aggregate


def calibrate_c(experiment=EXPERIMENT, workers=4):
    experiment = Path(experiment).resolve()
    d = read(experiment / "calibration/D/acceptance.json")
    if d["status"] != "passed" or not d["selected"]:
        raise RuntimeError("D acceptance must pass before C regression calibration")
    selected_id = d["selected"]["id"]
    output = experiment / "calibration/C"
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "task_calibration.jsonl"
    completed = {}
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            row = json.loads(line)
            completed[row["task_seed"]] = row
    models = study.load_models(SOURCE, "calibration")
    pending = [index for index, model in enumerate(models) if model.task.seed not in completed]
    started = time.monotonic()
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_task_c_calibration, index, selected_id) for index in pending]
            for future in futures:
                row = future.result()
                completed[row["task_seed"]] = row
                with rows_path.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                write_json(output / "progress.json", {
                    "status": "running", "completed_tasks": len(completed), "total_tasks": len(models),
                    "wall_seconds": time.monotonic() - started,
                    "updated_at_utc": datetime.now(UTC).isoformat(),
                })
    tasks = [completed[model.task.seed] for model in models]
    summary, aggregate = _summarize_c(tasks, selected_id)
    passed = summary["selected"]["configuration_aggregate_RMSE"] <= .03
    aggregate_path = output / "aggregate_measurement_pools.json"
    write_json(aggregate_path, aggregate)
    result = {
        "status": "passed" if passed else "failed",
        "tasks": len(tasks),
        "models": ["h0_final", "h3_adapt_final"],
        "configurations_per_model": len(old.execution_configurations("calibration")),
        "measurement_pools": REPEATS,
        "gate_RMSE": .03,
        "summary": summary,
        "rows_sha256": digest(rows_path),
        "aggregate_sha256": digest(aggregate_path),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "wall_seconds_this_invocation": time.monotonic() - started,
    }
    write_json(output / "acceptance.json", result)
    write_json(output / "progress.json", {"status": result["status"], "completed_tasks": len(tasks)})
    write_json(experiment / "progress.json", {
        "status": "C_calibration_passed" if passed else "calibration_failed_C",
        "selected": selected_id,
        "updated_at_utc": datetime.now(UTC).isoformat(),
    })
    print(json.dumps({"status": result["status"], "selected": summary["selected"]}))
    return result


def _frequency_rollout(inputs, query_budget, n_log, rule_id, rng):
    policy, reference = inputs["policy"], inputs["reference"]
    alpha = alpha_for(rule_id, policy.shape[1])
    cost = 1 + n_log
    episode_count = query_budget // cost + 1
    states = np.full(episode_count, inputs["initial_state"], dtype=np.int64)
    returns = np.zeros(episode_count, dtype=np.float64)
    steps = np.zeros(episode_count, dtype=np.int64)
    discount = np.ones(episode_count, dtype=np.float64)
    for _ in range(study.HORIZON):
        rows = np.flatnonzero(states >= 0)
        if not len(rows):
            break
        current = states[rows]
        cumulative = np.cumsum(policy[current], axis=1)
        action = np.sum(rng.random(len(rows))[:, None] > cumulative, axis=1)
        selected_probability = policy[current, action]
        selected_count = rng.binomial(n_log, selected_probability)
        estimated_probability = (selected_count + alpha) / (
            n_log + alpha * policy.shape[1]
        )
        returns[rows] += discount[rows] * (
            inputs["rewards"][current, action]
            - study.TAU * np.log(estimated_probability / reference[current, action])
        )
        states[rows] = inputs["next_state"][current, action]
        discount[rows] *= study.GAMMA
        steps[rows] += 1
    cumulative_queries = np.cumsum(steps * cost)
    completed = int(np.sum(cumulative_queries <= query_budget))
    if not completed:
        raise RuntimeError("Frequency rollout did not complete one episode")
    queries = int(cumulative_queries[completed - 1])
    return {
        "J_hat": float(np.mean(returns[:completed])),
        "episodes": completed,
        "queries": queries,
        "unused_queries": int(query_budget - queries),
    }


def _task_frequency_calibration(task_index, budgets, selected_rule):
    model = study.load_models(SOURCE, "calibration")[task_index]
    candidate_specs = [
        {"id": f"Nlog{n_log}_{selected_rule}", "N_log": n_log, "rule_id": selected_rule}
        for n_log in (4, 16, 64)
    ]
    conditions = {}
    for update in study.D_NODES:
        for harness in study.D_HARNESSES:
            key = f"u{update:04d}_{harness}"
            inputs = old._d_tables(SOURCE, "calibration", model, update, harness)
            exact = old._d_exact(inputs)
            pools = {}
            for repeat in range(REPEATS):
                pools[str(repeat)] = {}
                for spec in candidate_specs:
                    pools[str(repeat)][spec["id"]] = _frequency_rollout(
                        inputs, budgets[key][repeat], spec["N_log"], spec["rule_id"],
                        np.random.default_rng(stable_seed(
                            BASE_SEED, "frequency", "calibration", model.task.fingerprint,
                            key, repeat, spec["id"],
                        )),
                    )
            conditions[key] = {"J": exact["regularized_return"], "pools": pools}
    inputs = old._d_tables(SOURCE, "calibration", model, 0, "H0")
    exact = old._d_exact(inputs)
    pools = {}
    for repeat in range(REPEATS):
        pools[str(repeat)] = {}
        for spec in candidate_specs:
            pools[str(repeat)][spec["id"]] = _frequency_rollout(
                inputs, budgets["initial_H0"][repeat], spec["N_log"], spec["rule_id"],
                np.random.default_rng(stable_seed(
                    BASE_SEED, "frequency", "calibration", model.task.fingerprint,
                    "initial_H0", repeat, spec["id"],
                )),
            )
    conditions["initial_H0"] = {"J": exact["regularized_return"], "pools": pools}
    return {
        "task_seed": model.task.seed, "family": model.task.family,
        "content_sha256": model.task.fingerprint, "conditions": conditions,
    }


def calibrate_frequency_rollout(experiment=EXPERIMENT, workers=4):
    experiment = Path(experiment).resolve()
    d = read(experiment / "calibration/D/acceptance.json")
    c = read(experiment / "calibration/C/acceptance.json")
    if d["status"] != "passed" or c["status"] != "passed":
        raise RuntimeError("D and C gates must pass before direct-rollout calibration")
    selected_id = d["selected"]["id"]
    selected_rule = d["selected"]["rule_id"]
    source_rows = {
        row["task_seed"]: row
        for row in (json.loads(line) for line in (
            experiment / "calibration/D/task_calibration.jsonl"
        ).read_text().splitlines())
    }
    models = study.load_models(SOURCE, "calibration")
    output = experiment / "calibration/frequency_rollout"
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "task_calibration.jsonl"
    completed = {}
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            row = json.loads(line)
            completed[row["task_seed"]] = row
    pending = []
    for index, model in enumerate(models):
        if model.task.seed in completed:
            continue
        source = source_rows[model.task.seed]
        budgets = {
            key: [condition["pools"][str(repeat)][1][selected_id] for repeat in range(REPEATS)]
            for key, condition in source["conditions"].items()
        }
        pending.append((index, budgets, selected_rule))
    started = time.monotonic()
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_task_frequency_calibration, *arguments) for arguments in pending]
            for future in futures:
                row = future.result()
                completed[row["task_seed"]] = row
                with rows_path.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                write_json(output / "progress.json", {
                    "status": "running", "completed_tasks": len(completed), "total_tasks": len(models),
                    "wall_seconds": time.monotonic() - started,
                    "updated_at_utc": datetime.now(UTC).isoformat(),
                })
    tasks = [completed[model.task.seed] for model in models]
    candidate_ids = list(tasks[0]["conditions"]["initial_H0"]["pools"]["0"])
    comparison = []
    for candidate_id in candidate_ids:
        nodes = []
        all_queries, all_episodes = [], []
        for update in study.D_NODES:
            recovery_errors, ranking_errors = [], []
            for repeat in range(REPEATS):
                recovery_rows, ranking_rows = [], []
                for task in tasks:
                    baseline = task["conditions"]["initial_H0"]
                    h0 = task["conditions"][f"u{update:04d}_H0"]
                    h3 = task["conditions"][f"u{update:04d}_H3"]
                    baseline_hat = baseline["pools"][str(repeat)][candidate_id]
                    h0_hat = h0["pools"][str(repeat)][candidate_id]
                    h3_hat = h3["pools"][str(repeat)][candidate_id]
                    recovery_rows.append({
                        "family": task["family"], "prediction": h3_hat["J_hat"] - baseline_hat["J_hat"],
                        "truth": h3["J"] - baseline["J"],
                    })
                    ranking_rows.append({
                        "family": task["family"], "prediction": h3_hat["J_hat"] - h0_hat["J_hat"],
                        "truth": h3["J"] - h0["J"],
                    })
                    for value in (baseline_hat, h0_hat, h3_hat):
                        all_queries.append(value["queries"])
                        all_episodes.append(value["episodes"])
                recovery_errors.append(
                    _family_macro(recovery_rows, "prediction") - _family_macro(recovery_rows, "truth")
                )
                ranking_errors.append(
                    _family_macro(ranking_rows, "prediction") - _family_macro(ranking_rows, "truth")
                )
            nodes.append({
                "update": update,
                "recovery_RMSE": float(np.sqrt(np.mean(np.square(recovery_errors)))),
                "ranking_RMSE": float(np.sqrt(np.mean(np.square(ranking_errors)))),
            })
        comparison.append({
            "id": candidate_id,
            "N_log": int(candidate_id.split("_")[0][4:]),
            "rule_id": selected_rule,
            "max_node_return_difference_RMSE": max(
                max(row["recovery_RMSE"], row["ranking_RMSE"]) for row in nodes
            ),
            "mean_queries": float(np.mean(all_queries)),
            "mean_completed_episodes": float(np.mean(all_episodes)),
            "node_metrics": nodes,
        })
    selected = min(comparison, key=lambda row: (row["max_node_return_difference_RMSE"], row["N_log"]))
    result = {
        "status": "complete", "tasks": len(tasks), "measurement_pools": REPEATS,
        "diagnostic_budget_candidate_id": selected_id,
        "selected": selected, "comparison": comparison,
        "rows_sha256": digest(rows_path),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "wall_seconds_this_invocation": time.monotonic() - started,
    }
    write_json(output / "selection.json", result)
    write_json(output / "progress.json", {"status": "complete", "completed_tasks": len(tasks)})
    print(json.dumps({"status": "complete", "selected": selected}))
    return result


def freeze_calibration(experiment=EXPERIMENT):
    experiment = Path(experiment).resolve()
    d_path = experiment / "calibration/D/acceptance.json"
    c_path = experiment / "calibration/C/acceptance.json"
    frequency_path = experiment / "calibration/frequency_rollout/selection.json"
    d, c, frequency = read(d_path), read(c_path), read(frequency_path)
    if d["status"] != "passed" or c["status"] != "passed":
        raise RuntimeError("Calibration gates did not pass; test preparation is forbidden")
    selected = d["selected"]
    old_id = "m64_N16_per_action_alpha_0.5"
    cross_conditions = {
        "old_budget_old_rule": old_id,
        "old_budget_selected_rule": f"m64_N16_{selected['rule_id']}",
        "selected_budget_old_rule": f"m{selected['m']}_N{selected['N']}_per_action_alpha_0.5",
        "selected_budget_selected_rule": selected["id"],
    }
    record = {
        "status": "frozen",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "old_test_used_for_selection": False,
        "new_test_used_for_selection": False,
        "choices": {
            "measurement": {
                "id": selected["id"], "m": selected["m"], "N": selected["N"],
                "rule_id": selected["rule_id"],
            },
            "cross_conditions": cross_conditions,
            "frequency_rollout": {
                "id": frequency["selected"]["id"],
                "N_log": frequency["selected"]["N_log"],
                "rule_id": frequency["selected"]["rule_id"],
            },
            "measurement_pool": 0,
            "rollouts_per_task_condition": 32,
            "classification_tolerance": .01,
            "bootstrap_samples": 2000,
            "bootstrap_seed": 6205,
        },
        "calibration": {
            "D": {"path": str(d_path), "sha256": digest(d_path)},
            "C": {"path": str(c_path), "sha256": digest(c_path)},
            "frequency_rollout": {"path": str(frequency_path), "sha256": digest(frequency_path)},
            "protocol_sha256": digest(experiment / "calibration/protocol.json"),
        },
    }
    path = experiment / "calibration/frozen_choices.json"
    if path.exists():
        previous = read(path)
        record["frozen_at_utc"] = previous["frozen_at_utc"]
        if previous != record:
            raise ValueError("Frozen measurement-v2 choices changed")
        return previous
    write_json(path, record)
    proposal = {
        "schema_version": 1,
        "task_id": "section51_measurement_v2_2026-09-16",
        "purpose": "independent held-out validation of the calibration-selected finite measurement",
        "start_seed": TEST_START_SEED,
        "family_seed_offset": 1_000_000,
        "families": list(FAMILIES),
        "test_tasks_per_family": 192,
        "gamma": study.GAMMA,
        "tau": study.TAU,
        "horizon": study.HORIZON,
        "training": "none; seed 42 checkpoints frozen",
        "physical_gpus": list(study.PHYSICAL_GPUS),
        "test_cells": list(TEST_CELLS),
        "frozen_choices_sha256": digest(path),
        "prediction_before_scoring": True,
    }
    write_json(experiment / "proposal.json", proposal)
    write_json(experiment / "progress.json", {
        "status": "calibration_frozen", "selected": selected["id"],
        "updated_at_utc": datetime.now(UTC).isoformat(),
    })
    return record


def prepare_test(experiment=EXPERIMENT):
    experiment = Path(experiment).resolve()
    freeze_calibration(experiment)
    manifest_path = experiment / "suite/manifest.json"
    if manifest_path.exists():
        return read(manifest_path)
    exclusion_path = experiment / "exclusions.json"
    exclusions = study.collect_exclusions(experiment)
    write_json(exclusion_path, exclusions)
    for path, expected in exclusions["manifest_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Exclusion manifest changed during scan: {path}")
    forbidden_contents = set(exclusions["content_sha256"])
    forbidden_seeds = set(exclusions["seeds"])
    suite = experiment / "suite"
    (suite / "tasks").mkdir(parents=True, exist_ok=True)
    journal = experiment / "task_journal.jsonl"
    entries = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    seen = {entry["content_sha256"] for entry in entries}
    if len(seen) != len(entries) or seen & forbidden_contents:
        raise AssertionError("Resumed V2 task journal violates exclusion rules")
    with Browser(horizon=study.HORIZON) as browser:
        for family_index, family in enumerate(FAMILIES):
            count = sum(entry["family"] == family for entry in entries)
            prior = [entry["seed"] for entry in entries if entry["family"] == family]
            seed = max(prior) + 1 if prior else TEST_START_SEED + family_index * 1_000_000
            scanned = 0
            while count < 192:
                if scanned >= 100_000:
                    raise RuntimeError(f"Fresh IID scan exhausted for {family}")
                task = browser.reset(seed, family).task
                seed += 1
                scanned += 1
                if task.seed in forbidden_seeds or task.fingerprint in forbidden_contents | seen or is_ood(task):
                    continue
                model = compile_task(task, study.HORIZON)
                success_steps = verify_success_path(browser, model)
                relative = f"tasks/{task.key}.json"
                path = suite / relative
                write_json(path, asdict(task))
                entry = {
                    "family": family, "seed": task.seed, "split": "test", "json": relative,
                    "sha256": digest(path), "content_sha256": task.fingerprint,
                    "verified_success_steps": success_steps, "actions": len(task.actions),
                    "physical_states": task.physical_states,
                    "native_reachable_states": len(reachable_states(model)),
                }
                with journal.open("a") as stream:
                    stream.write(json.dumps(entry) + "\n")
                entries.append(entry)
                seen.add(task.fingerprint)
                count += 1
                write_json(experiment / "preparation.json", {
                    "status": "running", "family": family, "count": count, "tasks": len(entries)
                })
    if len(entries) != 576 or len(seen) != 576:
        raise AssertionError("V2 test suite has the wrong number of unique tasks")
    family_counts = Counter(entry["family"] for entry in entries)
    if any(family_counts[family] != 192 for family in FAMILIES):
        raise AssertionError("V2 test suite family quotas differ")
    manifest = {
        "version": "v2", "status": "complete", "profile": "section51_measurement_v2",
        "horizon": study.HORIZON, "families": list(FAMILIES), "entries": entries,
        "exclusions_sha256": digest(exclusion_path),
        "validation": "live native success replay; all retained manifests excluded by seed and content",
    }
    write_json(manifest_path, manifest)
    write_json(experiment / "splits.json", {"test": [entry["seed"] for entry in entries]})
    write_json(experiment / "preparation.json", {
        "status": "complete", "tasks": len(entries), "family_counts": family_counts,
        "content_overlap": 0, "seed_overlap": 0,
        "excluded_manifests": len(exclusions["manifest_sha256"]),
    })
    write_json(experiment / "progress.json", {
        "status": "test_suite_prepared", "tasks": len(entries),
        "updated_at_utc": datetime.now(UTC).isoformat(),
    })
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "register", "calibrate-d", "calibrate-c", "calibrate-frequency",
            "freeze", "prepare-test",
        ),
    )
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "register":
        print(json.dumps(register_protocol(args.experiment), indent=2))
    elif args.command == "calibrate-d":
        calibrate_d(args.experiment, args.workers)
    elif args.command == "calibrate-c":
        calibrate_c(args.experiment, args.workers)
    elif args.command == "calibrate-frequency":
        calibrate_frequency_rollout(args.experiment, args.workers)
    elif args.command == "freeze":
        print(json.dumps(freeze_calibration(args.experiment), indent=2))
    elif args.command == "prepare-test":
        manifest = prepare_test(args.experiment)
        print(json.dumps({"status": manifest["status"], "tasks": len(manifest["entries"])}))


if __name__ == "__main__":
    main()
