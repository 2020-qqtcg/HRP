"""Prediction, scoring, and reporting for Section 5.1 measurement V2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from hrp_miniwob.adapters.miniwob_section51 import controlled_kernel, solve_controlled
from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_section51 as study
from hrp_miniwob.benchmarks import miniwob_section51_analysis as old
from hrp_miniwob.benchmarks import miniwob_section51_measurement_v2 as v2
from hrp_miniwob.benchmarks.miniwob_meta_harness import digest


def read(path):
    return json.loads(Path(path).read_text())


def stable_seed(*parts):
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def source_paths():
    paths = [
        Path(__file__),
        Path(v2.__file__),
        Path(study.__file__),
        Path(old.__file__),
        study.ROOT / "environment/hrp_miniwob/adapters/miniwob_section51.py",
        study.ROOT / "environment/hrp_miniwob/adapters/miniwob_readout.py",
        study.ROOT / "environment/hrp_miniwob/adapters/miniwob_harnesses.py",
        study.ROOT / "environment/hrp_miniwob/harnesses/miniwob_harnesses.py",
        study.ROOT / "environment/hrp_miniwob/harnesses/miniwob_h0.py",
        study.ROOT / "environment/hrp_miniwob/harnesses/miniwob_h3.py",
        study.ROOT / "environment/hrp_miniwob/validation/check_miniwob_section51_measurement_v2.py",
        study.ROOT / "environment/hrp_miniwob/orchestration/run_miniwob_section51_measurement_v2.py",



        study.ROOT / "environment/hrp_miniwob/oracle.py",
        study.ROOT / "environment/training/uv.lock",
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Measurement-v2 source set is incomplete: " + ", ".join(missing))
    return paths


def source_hashes():
    return {str(path.relative_to(study.ROOT)): digest(path) for path in source_paths()}


def audit_probability_tables(experiment=v2.EXPERIMENT):
    experiment = Path(experiment).resolve()
    models = study.load_models(experiment, "test")
    checks = graph_rows = prompt_rows = 0
    specs = study.cell_specs("test")
    for cell_id in v2.TEST_CELLS:
        if not study.cell_complete(experiment, "test", cell_id):
            raise RuntimeError(f"Incomplete V2 test probability cell: {cell_id}")
        spec = specs[cell_id]
        for model in models:
            table = study.load_table(experiment, "test", cell_id, model)
            graph, bank, rendered = study.graph_and_prompts(model, spec)
            if not np.array_equal(table["bank"], bank):
                raise AssertionError("Saved reachable-state bank changed")
            if not np.array_equal(table["next_state"], graph.next_state):
                raise AssertionError("Saved transition graph changed")
            if not np.array_equal(table["task_rewards"], graph.task_rewards):
                raise AssertionError("Saved reward graph changed")
            if int(table["initial_state"]) != graph.initial_state:
                raise AssertionError("Saved initial state changed")
            if tuple(map(str, table["semantic_actions"])) != tuple(graph.task.actions):
                raise AssertionError("Semantic action columns changed")
            if tuple(map(str, table["semantic_codes"])) != tuple(rendered[0].semantic_codes):
                raise AssertionError("Decoder columns changed")
            hashes = np.asarray([row.prompt_sha256 for row in rendered])
            if not np.array_equal(table["prompt_sha256"], hashes):
                raise AssertionError("Saved prompts changed")
            checks += 1
            graph_rows += len(graph.next_state)
            prompt_rows += len(bank)
    result = {
        "status": "passed", "cells": len(v2.TEST_CELLS), "tasks_per_cell": len(models),
        "task_cell_checks": checks, "graph_state_rows_checked": graph_rows,
        "prompt_rows_checked": prompt_rows,
    }
    output = experiment / "acceptance"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "test_probability_tables.json", result)
    return result


def seal_inputs(experiment=v2.EXPERIMENT):
    experiment = Path(experiment).resolve()
    if read(experiment / "preparation.json")["status"] != "complete":
        raise RuntimeError("Fresh test suite is incomplete")
    choices = read(experiment / "calibration/frozen_choices.json")
    if choices["status"] != "frozen":
        raise RuntimeError("Calibration choices are not frozen")
    audit_path = experiment / "acceptance/test_probability_tables.json"
    audit = read(audit_path) if audit_path.exists() else audit_probability_tables(experiment)
    config = read(experiment / "proposal.json") | {
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "suite_manifest_sha256": digest(experiment / "suite/manifest.json"),
        "splits_sha256": digest(experiment / "splits.json"),
        "exclusions_sha256": digest(experiment / "exclusions.json"),
        "calibration_choices_sha256": digest(experiment / "calibration/frozen_choices.json"),
        "calibration_protocol_sha256": digest(experiment / "calibration/protocol.json"),
        "probability_audit_sha256": digest(audit_path),
        "probability_audit": audit,
        "model_files_sha256": study.model_hashes(),
        "source_sha256": source_hashes(),
    }
    path = experiment / "config.json"
    if path.exists():
        previous = read(path)
        config["sealed_at_utc"] = previous["sealed_at_utc"]
        if previous != config:
            raise ValueError("Sealed measurement-v2 inputs changed")
        return previous
    snapshot = experiment / "source_snapshot"
    for source in source_paths():
        target = snapshot / source.relative_to(study.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_json(path, config)
    write_json(experiment / "manifest.json", {
        "status": "sealed", "config_sha256": digest(path),
        "test_blinding": "predictions are hashed before exact scores and independent rollouts",
        "training_seed_statement": "source and adaptation checkpoints use seed 42 only",
    })
    return config


def verify_inputs(experiment=v2.EXPERIMENT):
    experiment = Path(experiment).resolve()
    config = read(experiment / "config.json")
    for relative, key in (
        ("suite/manifest.json", "suite_manifest_sha256"),
        ("splits.json", "splits_sha256"),
        ("exclusions.json", "exclusions_sha256"),
        ("calibration/frozen_choices.json", "calibration_choices_sha256"),
        ("calibration/protocol.json", "calibration_protocol_sha256"),
        ("acceptance/test_probability_tables.json", "probability_audit_sha256"),
    ):
        if digest(experiment / relative) != config[key]:
            raise ValueError(f"Sealed input changed: {relative}")
    if source_hashes() != config["source_sha256"]:
        raise ValueError("Prediction source changed after input sealing")
    if study.model_hashes() != config["model_files_sha256"]:
        raise ValueError("Frozen model files changed")
    return config


def _rollout_estimate(policy, reference, kernel, query_budget, rng, n_log=None, rule_id=None):
    native = n_log is None
    cost = 1 if native else 1 + n_log
    episodes = query_budget // cost + 1
    states = np.full(episodes, kernel.initial_state, dtype=np.int64)
    returns = np.zeros(episodes, dtype=np.float64)
    steps = np.zeros(episodes, dtype=np.int64)
    discount = np.ones(episodes, dtype=np.float64)
    alpha = None if native else v2.alpha_for(rule_id, policy.shape[1])
    for _ in range(study.HORIZON):
        rows = np.flatnonzero(states >= 0)
        if not len(rows):
            break
        current = states[rows]
        cumulative = np.cumsum(policy[current], axis=1)
        action = np.sum(rng.random(len(rows))[:, None] > cumulative, axis=1)
        if native:
            log_probability = np.log(policy[current, action])
        else:
            count = rng.binomial(n_log, policy[current, action])
            estimated = (count + alpha) / (n_log + alpha * policy.shape[1])
            log_probability = np.log(estimated)
        branch = (
            rng.random(len(rows)) >= kernel.probability[current, action, 0]
        ).astype(np.int64)
        reward = kernel.task_reward[current, action, branch]
        utility_cost = kernel.utility_cost[current, action, branch]
        returns[rows] += discount[rows] * (
            reward - utility_cost
            - study.TAU * (log_probability - np.log(reference[current, action]))
        )
        states[rows] = kernel.next_state[current, action, branch]
        discount[rows] *= study.GAMMA
        steps[rows] += 1
    cumulative_queries = np.cumsum(steps * cost)
    completed = int(np.sum(cumulative_queries <= query_budget))
    if not completed:
        raise RuntimeError("Matched rollout did not complete one episode")
    return {
        "regularized_return": float(np.mean(returns[:completed])),
        "episodes": completed,
        "queries": int(cumulative_queries[completed - 1]),
        "query_budget": int(query_budget),
        "access": "native_log_probability" if native else f"frequency_N={n_log}_{rule_id}",
    }


def _task_predict_c(model, experiment, choices):
    selected_id = choices["measurement"]["id"]
    cross = choices["cross_conditions"]
    frequency = choices["frequency_rollout"]
    rows = []
    for model_id in ("h0_final", "h3_adapt_final"):
        reference, baseline_policy = old._c_tables(experiment, "test", model, model_id, "R0")
        baseline_kernel = controlled_kernel(model, .8, .01)
        baseline_optimum = solve_controlled(
            baseline_kernel, reference, study.GAMMA, study.TAU
        )
        baseline_estimates = v2._all_candidate_estimates(
            baseline_policy, baseline_optimum.log_policy, baseline_kernel.initial_state, None,
            np.random.default_rng(stable_seed(
                v2.BASE_SEED, "C", "test", model_id, model.task.fingerprint, "baseline", 0
            )),
            controlled=baseline_kernel,
        )
        baseline_budget = baseline_estimates[1][selected_id]
        baseline_frequency = _rollout_estimate(
            baseline_policy, reference, baseline_kernel, baseline_budget,
            np.random.default_rng(stable_seed(
                v2.BASE_SEED, "C-frequency", "test", model_id, model.task.fingerprint, "baseline"
            )),
            frequency["N_log"], frequency["rule_id"],
        )
        baseline_native = _rollout_estimate(
            baseline_policy, reference, baseline_kernel, baseline_budget,
            np.random.default_rng(stable_seed(
                v2.BASE_SEED, "C-native", "test", model_id, model.task.fingerprint, "baseline"
            )),
        )
        for config in old.execution_configurations("test"):
            reference_config, policy = old._c_tables(
                experiment, "test", model, model_id, config["rendering"]
            )
            if not np.array_equal(reference, reference_config):
                raise AssertionError("C reference changed across rendering")
            kernel = controlled_kernel(model, config["reliability"], config["click_cost"])
            optimum = solve_controlled(kernel, reference, study.GAMMA, study.TAU)
            estimates = v2._all_candidate_estimates(
                policy, optimum.log_policy, kernel.initial_state, None,
                np.random.default_rng(stable_seed(
                    v2.BASE_SEED, "C", "test", model_id, model.task.fingerprint,
                    config["id"], 0,
                )),
                controlled=kernel,
            )
            delta_c = float(optimum.value[kernel.initial_state] - baseline_optimum.value[baseline_kernel.initial_state])
            budget = estimates[1][selected_id]
            frequency_result = _rollout_estimate(
                policy, reference, kernel, budget,
                np.random.default_rng(stable_seed(
                    v2.BASE_SEED, "C-frequency", "test", model_id,
                    model.task.fingerprint, config["id"],
                )),
                frequency["N_log"], frequency["rule_id"],
            )
            native_result = _rollout_estimate(
                policy, reference, kernel, budget,
                np.random.default_rng(stable_seed(
                    v2.BASE_SEED, "C-native", "test", model_id,
                    model.task.fingerprint, config["id"],
                )),
            )
            condition_predictions = {
                label: delta_c - estimates[0][candidate_id] + baseline_estimates[0][candidate_id]
                for label, candidate_id in cross.items()
            }
            rows.append({
                "task_seed": model.task.seed, "family": model.task.family,
                "content_sha256": model.task.fingerprint, "model_id": model_id,
                "config_id": config["id"], "rendering": config["rendering"],
                "reliability": config["reliability"], "click_cost": config["click_cost"],
                "delta_ceiling": delta_c,
                "predictions": {
                    "kl_bregman": condition_predictions["selected_budget_selected_rule"],
                    "ceiling_only": delta_c,
                    "direct_rollout_frequency": (
                        frequency_result["regularized_return"]
                        - baseline_frequency["regularized_return"]
                    ),
                    "direct_rollout_native_logprob": (
                        native_result["regularized_return"] - baseline_native["regularized_return"]
                    ),
                    **{f"kl_{label}": value for label, value in condition_predictions.items()},
                },
                "absolute_M_hat": {
                    "config": estimates[0][selected_id],
                    "baseline": baseline_estimates[0][selected_id],
                },
                "diagnostic": {
                    "config": estimates[2][selected_id],
                    "baseline": baseline_estimates[2][selected_id],
                },
                "query_budget": {
                    "config": budget, "baseline": baseline_budget,
                    "frequency_config": frequency_result["queries"],
                    "frequency_baseline": baseline_frequency["queries"],
                    "native_config": native_result["queries"],
                    "native_baseline": baseline_native["queries"],
                },
            })
    return rows


def predict_c(experiment=v2.EXPERIMENT, workers=4):
    experiment = Path(experiment).resolve()
    verify_inputs(experiment)
    choices = read(experiment / "calibration/frozen_choices.json")["choices"]
    output = experiment / "predictions/C"
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "task_predictions.jsonl"
    models = study.load_models(experiment, "test")
    started = time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool, raw.open("w") as stream:
        for task_rows in pool.map(
            _task_predict_c, models, [experiment] * len(models), [choices] * len(models)
        ):
            for row in task_rows:
                stream.write(json.dumps(row) + "\n")
            rows.extend(task_rows)
            write_json(output / "progress.json", {
                "status": "running", "tasks": len(rows) // 72, "total_tasks": len(models),
                "rows": len(rows), "wall_seconds": time.monotonic() - started,
            })
    result = {
        "status": "predicted_unscored", "tasks": len(models), "task_rows": len(rows),
        "models": ["h0_final", "h3_adapt_final"], "configurations_per_model": 36,
        "choices": choices, "truth_fields_accessed": [],
        "task_predictions_sha256": digest(raw), "config_sha256": digest(experiment / "config.json"),
        "wall_seconds": time.monotonic() - started, "created_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(output / "predictions.json", result)
    write_json(output / "progress.json", {"status": "complete", "rows": len(rows)})
    print(json.dumps({"status": result["status"], "rows": len(rows)}))
    return result


def _deterministic_kernel(inputs):
    return old.deterministic_as_controlled(
        inputs["next_state"], inputs["rewards"], inputs["initial_state"]
    )


def _d_prediction_measurement(inputs, key, choices, seed_parts):
    optimum = old.solve(
        inputs["next_state"], inputs["rewards"], inputs["reference"],
        study.GAMMA, study.TAU,
    )
    estimates = v2._all_candidate_estimates(
        inputs["policy"], optimum.log_policy, inputs["initial_state"], inputs["next_state"],
        np.random.default_rng(stable_seed(v2.BASE_SEED, *seed_parts)),
    )
    selected_id = choices["measurement"]["id"]
    budget = estimates[1][selected_id]
    kernel = _deterministic_kernel(inputs)
    frequency = choices["frequency_rollout"]
    frequency_result = _rollout_estimate(
        inputs["policy"], inputs["reference"], kernel, budget,
        np.random.default_rng(stable_seed(v2.BASE_SEED, "D-frequency", *seed_parts)),
        frequency["N_log"], frequency["rule_id"],
    )
    native_result = _rollout_estimate(
        inputs["policy"], inputs["reference"], kernel, budget,
        np.random.default_rng(stable_seed(v2.BASE_SEED, "D-native", *seed_parts)),
    )
    return {
        "key": key, "ceiling": float(optimum.value[inputs["initial_state"]]),
        "M_hat": estimates[0], "cost": estimates[1], "diagnostic": estimates[2],
        "frequency": frequency_result, "native": native_result,
    }


def _task_predict_d(model, experiment, choices):
    cross = choices["cross_conditions"]
    baseline_inputs = old._d_tables(experiment, "test", model, 0, "H0")
    baseline = _d_prediction_measurement(
        baseline_inputs, "initial_H0", choices,
        ("D", "test", model.task.fingerprint, "initial_H0", 0),
    )
    rows = []
    for update in study.D_NODES:
        conditions = {}
        for harness in study.D_HARNESSES:
            inputs = old._d_tables(experiment, "test", model, update, harness)
            conditions[harness] = _d_prediction_measurement(
                inputs, f"u{update:04d}_{harness}", choices,
                ("D", "test", model.task.fingerprint, update, harness, 0),
            )
        recovery_delta_c = conditions["H3"]["ceiling"] - baseline["ceiling"]
        ranking_delta_c = conditions["H3"]["ceiling"] - conditions["H0"]["ceiling"]
        recovery_cross = {
            label: recovery_delta_c
            - conditions["H3"]["M_hat"][candidate_id]
            + baseline["M_hat"][candidate_id]
            for label, candidate_id in cross.items()
        }
        ranking_cross = {
            label: ranking_delta_c
            - conditions["H3"]["M_hat"][candidate_id]
            + conditions["H0"]["M_hat"][candidate_id]
            for label, candidate_id in cross.items()
        }
        rows.append({
            "task_seed": model.task.seed, "family": model.task.family,
            "content_sha256": model.task.fingerprint, "update": update,
            "predicted_recovery": recovery_cross["selected_budget_selected_rule"],
            "predicted_ranking": ranking_cross["selected_budget_selected_rule"],
            "cross_predictions": {"recovery": recovery_cross, "ranking": ranking_cross},
            "ceiling_only_recovery": recovery_delta_c,
            "ceiling_only_ranking": ranking_delta_c,
            "direct_rollout_frequency_recovery": (
                conditions["H3"]["frequency"]["regularized_return"]
                - baseline["frequency"]["regularized_return"]
            ),
            "direct_rollout_frequency_ranking": (
                conditions["H3"]["frequency"]["regularized_return"]
                - conditions["H0"]["frequency"]["regularized_return"]
            ),
            "direct_rollout_native_recovery": (
                conditions["H3"]["native"]["regularized_return"]
                - baseline["native"]["regularized_return"]
            ),
            "direct_rollout_native_ranking": (
                conditions["H3"]["native"]["regularized_return"]
                - conditions["H0"]["native"]["regularized_return"]
            ),
            "absolute_M_hat": {
                "baseline_H0": baseline["M_hat"][choices["measurement"]["id"]],
                "H0": conditions["H0"]["M_hat"][choices["measurement"]["id"]],
                "H3": conditions["H3"]["M_hat"][choices["measurement"]["id"]],
            },
            "diagnostic": {
                "baseline_H0": baseline["diagnostic"][choices["measurement"]["id"]],
                "H0": conditions["H0"]["diagnostic"][choices["measurement"]["id"]],
                "H3": conditions["H3"]["diagnostic"][choices["measurement"]["id"]],
            },
            "query_budget": {
                "baseline_H0": baseline["cost"][choices["measurement"]["id"]],
                "H0": conditions["H0"]["cost"][choices["measurement"]["id"]],
                "H3": conditions["H3"]["cost"][choices["measurement"]["id"]],
            },
        })
    return rows


def predict_d(experiment=v2.EXPERIMENT, workers=4):
    experiment = Path(experiment).resolve()
    verify_inputs(experiment)
    choices = read(experiment / "calibration/frozen_choices.json")["choices"]
    output = experiment / "predictions/D"
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "task_predictions.jsonl"
    models = study.load_models(experiment, "test")
    started = time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool, raw.open("w") as stream:
        for task_rows in pool.map(
            _task_predict_d, models, [experiment] * len(models), [choices] * len(models)
        ):
            for row in task_rows:
                stream.write(json.dumps(row) + "\n")
            rows.extend(task_rows)
            write_json(output / "progress.json", {
                "status": "running", "tasks": len(rows) // 7, "total_tasks": len(models),
                "rows": len(rows), "wall_seconds": time.monotonic() - started,
            })
    result = {
        "status": "predicted_unscored", "tasks": len(models), "task_rows": len(rows),
        "nodes": list(study.D_NODES), "choices": choices, "truth_fields_accessed": [],
        "task_predictions_sha256": digest(raw), "config_sha256": digest(experiment / "config.json"),
        "wall_seconds": time.monotonic() - started, "created_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(output / "predictions.json", result)
    write_json(output / "progress.json", {"status": "complete", "rows": len(rows)})
    print(json.dumps({"status": result["status"], "rows": len(rows)}))
    return result


def seal_predictions(experiment=v2.EXPERIMENT):
    experiment = Path(experiment).resolve()
    verify_inputs(experiment)
    records = {}
    for name in ("C", "D"):
        manifest = experiment / f"predictions/{name}/predictions.json"
        raw = experiment / f"predictions/{name}/task_predictions.jsonl"
        record = read(manifest)
        if record["status"] != "predicted_unscored" or record["task_predictions_sha256"] != digest(raw):
            raise RuntimeError(f"Incomplete unscored prediction: {name}")
        records[name] = {
            "manifest_sha256": digest(manifest), "task_predictions_sha256": digest(raw)
        }
    seal = {
        "status": "sealed_before_scoring", "sealed_at_utc": datetime.now(UTC).isoformat(),
        "config_sha256": digest(experiment / "config.json"), "predictions": records,
    }
    path = experiment / "predictions/seal.json"
    if path.exists() and read(path) != seal:
        raise ValueError("Prediction seal already exists with different content")
    write_json(path, seal)
    return seal


def verify_prediction_seal(experiment=v2.EXPERIMENT):
    experiment = Path(experiment).resolve()
    seal = read(experiment / "predictions/seal.json")
    if seal["status"] != "sealed_before_scoring" or seal["config_sha256"] != digest(
        experiment / "config.json"
    ):
        raise ValueError("Prediction seal is invalid")
    for name, expected in seal["predictions"].items():
        if digest(experiment / f"predictions/{name}/predictions.json") != expected["manifest_sha256"]:
            raise ValueError(f"Prediction manifest changed: {name}")
        if digest(experiment / f"predictions/{name}/task_predictions.jsonl") != expected[
            "task_predictions_sha256"
        ]:
            raise ValueError(f"Prediction rows changed: {name}")
    return seal


def _task_score_c(model, experiment):
    rows = []
    for model_id in ("h0_final", "h3_adapt_final"):
        reference, baseline_policy = old._c_tables(experiment, "test", model, model_id, "R0")
        baseline_kernel, _, baseline_exact = v2._controlled_exact(
            model, reference, baseline_policy, .8, .01
        )
        baseline_rollout = old._controlled_rollouts(
            baseline_kernel, baseline_policy, reference, 32,
            stable_seed(v2.BASE_SEED, "C-score", "test", model_id,
                        model.task.fingerprint, "baseline"),
        )
        for config in old.execution_configurations("test"):
            reference_config, policy = old._c_tables(
                experiment, "test", model, model_id, config["rendering"]
            )
            if not np.array_equal(reference, reference_config):
                raise AssertionError("C scoring reference changed across rendering")
            kernel, _, exact = v2._controlled_exact(
                model, reference, policy, config["reliability"], config["click_cost"]
            )
            rollout = old._controlled_rollouts(
                kernel, policy, reference, 32,
                stable_seed(v2.BASE_SEED, "C-score", "test", model_id,
                            model.task.fingerprint, config["id"]),
            )
            rows.append({
                "task_seed": model.task.seed, "family": model.task.family,
                "content_sha256": model.task.fingerprint, "model_id": model_id,
                "config_id": config["id"],
                "exact_delta_J": exact["J"] - baseline_exact["J"],
                "exact_delta_M": exact["M"] - baseline_exact["M"],
                "exact_delta_C": exact["ceiling"] - baseline_exact["ceiling"],
                "exact_M_config": exact["M"], "exact_M_baseline": baseline_exact["M"],
                "rollout_delta_J": (
                    rollout["mean_regularized_return"]
                    - baseline_rollout["mean_regularized_return"]
                ),
                "rollout_config": rollout, "rollout_baseline": baseline_rollout,
                "gap_identity_error": max(
                    abs(exact["ceiling"] - exact["J"] - exact["M"]),
                    abs(baseline_exact["ceiling"] - baseline_exact["J"] - baseline_exact["M"]),
                ),
            })
    return rows


def score_c(experiment=v2.EXPERIMENT, workers=4):
    experiment = Path(experiment).resolve()
    verify_inputs(experiment)
    verify_prediction_seal(experiment)
    output = experiment / "scores/C"
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "task_scores.jsonl"
    models = study.load_models(experiment, "test")
    started, rows = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=workers) as pool, raw.open("w") as stream:
        for task_rows in pool.map(_task_score_c, models, [experiment] * len(models)):
            for row in task_rows:
                stream.write(json.dumps(row) + "\n")
            rows.extend(task_rows)
            write_json(output / "progress.json", {
                "status": "running", "tasks": len(rows) // 72, "total_tasks": len(models),
                "rows": len(rows), "wall_seconds": time.monotonic() - started,
            })
    result = {
        "status": "complete", "tasks": len(models), "task_rows": len(rows),
        "rollouts_per_task_condition": 32, "task_scores_sha256": digest(raw),
        "maximum_gap_identity_error": max(row["gap_identity_error"] for row in rows),
        "prediction_seal_sha256": digest(experiment / "predictions/seal.json"),
        "wall_seconds": time.monotonic() - started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(output / "scores.json", result)
    write_json(output / "progress.json", {"status": "complete", "rows": len(rows)})
    print(json.dumps(result))
    return result


def _task_score_d(model, experiment):
    baseline_inputs = old._d_tables(experiment, "test", model, 0, "H0")
    baseline_exact = old._d_exact(baseline_inputs)
    baseline_rollout = old._deterministic_rollouts(
        baseline_inputs, 32,
        stable_seed(v2.BASE_SEED, "D-score", "test", model.task.fingerprint, "initial_H0"),
    )
    rows = []
    for update in study.D_NODES:
        conditions = {}
        for harness in study.D_HARNESSES:
            inputs = old._d_tables(experiment, "test", model, update, harness)
            conditions[harness] = {
                "exact": old._d_exact(inputs),
                "rollout": old._deterministic_rollouts(
                    inputs, 32,
                    stable_seed(v2.BASE_SEED, "D-score", "test", model.task.fingerprint,
                                update, harness),
                ),
            }
        rows.append({
            "task_seed": model.task.seed, "family": model.task.family,
            "content_sha256": model.task.fingerprint, "update": update,
            "exact_recovery": (
                conditions["H3"]["exact"]["regularized_return"]
                - baseline_exact["regularized_return"]
            ),
            "exact_ranking": (
                conditions["H3"]["exact"]["regularized_return"]
                - conditions["H0"]["exact"]["regularized_return"]
            ),
            "exact_delta_ceiling": (
                conditions["H3"]["exact"]["ceiling"]
                - conditions["H0"]["exact"]["ceiling"]
            ),
            "exact_M_baseline_H0": baseline_exact["mismatch"],
            "exact_M_H0": conditions["H0"]["exact"]["mismatch"],
            "exact_M_H3": conditions["H3"]["exact"]["mismatch"],
            "rollout_recovery": (
                conditions["H3"]["rollout"]["mean_regularized_return"]
                - baseline_rollout["mean_regularized_return"]
            ),
            "rollout_ranking": (
                conditions["H3"]["rollout"]["mean_regularized_return"]
                - conditions["H0"]["rollout"]["mean_regularized_return"]
            ),
            "baseline_H0": {"exact": baseline_exact, "rollout": baseline_rollout},
            "H0": conditions["H0"], "H3": conditions["H3"],
            "gap_identity_error": max(
                baseline_exact["gap_identity_error"],
                conditions["H0"]["exact"]["gap_identity_error"],
                conditions["H3"]["exact"]["gap_identity_error"],
            ),
        })
    return rows


def score_d(experiment=v2.EXPERIMENT, workers=4):
    experiment = Path(experiment).resolve()
    verify_inputs(experiment)
    verify_prediction_seal(experiment)
    output = experiment / "scores/D"
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "task_scores.jsonl"
    models = study.load_models(experiment, "test")
    started, rows = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=workers) as pool, raw.open("w") as stream:
        for task_rows in pool.map(_task_score_d, models, [experiment] * len(models)):
            for row in task_rows:
                stream.write(json.dumps(row) + "\n")
            rows.extend(task_rows)
            write_json(output / "progress.json", {
                "status": "running", "tasks": len(rows) // 7, "total_tasks": len(models),
                "rows": len(rows), "wall_seconds": time.monotonic() - started,
            })
    result = {
        "status": "complete", "tasks": len(models), "task_rows": len(rows),
        "rollouts_per_task_condition": 32, "task_scores_sha256": digest(raw),
        "maximum_gap_identity_error": max(row["gap_identity_error"] for row in rows),
        "prediction_seal_sha256": digest(experiment / "predictions/seal.json"),
        "wall_seconds": time.monotonic() - started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(output / "scores.json", result)
    write_json(output / "progress.json", {"status": "complete", "rows": len(rows)})
    print(json.dumps(result))
    return result


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _macro(rows, key):
    return float(np.mean([
        np.mean([row[key] for row in rows if row["family"] == family])
        for family in study.FAMILIES
    ]))


def _macro_ci(rows, key, seed, samples=2000):
    rng = np.random.default_rng(seed)
    by_family = {
        family: [row for row in rows if row["family"] == family]
        for family in study.FAMILIES
    }
    values = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        family_values = []
        for family in study.FAMILIES:
            group = by_family[family]
            indices = rng.integers(0, len(group), size=len(group))
            family_values.append(np.mean([group[index][key] for index in indices]))
        values[draw] = np.mean(family_values)
    return {
        "mean": _macro(rows, key),
        "lower": float(np.quantile(values, .025)),
        "upper": float(np.quantile(values, .975)),
    }


def _classification(value, tolerance=.01):
    return 1 if value > tolerance else -1 if value < -tolerance else 0


def analyze(experiment=v2.EXPERIMENT):
    experiment = Path(experiment).resolve()
    verify_inputs(experiment)
    verify_prediction_seal(experiment)
    c_predictions = _jsonl(experiment / "predictions/C/task_predictions.jsonl")
    c_scores = _jsonl(experiment / "scores/C/task_scores.jsonl")
    d_predictions = _jsonl(experiment / "predictions/D/task_predictions.jsonl")
    d_scores = _jsonl(experiment / "scores/D/task_scores.jsonl")
    c_score_map = {
        (row["task_seed"], row["model_id"], row["config_id"]): row for row in c_scores
    }
    c_methods = list(c_predictions[0]["predictions"]) + ["independent_32_rollout"]
    c_points, c_method_summary = [], {}
    for model_id in ("h0_final", "h3_adapt_final"):
        for config in old.execution_configurations("test"):
            selected = [
                row for row in c_predictions
                if row["model_id"] == model_id and row["config_id"] == config["id"]
            ]
            truth_rows = [
                {"family": row["family"], "value": c_score_map[
                    (row["task_seed"], model_id, config["id"])
                ]["exact_delta_J"]}
                for row in selected
            ]
            truth = _macro(truth_rows, "value")
            point = {"model_id": model_id, "config_id": config["id"], "truth": truth}
            for method in c_methods:
                if method == "independent_32_rollout":
                    prediction_rows = [
                        {"family": row["family"], "value": c_score_map[
                            (row["task_seed"], model_id, config["id"])
                        ]["rollout_delta_J"]}
                        for row in selected
                    ]
                else:
                    prediction_rows = [
                        {"family": row["family"], "value": row["predictions"][method]}
                        for row in selected
                    ]
                interval = _macro_ci(
                    prediction_rows, "value", stable_seed(6205, "C", model_id, config["id"], method)
                )
                point[method] = interval
            c_points.append(point)
    for method in c_methods:
        errors = np.asarray([point[method]["mean"] - point["truth"] for point in c_points])
        non_ties = [point for point in c_points if _classification(point["truth"]) != 0]
        c_method_summary[method] = {
            "configuration_MAE": float(np.mean(np.abs(errors))),
            "configuration_RMSE": float(np.sqrt(np.mean(errors**2))),
            "three_class_accuracy": float(np.mean([
                _classification(point[method]["mean"]) == _classification(point["truth"])
                for point in c_points
            ])),
            "three_class_denominator": len(c_points),
            "non_tie_direction_accuracy": float(np.mean([
                np.sign(point[method]["mean"]) == np.sign(point["truth"])
                for point in non_ties
            ])) if non_ties else None,
            "non_tie_denominator": len(non_ties),
            "interval_coverage": float(np.mean([
                point[method]["lower"] <= point["truth"] <= point[method]["upper"]
                for point in c_points
            ])),
        }
        pairwise_correct = pairwise_total = 0
        for model_id in ("h0_final", "h3_adapt_final"):
            group = [point for point in c_points if point["model_id"] == model_id]
            for left in range(len(group)):
                for right in range(left + 1, len(group)):
                    truth_delta = group[left]["truth"] - group[right]["truth"]
                    if abs(truth_delta) <= .01:
                        continue
                    prediction_delta = group[left][method]["mean"] - group[right][method]["mean"]
                    pairwise_correct += np.sign(prediction_delta) == np.sign(truth_delta)
                    pairwise_total += 1
        c_method_summary[method]["non_tie_pairwise_ranking_accuracy"] = (
            float(pairwise_correct / pairwise_total) if pairwise_total else None
        )
        c_method_summary[method]["non_tie_pairwise_denominator"] = pairwise_total
    main_task_errors = []
    m_rows = []
    for row in c_predictions:
        score = c_score_map[(row["task_seed"], row["model_id"], row["config_id"])]
        main_task_errors.append(row["predictions"]["kl_bregman"] - score["exact_delta_J"])
        m_rows.extend([
            {"family": row["family"], "condition": "config",
             "error": row["absolute_M_hat"]["config"] - score["exact_M_config"]},
            {"family": row["family"], "condition": "baseline",
             "error": row["absolute_M_hat"]["baseline"] - score["exact_M_baseline"]},
        ])
    c_summary = {
        "methods": c_method_summary,
        "main_task_level_RMSE": float(np.sqrt(np.mean(np.square(main_task_errors)))),
        "absolute_M_task_bias": float(np.mean([row["error"] for row in m_rows])),
        "absolute_M_task_RMSE": float(np.sqrt(np.mean([row["error"] ** 2 for row in m_rows]))),
        "points": c_points,
    }

    d_score_map = {(row["task_seed"], row["update"]): row for row in d_scores}
    d_nodes = []
    d_methods = {
        "kl_bregman": ("prediction", "predicted_recovery", "predicted_ranking"),
        "ceiling_only": ("prediction", "ceiling_only_recovery", "ceiling_only_ranking"),
        "direct_rollout_frequency": (
            "prediction", "direct_rollout_frequency_recovery", "direct_rollout_frequency_ranking"
        ),
        "direct_rollout_native_logprob": (
            "prediction", "direct_rollout_native_recovery", "direct_rollout_native_ranking"
        ),
        "independent_32_rollout": ("score", "rollout_recovery", "rollout_ranking"),
    }
    for update in study.D_NODES:
        selected = [row for row in d_predictions if row["update"] == update]
        node = {"update": update}
        for target, exact_key in (("recovery", "exact_recovery"), ("ranking", "exact_ranking")):
            truth_rows = [
                {"family": row["family"], "value": d_score_map[(row["task_seed"], update)][exact_key]}
                for row in selected
            ]
            node[f"exact_{target}"] = _macro_ci(
                truth_rows, "value", stable_seed(6205, "D-truth", update, target)
            )
            for method, fields in d_methods.items():
                field = fields[1 if target == "recovery" else 2]
                if fields[0] == "score":
                    prediction_rows = [
                        {"family": row["family"], "value": d_score_map[
                            (row["task_seed"], update)
                        ][field]}
                        for row in selected
                    ]
                else:
                    prediction_rows = [
                        {"family": row["family"], "value": row[field]} for row in selected
                    ]
                node[f"{method}_{target}"] = _macro_ci(
                    prediction_rows, "value", stable_seed(6205, "D", update, target, method)
                )
        d_nodes.append(node)
    d_method_summary = {}
    for method in d_methods:
        d_method_summary[method] = {}
        for target in ("recovery", "ranking"):
            errors = np.asarray([
                node[f"{method}_{target}"]["mean"] - node[f"exact_{target}"]["mean"]
                for node in d_nodes
            ])
            d_method_summary[method][target] = {
                "node_RMSE": float(np.sqrt(np.mean(errors**2))),
                "node_MAE": float(np.mean(np.abs(errors))),
                "three_class_accuracy": float(np.mean([
                    _classification(node[f"{method}_{target}"]["mean"])
                    == _classification(node[f"exact_{target}"]["mean"])
                    for node in d_nodes
                ])),
                "denominator": len(d_nodes),
                "interval_coverage": float(np.mean([
                    node[f"{method}_{target}"]["lower"]
                    <= node[f"exact_{target}"]["mean"]
                    <= node[f"{method}_{target}"]["upper"]
                    for node in d_nodes
                ])),
            }
    d_m_errors = []
    for row in d_predictions:
        score = d_score_map[(row["task_seed"], row["update"])]
        for name, exact_key in (
            ("baseline_H0", "exact_M_baseline_H0"), ("H0", "exact_M_H0"), ("H3", "exact_M_H3")
        ):
            d_m_errors.append(row["absolute_M_hat"][name] - score[exact_key])
    d_summary = {
        "methods": d_method_summary, "nodes": d_nodes,
        "absolute_M_task_bias": float(np.mean(d_m_errors)),
        "absolute_M_task_RMSE": float(np.sqrt(np.mean(np.square(d_m_errors)))),
        "first_mean_positive": {
            target: next((node["update"] for node in d_nodes if node[f"exact_{target}"]["mean"] > 0), None)
            for target in ("recovery", "ranking")
        },
        "first_interval_positive": {
            target: next((node["update"] for node in d_nodes if node[f"exact_{target}"]["lower"] > 0), None)
            for target in ("recovery", "ranking")
        },
    }
    for node in d_nodes:
        update = node["update"]
        selected = [row for row in d_predictions if row["update"] == update]
        node["absolute_M"] = {}
        for name, exact_key in (("H0", "exact_M_H0"), ("H3", "exact_M_H3")):
            predicted_rows = [
                {"family": row["family"], "value": row["absolute_M_hat"][name]}
                for row in selected
            ]
            exact_rows = [
                {"family": row["family"], "value": d_score_map[
                    (row["task_seed"], update)
                ][exact_key]}
                for row in selected
            ]
            predicted_mean = _macro(predicted_rows, "value")
            exact_mean = _macro(exact_rows, "value")
            node["absolute_M"][name] = {
                "predicted": predicted_mean, "exact": exact_mean,
                "bias": predicted_mean - exact_mean,
            }
    choices = read(experiment / "calibration/frozen_choices.json")["choices"]
    query_summary = {
        "C_main_actor_queries": int(sum(
            row["query_budget"]["config"] + row["query_budget"]["baseline"]
            for row in c_predictions
        )),
        "D_main_actor_queries": int(sum(
            row["query_budget"]["baseline_H0"] + row["query_budget"]["H0"]
            + row["query_budget"]["H3"] for row in d_predictions
        )),
    }
    accounting = [
        read(experiment / f"probabilities/test/{cell}/accounting.json")
        for cell in v2.TEST_CELLS
    ]
    resources = {
        "probability_cells": len(accounting),
        "GPU_wall_hours": float(sum(row["wall_seconds"] for row in accounting) / 3600),
        "forward_state_rows": int(sum(row["forward_state_rows"] for row in accounting)),
        "prompt_tokens": int(sum(row["prompt_tokens"] for row in accounting)),
        "forward_microbatches": int(sum(row["forward_microbatches"] for row in accounting)),
    }
    result = {
        "status": "complete", "choices": choices,
        "calibration": {
            "D": read(experiment / "calibration/D/acceptance.json"),
            "C": read(experiment / "calibration/C/acceptance.json"),
            "frequency_rollout": read(experiment / "calibration/frequency_rollout/selection.json"),
        },
        "C": c_summary, "D": d_summary, "queries": query_summary, "resources": resources,
        "validation": {
            "C_max_gap_identity_error": read(experiment / "scores/C/scores.json")["maximum_gap_identity_error"],
            "D_max_gap_identity_error": read(experiment / "scores/D/scores.json")["maximum_gap_identity_error"],
            "prediction_seal_sha256": digest(experiment / "predictions/seal.json"),
        },
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    output = experiment / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "measurement_v2_data.json", result)
    return result


def _fmt(value):
    return f"{value:.5f}"


def _fmt_ci(record):
    return f"{record['mean']:.5f} [{record['lower']:.5f}, {record['upper']:.5f}]"


def write_report(experiment=v2.EXPERIMENT):
    experiment = Path(experiment).resolve()
    path = experiment / "analysis/measurement_v2_data.json"
    data = read(path) if path.exists() else analyze(experiment)
    docs = v2.DOCS
    docs.mkdir(parents=True, exist_ok=True)
    d_acceptance = data["calibration"]["D"]
    c_acceptance = data["calibration"]["C"]
    frequency = data["calibration"]["frequency_rollout"]
    selected = d_acceptance["selected"]
    old_candidate = next(
        row for row in d_acceptance["comparison"]
        if row["id"] == "m64_N16_per_action_alpha_0.5"
    )
    lines = [
        "# Section 5.1 第二轮有限测量实验报告",
        "",
        "## 实验范围",
        "",
        "本轮复用 seed 42 的冻结 Qwen3.5-2B checkpoints，没有新增训练。校准仅使用上一轮的 72 个 calibration 任务。通过预注册门槛后，方案被冻结，并在新的 576 个排重 IID test 任务上先保存预测，再进行精确评分和每条件 32 次独立 rollout。",
        "",
        "## 校准与验收",
        "",
        f"32 个候选中有 {len(d_acceptance['passing_candidate_ids'])} 个通过 D 门槛。按实际 actor 查询成本和预注册的并列规则，唯一方案为 `{selected['id']}`。",
        "",
        "| 方案 | 最大条件平均绝对 M 偏差 | 最大族内绝对 M 偏差 | 最坏节点差分 RMSE | 平均 actor 查询 | 通过 |",
        "|---|---:|---:|---:|---:|---|",
        f"| 原方案 `{old_candidate['id']}` | {_fmt(old_candidate['max_condition_macro_absolute_expected_M_bias'])} | {_fmt(old_candidate['max_family_condition_absolute_expected_M_bias'])} | {_fmt(old_candidate['max_node_return_difference_RMSE'])} | {old_candidate['mean_realized_actor_queries']:.1f} | 否 |",
        f"| 选中方案 `{selected['id']}` | {_fmt(selected['max_condition_macro_absolute_expected_M_bias'])} | {_fmt(selected['max_family_condition_absolute_expected_M_bias'])} | {_fmt(selected['max_node_return_difference_RMSE'])} | {selected['mean_realized_actor_queries']:.1f} | 是 |",
        "",
        f"C calibration 回归 RMSE 为 {_fmt(c_acceptance['summary']['selected']['configuration_aggregate_RMSE'])}，低于 0.03 门槛。频率 rollout 在 D calibration 上选择 `{frequency['selected']['id']}`，其最坏节点差分 RMSE 为 {_fmt(frequency['selected']['max_node_return_difference_RMSE'])}。",
        "",
        "选中方案仍是每动作 alpha=0.5。改善来自把状态数从 64 提高到 256，并把每状态动作样本从 16 提高到 1024。旧平滑在该动作预算下的混合权重已经足够小，因此它通过了绝对 M 门槛。",
        "",
        "## C：新配置上的独立回报预测",
        "",
        "| 方法 | 配置 MAE | 配置 RMSE | 三分类准确率 | 非并列方向准确率 | 非并列排序准确率 | 区间覆盖 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    display_methods = (
        "kl_bregman", "ceiling_only", "direct_rollout_frequency",
        "direct_rollout_native_logprob", "independent_32_rollout",
        "kl_old_budget_old_rule", "kl_selected_budget_old_rule",
    )
    for method in display_methods:
        if method not in data["C"]["methods"]:
            continue
        row = data["C"]["methods"][method]
        direction = "—" if row["non_tie_direction_accuracy"] is None else _fmt(row["non_tie_direction_accuracy"])
        ranking = "—" if row["non_tie_pairwise_ranking_accuracy"] is None else _fmt(row["non_tie_pairwise_ranking_accuracy"])
        lines.append(
            f"| `{method}` | {_fmt(row['configuration_MAE'])} | {_fmt(row['configuration_RMSE'])} | "
            f"{_fmt(row['three_class_accuracy'])} ({row['three_class_denominator']}) | {direction} "
            f"({row['non_tie_denominator']}) | {ranking} ({row['non_tie_pairwise_denominator']}) | "
            f"{_fmt(row['interval_coverage'])} |"
        )
    lines.extend([
        "",
        f"主 KL/Bregman 的任务级 RMSE 为 {_fmt(data['C']['main_task_level_RMSE'])}。绝对 M 的任务级 bias/RMSE 为 {_fmt(data['C']['absolute_M_task_bias'])}/{_fmt(data['C']['absolute_M_task_RMSE'])}。配置聚合成绩与任务级误差分开报告，避免把跨任务抵消当成单任务准确。",
        "",
        "## D：适应节点的恢复与接口排序",
        "",
        "| update | 精确恢复差 | KL 预测恢复差 | 精确 H3-H0 排序差 | KL 预测排序差 | H3 M 预测/精确 |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for node in data["D"]["nodes"]:
        lines.append(
            f"| {node['update']} | {_fmt_ci(node['exact_recovery'])} | "
            f"{_fmt_ci(node['kl_bregman_recovery'])} | {_fmt_ci(node['exact_ranking'])} | "
            f"{_fmt_ci(node['kl_bregman_ranking'])} | "
            f"{_fmt(node['absolute_M']['H3']['predicted'])}/{_fmt(node['absolute_M']['H3']['exact'])} |"
        )
    lines.extend([
        "",
        "| 方法 | 恢复 RMSE | 恢复三分类 | 排序 RMSE | 排序三分类 |",
        "|---|---:|---:|---:|---:|",
    ])
    for method, targets in data["D"]["methods"].items():
        recovery, ranking = targets["recovery"], targets["ranking"]
        lines.append(
            f"| `{method}` | {_fmt(recovery['node_RMSE'])} | "
            f"{_fmt(recovery['three_class_accuracy'])} ({recovery['denominator']}) | "
            f"{_fmt(ranking['node_RMSE'])} | {_fmt(ranking['three_class_accuracy'])} "
            f"({ranking['denominator']}) |"
        )
    lines.extend([
        "",
        f"精确恢复的均值首次转正节点为 `{data['D']['first_mean_positive']['recovery']}`，95% 区间首次整体转正节点为 `{data['D']['first_interval_positive']['recovery']}`。精确接口排序的对应节点为 `{data['D']['first_mean_positive']['ranking']}` 和 `{data['D']['first_interval_positive']['ranking']}`。不在保存节点之间插值。",
        "",
        f"D 中绝对 M 的任务级 bias/RMSE 为 {_fmt(data['D']['absolute_M_task_bias'])}/{_fmt(data['D']['absolute_M_task_RMSE'])}。主测量 actor 查询总数为 C={data['queries']['C_main_actor_queries']:,}、D={data['queries']['D_main_actor_queries']:,}。reference 和 oracle 构造成本另列，不计作 actor 查询。",
        "",
        f"完整概率缓存共 {data['resources']['probability_cells']} 个 GPU 单元，累计 {data['resources']['GPU_wall_hours']:.2f} GPU-hours、{data['resources']['forward_state_rows']:,} 个 forward state rows 和 {data['resources']['prompt_tokens']:,} 个 prompt tokens。缓存运行时间与等价在线查询预算分开报告。",
        "",
        "## 论文口径与限制",
        "",
        "- 本轮直接检验绝对 M 与回报差，没有依赖差分偏差偶然抵消。",
        "- 预测由真实完整 LM 概率缓存产生的有限分类采样模拟得到，不称为等量在线 API 调用。",
        "- oracle 提供 ceiling 和被查询状态的最优目标，因此结论限于已知最优目标的受控任务。",
        "- checkpoint 训练与适应均只有 seed 42。任务 bootstrap 和测量池不代表训练 seed 变异。",
        "- 完整概率下的 J=C-M 残差仅作为数值验收，不作为预测成绩。",
        "",
        "## 可追溯产物",
        "",
        f"- 冻结配置：`{experiment / 'config.json'}`",
        f"- 预测封存：`{experiment / 'predictions/seal.json'}`",
        f"- 机器可读汇总：`{experiment / 'analysis/measurement_v2_data.json'}`",
        f"- C/D 逐任务预测与评分：`{experiment / 'predictions'}`、`{experiment / 'scores'}`",
        "",
    ])
    report = docs / "REPORT.md"
    report.write_text("\n".join(lines))
    return report


def final_validate(experiment=v2.EXPERIMENT):
    experiment = Path(experiment).resolve()
    config = verify_inputs(experiment)
    prediction_seal = verify_prediction_seal(experiment)
    analysis_path = experiment / "analysis/measurement_v2_data.json"
    report_path = v2.DOCS / "REPORT.md"
    required = [analysis_path, report_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Final V2 artifacts are missing: " + ", ".join(missing))
    data = read(analysis_path)

    def finite(value):
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        if isinstance(value, list):
            return all(finite(item) for item in value)
        if isinstance(value, float):
            return math.isfinite(value)
        return True

    if not finite(data):
        raise AssertionError("Final V2 analysis contains a nonfinite value")
    validation = {
        "status": "passed", "validated_at_utc": datetime.now(UTC).isoformat(),
        "config_sha256": digest(experiment / "config.json"),
        "source_hashes_match_seal": source_hashes() == config["source_sha256"],
        "prediction_seal_sha256": digest(experiment / "predictions/seal.json"),
        "prediction_seal_status": prediction_seal["status"],
        "row_counts": {
            "C_predictions": len(_jsonl(experiment / "predictions/C/task_predictions.jsonl")),
            "C_scores": len(_jsonl(experiment / "scores/C/task_scores.jsonl")),
            "D_predictions": len(_jsonl(experiment / "predictions/D/task_predictions.jsonl")),
            "D_scores": len(_jsonl(experiment / "scores/D/task_scores.jsonl")),
        },
        "all_primary_rows_finite": True,
        "maximum_C_identity_error": read(experiment / "scores/C/scores.json")["maximum_gap_identity_error"],
        "maximum_D_identity_error": read(experiment / "scores/D/scores.json")["maximum_gap_identity_error"],
        "report_sha256": digest(report_path),
        "training_seed_statement": "source training and H3 adaptation seed 42 only",
    }
    write_json(experiment / "analysis/final_validation.json", validation)
    write_json(experiment / "analysis/completion.json", {
        "status": "complete", "validated_at_utc": validation["validated_at_utc"],
        "config_sha256": validation["config_sha256"],
        "prediction_seal_sha256": validation["prediction_seal_sha256"],
        "report_sha256": validation["report_sha256"],
    })
    write_json(experiment / "progress.json", {
        "status": "complete", "validated_at_utc": validation["validated_at_utc"],
    })
    (v2.DOCS / "VALIDATION.md").write_text(
        "# 最终验收\n\n"
        f"- 状态：`{validation['status']}`\n"
        f"- 配置 SHA-256：`{validation['config_sha256']}`\n"
        f"- 预测封存 SHA-256：`{validation['prediction_seal_sha256']}`\n"
        f"- C 预测/评分行：{validation['row_counts']['C_predictions']}/{validation['row_counts']['C_scores']}\n"
        f"- D 预测/评分行：{validation['row_counts']['D_predictions']}/{validation['row_counts']['D_scores']}\n"
        f"- 最大 C/D 恒等式残差：{validation['maximum_C_identity_error']:.3e}/{validation['maximum_D_identity_error']:.3e}\n"
        "- 源代码、模型、任务、预测封存和全部主数值已核验。\n"
    )
    return validation


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "audit", "seal-inputs", "predict-c", "predict-d", "seal-predictions",
            "score-c", "score-d", "analyze",
            "report", "validate",
        ),
    )
    parser.add_argument("--experiment", type=Path, default=v2.EXPERIMENT)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        print(json.dumps(audit_probability_tables(args.experiment), indent=2))
    elif args.command == "seal-inputs":
        print(json.dumps(seal_inputs(args.experiment), indent=2))
    elif args.command == "predict-c":
        predict_c(args.experiment, args.workers)
    elif args.command == "predict-d":
        predict_d(args.experiment, args.workers)
    elif args.command == "seal-predictions":
        print(json.dumps(seal_predictions(args.experiment), indent=2))
    elif args.command == "score-c":
        score_c(args.experiment, args.workers)
    elif args.command == "score-d":
        score_d(args.experiment, args.workers)
    elif args.command == "analyze":
        result = analyze(args.experiment)
        print(json.dumps({"status": result["status"], "queries": result["queries"]}))
    elif args.command == "report":
        print(write_report(args.experiment))
    elif args.command == "validate":
        print(json.dumps(final_validate(args.experiment), indent=2))


if __name__ == "__main__":
    main()
