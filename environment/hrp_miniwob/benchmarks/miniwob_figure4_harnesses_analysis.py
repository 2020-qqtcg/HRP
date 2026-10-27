"""Calibration, blinded prediction, scoring and reporting for Figure 4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_figure4_harnesses as study
from hrp_miniwob.benchmarks import miniwob_section51_analysis as diagnostics
from hrp_miniwob.benchmarks.miniwob_meta_harness import digest
from hrp_miniwob.oracle import solve

M, N, ALPHA = 256, 1024, .5
CALIBRATION_REPEATS = 32
BOOTSTRAP_SAMPLES = 2000
BASE_SEED = 7404


def read(path):
    return json.loads(Path(path).read_text())


def stable_seed(*parts):
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _inputs(experiment, split, harness, model):
    table = study.load_table(experiment, split, harness, model)
    return {
        "next_state": table["next_state"], "rewards": table["task_rewards"],
        "initial_state": int(table["initial_state"]),
        "reference": study.full_policy(table, "log_reference"),
        "policy": study.full_policy(table, "log_policy"),
        "bank": table["bank"], "actions": tuple(map(str, table["semantic_actions"])),
    }


def _oracle(inputs):
    return solve(
        inputs["next_state"], inputs["rewards"], inputs["reference"],
        study.GAMMA, study.TAU,
    )


def _finite_measurement(inputs, optimum, seed):
    environment = diagnostics.SamplingEnvironment(
        inputs["policy"], inputs["initial_state"], next_state=inputs["next_state"]
    )
    result = diagnostics.single_probe(
        environment, optimum.log_policy, M, N, ALPHA, np.random.default_rng(seed)
    )
    return {
        "M_hat": result["mismatch"],
        "pilot_decisions": result["pilot_decisions"],
        "action_samples": result["action_samples"],
        "nominal_action_samples": result["nominal_action_samples"],
        "actor_queries": result["pilot_decisions"] + result["action_samples"],
        "absorbing_fraction": result["absorbing_fraction"],
        "zero_count_fraction": result["zero_count_fraction"],
        "energy_upper_bound": result["energy_upper_bound"],
    }


def _exact(inputs):
    result = diagnostics.deterministic_exact(
        inputs["next_state"], inputs["rewards"], inputs["reference"],
        inputs["policy"], inputs["initial_state"],
    )
    return {key: value for key, value in result.items()
            if key not in {"occupancy", "optimal_policy", "optimal_log_policy", "energy_residual"}}


def _task_calibration(arguments):
    model, experiment = arguments
    rows = []
    for harness in study.HARNESSES:
        inputs = _inputs(experiment, "calibration", harness, model)
        optimum = _oracle(inputs)
        exact = _exact(inputs)
        for repeat in range(CALIBRATION_REPEATS):
            measurement = _finite_measurement(
                inputs, optimum,
                stable_seed(BASE_SEED, "calibration", model.task.fingerprint, harness, repeat),
            )
            rows.append({
                "task_seed": model.task.seed, "family": model.task.family,
                "content_sha256": model.task.fingerprint, "harness": harness,
                "repeat": repeat, "true_M": exact["mismatch"],
                "ceiling": exact["ceiling"], "J_exact": exact["regularized_return"],
                **measurement, "M_error": measurement["M_hat"] - exact["mismatch"],
            })
    return rows


def _macro(rows, key):
    return float(np.mean([
        np.mean([row[key] for row in rows if row["family"] == family])
        for family in study.section51.FAMILIES
    ]))


def calibrate(experiment=study.EXPERIMENT, workers=4):
    experiment = Path(experiment).resolve()
    for harness in study.HARNESSES:
        if not study.cell_complete(experiment, "calibration", harness):
            raise RuntimeError(f"Calibration probability cell incomplete: {harness}")
    output = experiment / "calibration"
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "task_measurements.jsonl"
    task_models = study.models("calibration")
    rows, started = [], time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool, raw.open("w") as stream:
        for task_rows in pool.map(_task_calibration, [(model, experiment) for model in task_models]):
            for row in task_rows:
                stream.write(json.dumps(row) + "\n")
            rows.extend(task_rows)
    summary_rows = []
    for harness in study.HARNESSES:
        selected = [row for row in rows if row["harness"] == harness]
        repeat_biases = []
        for repeat in range(CALIBRATION_REPEATS):
            pool_rows = [row for row in selected if row["repeat"] == repeat]
            repeat_biases.append(_macro(pool_rows, "M_error"))
        family = {}
        for name in study.section51.FAMILIES:
            group = [row for row in selected if row["family"] == name]
            family[name] = {
                "bias": float(np.mean([row["M_error"] for row in group])),
                "RMSE": float(np.sqrt(np.mean(np.square([row["M_error"] for row in group])))),
            }
        summary_rows.append({
            "harness": harness,
            "macro_bias": float(np.mean(repeat_biases)),
            "macro_RMSE_across_pools": float(np.sqrt(np.mean(np.square(repeat_biases)))),
            "task_pool_RMSE": float(np.sqrt(np.mean(np.square([row["M_error"] for row in selected])))),
            "mean_actor_queries": float(np.mean([row["actor_queries"] for row in selected])),
            "mean_zero_count_fraction": float(np.mean([row["zero_count_fraction"] for row in selected])),
            "family": family,
        })
    result = {
        "status": "complete", "tasks": len(task_models), "harnesses": list(study.HARNESSES),
        "repeats": CALIBRATION_REPEATS, "rows": len(rows),
        "fixed_estimator": {"m": M, "N": N, "alpha": ALPHA},
        "selection": "none; estimator inherited unchanged from Measurement V2",
        "summary": summary_rows, "task_measurements_sha256": digest(raw),
        "wall_seconds": time.monotonic() - started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(output / "summary.json", result)
    return result


def _task_predict(arguments):
    model, experiment = arguments
    rows = []
    for harness in study.HARNESSES:
        inputs = _inputs(experiment, "test", harness, model)
        optimum = _oracle(inputs)
        measurement = _finite_measurement(
            inputs, optimum,
            stable_seed(BASE_SEED, "test-prediction", model.task.fingerprint, harness, 0),
        )
        ceiling = float(optimum.value[inputs["initial_state"]])
        rows.append({
            "task_seed": model.task.seed, "family": model.task.family,
            "content_sha256": model.task.fingerprint, "harness": harness,
            "ceiling": ceiling, **measurement,
            "J_pred": ceiling - measurement["M_hat"],
        })
    return rows


def predict(experiment=study.EXPERIMENT, workers=4):
    experiment = Path(experiment).resolve()
    study.verify_seal(experiment)
    study.audit_tables(experiment, "test")
    output = experiment / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "task_predictions.jsonl"
    task_models = study.models("test")
    rows, started = [], time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool, raw.open("w") as stream:
        for task_rows in pool.map(_task_predict, [(model, experiment) for model in task_models]):
            for row in task_rows:
                stream.write(json.dumps(row) + "\n")
            rows.extend(task_rows)
            write_json(output / "progress.json", {
                "status": "running", "rows": len(rows), "total_rows": 3456,
                "updated_at_utc": datetime.now(UTC).isoformat(),
            })
    result = {
        "status": "predicted_unscored", "split": "test", "tasks": len(task_models),
        "task_rows": len(rows), "harnesses": list(study.HARNESSES),
        "estimator": {"m": M, "N": N, "alpha": ALPHA},
        "truth_fields_accessed": [],
        "source_note": "finite samples simulated from complete frozen LM probability caches; equivalent actor-query counts are not actual API calls",
        "task_predictions_sha256": digest(raw), "config_sha256": digest(experiment / "config.json"),
        "wall_seconds": time.monotonic() - started,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    if len(rows) != 3456:
        raise AssertionError(f"Expected 3456 predictions, found {len(rows)}")
    write_json(output / "predictions.json", result)
    write_json(output / "progress.json", {"status": "complete", "rows": len(rows), "total_rows": 3456})
    return result


def seal_predictions(experiment=study.EXPERIMENT):
    experiment = Path(experiment).resolve()
    study.verify_seal(experiment)
    manifest = read(experiment / "predictions/predictions.json")
    raw = experiment / "predictions/task_predictions.jsonl"
    if manifest.get("status") != "predicted_unscored" or manifest["task_predictions_sha256"] != digest(raw):
        raise RuntimeError("Figure 4 predictions are incomplete")
    seal = {
        "status": "sealed_before_scoring", "sealed_at_utc": datetime.now(UTC).isoformat(),
        "config_sha256": digest(experiment / "config.json"),
        "manifest_sha256": digest(experiment / "predictions/predictions.json"),
        "task_predictions_sha256": digest(raw), "rows": 3456,
    }
    write_json(experiment / "predictions/seal.json", seal)
    return seal


def verify_prediction_seal(experiment=study.EXPERIMENT):
    experiment = Path(experiment).resolve()
    seal = read(experiment / "predictions/seal.json")
    if seal.get("status") != "sealed_before_scoring" or seal["config_sha256"] != digest(experiment / "config.json"):
        raise ValueError("Figure 4 prediction seal is invalid")
    if seal["manifest_sha256"] != digest(experiment / "predictions/predictions.json"):
        raise ValueError("Figure 4 prediction manifest changed")
    if seal["task_predictions_sha256"] != digest(experiment / "predictions/task_predictions.jsonl"):
        raise ValueError("Figure 4 prediction rows changed")
    return seal


def _task_score(arguments):
    model, experiment = arguments
    rows = []
    for harness in study.HARNESSES:
        inputs = _inputs(experiment, "test", harness, model)
        exact = _exact(inputs)
        rollout = diagnostics._deterministic_rollouts(
            inputs, 32,
            stable_seed(BASE_SEED, "test-score-rollout", model.task.fingerprint, harness),
        )
        rows.append({
            "task_seed": model.task.seed, "family": model.task.family,
            "content_sha256": model.task.fingerprint, "harness": harness,
            "ceiling": exact["ceiling"], "M_exact": exact["mismatch"],
            "J_exact": exact["regularized_return"], "success_exact": exact["success_probability"],
            "discounted_task_reward": exact["discounted_task_reward"],
            "discounted_kl_penalty": exact["discounted_kl_penalty"],
            "success_rollout": rollout["mean_success"],
            "J_rollout": rollout["mean_regularized_return"], "rollout": rollout,
            "gap_identity_error": exact["gap_identity_error"],
            "return_decomposition_error": exact["return_decomposition_error"],
        })
    return rows


def score(experiment=study.EXPERIMENT, workers=4):
    experiment = Path(experiment).resolve()
    study.verify_seal(experiment)
    verify_prediction_seal(experiment)
    output = experiment / "scores"
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "task_scores.jsonl"
    task_models = study.models("test")
    rows, started = [], time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool, raw.open("w") as stream:
        for task_rows in pool.map(_task_score, [(model, experiment) for model in task_models]):
            for row in task_rows:
                stream.write(json.dumps(row) + "\n")
            rows.extend(task_rows)
            write_json(output / "progress.json", {
                "status": "running", "rows": len(rows), "total_rows": 3456,
                "updated_at_utc": datetime.now(UTC).isoformat(),
            })
    if len(rows) != 3456:
        raise AssertionError(f"Expected 3456 score rows, found {len(rows)}")
    result = {
        "status": "complete", "tasks": len(task_models), "task_rows": len(rows),
        "rollouts_per_task_harness": 32, "task_scores_sha256": digest(raw),
        "prediction_seal_sha256": digest(experiment / "predictions/seal.json"),
        "maximum_gap_identity_error": max(row["gap_identity_error"] for row in rows),
        "maximum_return_decomposition_error": max(row["return_decomposition_error"] for row in rows),
        "wall_seconds": time.monotonic() - started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(output / "scores.json", result)
    write_json(output / "progress.json", {"status": "complete", "rows": len(rows), "total_rows": 3456})
    return result


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _mean_ci(rows, key, seed):
    rng = np.random.default_rng(seed)
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    draws = np.mean(values[rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))], axis=1)
    return {"mean": float(values.mean()), "lower": float(np.quantile(draws, .025)), "upper": float(np.quantile(draws, .975))}


def _macro_ci(rows, key, seed):
    rng = np.random.default_rng(seed)
    groups = {family: [row for row in rows if row["family"] == family] for family in study.section51.FAMILIES}
    draws = np.empty(BOOTSTRAP_SAMPLES)
    for draw in range(BOOTSTRAP_SAMPLES):
        draws[draw] = np.mean([
            np.mean([group[index][key] for index in rng.integers(0, len(group), size=len(group))])
            for group in groups.values()
        ])
    mean = float(np.mean([np.mean([row[key] for row in group]) for group in groups.values()]))
    return {"mean": mean, "lower": float(np.quantile(draws, .025)), "upper": float(np.quantile(draws, .975))}


def _rank(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _correlation(left, right, rank=False):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if rank:
        left, right = _rank(left), _rank(right)
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def analyze(experiment=study.EXPERIMENT):
    experiment = Path(experiment).resolve()
    study.verify_seal(experiment)
    verify_prediction_seal(experiment)
    predictions = _jsonl(experiment / "predictions/task_predictions.jsonl")
    scores = _jsonl(experiment / "scores/task_scores.jsonl")
    prediction_map = {(row["task_seed"], row["harness"]): row for row in predictions}
    combined = []
    for score_row in scores:
        prediction = prediction_map[(score_row["task_seed"], score_row["harness"])]
        combined.append(score_row | {
            "M_hat": prediction["M_hat"], "J_pred": prediction["J_pred"],
            "actor_queries": prediction["actor_queries"],
            "J_error": prediction["J_pred"] - score_row["J_exact"],
            "M_error": prediction["M_hat"] - score_row["M_exact"],
            "ceiling_error": prediction["ceiling"] - score_row["J_exact"],
        })
    points = []
    for harness in study.HARNESSES:
        for family in study.section51.FAMILIES:
            rows = [row for row in combined if row["harness"] == harness and row["family"] == family]
            point = {"harness": harness, "family": family, "n_tasks": len(rows)}
            for key in ("ceiling", "M_hat", "M_exact", "J_pred", "J_exact", "success_rollout", "success_exact"):
                interval = _mean_ci(rows, key, stable_seed(BASE_SEED, "bootstrap", harness, family, key))
                point[key] = interval["mean"]
                point[f"{key}_ci"] = [interval["lower"], interval["upper"]]
            points.append(point)
    aggregate = []
    for harness in study.HARNESSES:
        rows = [row for row in combined if row["harness"] == harness]
        record = {"harness": harness, "n_tasks": len(rows), "family_weighting": "three families equally weighted"}
        for key in ("ceiling", "M_hat", "M_exact", "J_pred", "J_exact", "success_rollout", "success_exact"):
            interval = _macro_ci(rows, key, stable_seed(BASE_SEED, "bootstrap-aggregate", harness, key))
            record[key] = interval["mean"]
            record[f"{key}_ci"] = [interval["lower"], interval["upper"]]
        aggregate.append(record)

    def errors(rows, prediction_key):
        differences = np.asarray([row[prediction_key] - row["J_exact"] for row in rows])
        return {"MAE": float(np.mean(np.abs(differences))), "RMSE": float(np.sqrt(np.mean(differences**2)))}

    metrics = {
        "task_level": {
            "finite_mismatch": errors(combined, "J_pred"),
            "ceiling_only": errors(combined, "ceiling"),
            "M_bias": float(np.mean([row["M_error"] for row in combined])),
            "M_RMSE": float(np.sqrt(np.mean(np.square([row["M_error"] for row in combined])))),
        },
        "family_points": {
            "finite_mismatch": errors(points, "J_pred"), "ceiling_only": errors(points, "ceiling"),
            "pearson_predicted_observed_J": _correlation([p["J_pred"] for p in points], [p["J_exact"] for p in points]),
            "spearman_predicted_observed_J": _correlation([p["J_pred"] for p in points], [p["J_exact"] for p in points], rank=True),
            "pearson_predicted_J_success_rollout": _correlation([p["J_pred"] for p in points], [p["success_rollout"] for p in points]),
            "spearman_predicted_J_success_rollout": _correlation([p["J_pred"] for p in points], [p["success_rollout"] for p in points], rank=True),
        },
        "harness_aggregate": {
            "finite_mismatch": errors(aggregate, "J_pred"), "ceiling_only": errors(aggregate, "ceiling"),
            "pearson_predicted_observed_J": _correlation([p["J_pred"] for p in aggregate], [p["J_exact"] for p in aggregate]),
            "spearman_predicted_observed_J": _correlation([p["J_pred"] for p in aggregate], [p["J_exact"] for p in aggregate], rank=True),
            "pearson_predicted_J_success_rollout": _correlation([p["J_pred"] for p in aggregate], [p["success_rollout"] for p in aggregate]),
            "spearman_predicted_J_success_rollout": _correlation([p["J_pred"] for p in aggregate], [p["success_rollout"] for p in aggregate], rank=True),
        },
    }
    accounting = [read(study.cell_directory(experiment, split, harness) / "accounting.json")
                  for split in ("calibration", "test") for harness in study.HARNESSES]
    data = {
        "status": "complete",
        "source_model": study.proposal()["source_model"],
        "harnesses": list(study.HARNESSES), "tasks_per_family": 192,
        "estimator": {"m": M, "N": N, "alpha": ALPHA},
        "sources": {
            "config": str(experiment / "config.json"),
            "config_sha256": digest(experiment / "config.json"),
            "prediction_seal": str(experiment / "predictions/seal.json"),
            "prediction_seal_sha256": digest(experiment / "predictions/seal.json"),
            "task_predictions": str(experiment / "predictions/task_predictions.jsonl"),
            "task_predictions_sha256": digest(experiment / "predictions/task_predictions.jsonl"),
            "task_scores": str(experiment / "scores/task_scores.jsonl"),
            "task_scores_sha256": digest(experiment / "scores/task_scores.jsonl"),
        },
        "points": points, "aggregate": aggregate, "metrics": metrics,
        "resources": {
            "probability_cells": 12,
            "reused_probability_cells": sum(row["reused_from_v2"] for row in accounting),
            "new_gpu_probability_cells": sum(not row["reused_from_v2"] for row in accounting),
            "new_forward_state_rows": sum(row["forward_state_rows"] for row in accounting),
            "new_prompt_tokens": sum(row["prompt_tokens"] for row in accounting),
            "gpu_cell_wall_hours": sum(row["wall_seconds"] for row in accounting if not row["reused_from_v2"]) / 3600,
            "equivalent_test_actor_queries": int(sum(row["actor_queries"] for row in predictions)),
            "rollout_episodes": len(scores) * 32,
        },
        "statistics": {"bootstrap_samples": BOOTSTRAP_SAMPLES, "paired_unit": "task within family", "training_seed": 42},
    }
    output = experiment / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "figure4_harness_data.json", data)
    return data


def _fmt(value):
    return f"{value:.6f}"


def write_report(experiment=study.EXPERIMENT):
    experiment = Path(experiment).resolve()
    data = read(experiment / "analysis/figure4_harness_data.json")
    calibration = read(experiment / "calibration/summary.json")
    metrics = data["metrics"]
    lines = [
        "# Figure 4：固定 H0 模型在 H0–H5 原生 harness 下的预测实验", "",
        "## 实验状态", "",
        "实验完整完成。唯一 actor 为 E1 H0 update 1024，训练 seed 为 42；没有新增训练，也没有使用 H3-adapted 权重。任务严格复用 Measurement V2 的 72 个 calibration 和 576 个 IID test 实例。", "",
        "统一有限测量为 `m=256, N=1024, alpha=0.5`。六个 harness 各自使用原生动作、控制状态、提示、执行动力学和冻结基础模型 reference。H0/H3 概率表经过语义与哈希核验后复用，H1/H2/H4/H5 重新前向。", "",
        "## Calibration", "",
        "| Harness | macro bias | pool RMSE | task-pool RMSE | mean queries |", "|---|---:|---:|---:|---:|",
    ]
    for row in calibration["summary"]:
        lines.append(f"| {row['harness']} | {_fmt(row['macro_bias'])} | {_fmt(row['macro_RMSE_across_pools'])} | {_fmt(row['task_pool_RMSE'])} | {row['mean_actor_queries']:.1f} |")
    lines += ["", "该阶段不重新选择预算或平滑规则；全部结果均保留。", "", "## Test 总均值", "",
              "| Harness | C | M_hat | M exact | J predicted | J exact | rollout success | exact success |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in data["aggregate"]:
        lines.append("| {harness} | {ceiling:.6f} | {M_hat:.6f} | {M_exact:.6f} | {J_pred:.6f} | {J_exact:.6f} | {success_rollout:.6f} | {success_exact:.6f} |".format(**row))
    family = metrics["family_points"]
    aggregate = metrics["harness_aggregate"]
    task = metrics["task_level"]
    lines += [
        "", "## 预测精度与关联", "",
        f"- 18 个 harness×任务族点：有限 mismatch MAE/RMSE = `{family['finite_mismatch']['MAE']:.6f}` / `{family['finite_mismatch']['RMSE']:.6f}`；ceiling-only = `{family['ceiling_only']['MAE']:.6f}` / `{family['ceiling_only']['RMSE']:.6f}`。",
        f"- 6 个 harness 总均值：有限 mismatch MAE/RMSE = `{aggregate['finite_mismatch']['MAE']:.6f}` / `{aggregate['finite_mismatch']['RMSE']:.6f}`；ceiling-only = `{aggregate['ceiling_only']['MAE']:.6f}` / `{aggregate['ceiling_only']['RMSE']:.6f}`。",
        f"- 逐任务有限预测 MAE/RMSE = `{task['finite_mismatch']['MAE']:.6f}` / `{task['finite_mismatch']['RMSE']:.6f}`；绝对 M bias/RMSE = `{task['M_bias']:.6f}` / `{task['M_RMSE']:.6f}`。",
        f"- 18 点预测 J 与实际 J 的 Pearson/Spearman = `{family['pearson_predicted_observed_J']:.6f}` / `{family['spearman_predicted_observed_J']:.6f}`。",
        f"- 18 点预测 J 与独立 rollout success 的 Pearson/Spearman = `{family['pearson_predicted_J_success_rollout']:.6f}` / `{family['spearman_predicted_J_success_rollout']:.6f}`。该项仅为经验关联，没有拟合成功率预测器。",
        "", "## 资源与限制", "",
        f"- 概率单元 12 个，其中复用 {data['resources']['reused_probability_cells']} 个，新增 GPU 单元 {data['resources']['new_gpu_probability_cells']} 个。",
        f"- 新前向状态行 {data['resources']['new_forward_state_rows']:,}，prompt tokens {data['resources']['new_prompt_tokens']:,}，GPU 单元累计 wall time {data['resources']['gpu_cell_wall_hours']:.2f} 小时。",
        f"- test 有限测量等价 actor queries {data['resources']['equivalent_test_actor_queries']:,}；独立 rollout episodes {data['resources']['rollout_episodes']:,}。缓存采样不等同于实际发出了同等数量的在线 API 请求。",
        "- 结果条件于单一训练 seed 42；bootstrap 只刻画任务实例抽样不确定性。已知 oracle ceiling 和被查询状态的最优策略仍是本受控实验的前提。",
        "", "## 可追溯产物", "",
        f"- 汇总数据：`{experiment / 'analysis/figure4_harness_data.json'}`",
        f"- 封存预测：`{experiment / 'predictions/task_predictions.jsonl'}`",
        f"- 独立评分：`{experiment / 'scores/task_scores.jsonl'}`",
    ]
    path = study.DOCS / "REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def final_validate(experiment=study.EXPERIMENT):
    experiment = Path(experiment).resolve()
    config = study.verify_seal(experiment)
    seal = verify_prediction_seal(experiment)
    study.audit_tables(experiment, "test")
    data_path = experiment / "analysis/figure4_harness_data.json"
    report_path = study.DOCS / "REPORT.md"
    required = [data_path, report_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Figure 4 final artifacts missing: " + ", ".join(missing))
    data = read(data_path)
    if data.get("status") != "complete" or len(data.get("points", [])) != 18 or len(data.get("aggregate", [])) != 6:
        raise AssertionError("Figure 4 analysis data is incomplete")
    numbers = []
    for row in data["points"] + data["aggregate"]:
        numbers += [value for value in row.values() if isinstance(value, float)]
    if not all(math.isfinite(value) for value in numbers):
        raise AssertionError("Figure 4 data contains nonfinite values")
    predictions = _jsonl(experiment / "predictions/task_predictions.jsonl")
    scores = _jsonl(experiment / "scores/task_scores.jsonl")
    score_manifest = read(experiment / "scores/scores.json")
    validation = {
        "status": "passed", "validated_at_utc": datetime.now(UTC).isoformat(),
        "config_sha256": digest(experiment / "config.json"),
        "source_hashes_match_seal": study.source_hashes() == config["source_sha256"],
        "prediction_seal_sha256": digest(experiment / "predictions/seal.json"),
        "prediction_seal_status": seal["status"],
        "row_counts": {"predictions": len(predictions), "scores": len(scores), "family_points": 18, "aggregate_points": 6},
        "maximum_gap_identity_error": score_manifest["maximum_gap_identity_error"],
        "maximum_return_decomposition_error": score_manifest["maximum_return_decomposition_error"],
        "report_sha256": digest(report_path),
        "training_seed_statement": "frozen E1 H0 update 1024; source training seed 42 only",
    }
    output = experiment / "analysis"
    write_json(output / "final_validation.json", validation)
    write_json(output / "completion.json", {
        "status": "complete", "validated_at_utc": validation["validated_at_utc"],
        "config_sha256": validation["config_sha256"],
        "prediction_seal_sha256": validation["prediction_seal_sha256"],
        "report_sha256": validation["report_sha256"],
    })
    write_json(experiment / "progress.json", {"status": "complete", "completed_cells": 12, "total_cells": 12, "validated_at_utc": validation["validated_at_utc"]})
    write_json(experiment / "MONITOR.json", {
        "task_id": "figure4_h0_h5_prediction_2026-09-17", "status": "complete",
        "monitor_interval_seconds": 600, "monitor_stopped_at_utc": validation["validated_at_utc"],
        "completed_cells": 12, "total_cells": 12, "processes": [], "failures": [],
    })
    (study.DOCS / "VALIDATION.md").write_text(
        "# Figure 4 最终验收\n\n"
        f"- 状态：`{validation['status']}`\n"
        f"- 配置 SHA-256：`{validation['config_sha256']}`\n"
        f"- 预测封存 SHA-256：`{validation['prediction_seal_sha256']}`\n"
        f"- 预测／评分行：{len(predictions)}/{len(scores)}\n"
        f"- 18 个族内点、6 个总均值点均为有限实测结果。\n"
        f"- 最大 gap／return 分解残差：{validation['maximum_gap_identity_error']:.3e}/{validation['maximum_return_decomposition_error']:.3e}\n"
        "- checkpoint、任务、原生动作语义、概率表、预测封存和独立评分均已核验。\n"
    )
    return validation


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("calibrate", "predict", "seal-predictions", "score", "analyze", "report", "validate"))
    parser.add_argument("--experiment", type=Path, default=study.EXPERIMENT)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "calibrate":
        print(json.dumps(calibrate(args.experiment, args.workers), indent=2))
    elif args.command == "predict":
        print(json.dumps(predict(args.experiment, args.workers), indent=2))
    elif args.command == "seal-predictions":
        print(json.dumps(seal_predictions(args.experiment), indent=2))
    elif args.command == "score":
        print(json.dumps(score(args.experiment, args.workers), indent=2))
    elif args.command == "analyze":
        result = analyze(args.experiment)
        print(json.dumps({"status": result["status"], "metrics": result["metrics"]}, indent=2))
    elif args.command == "report":
        print(write_report(args.experiment))
    else:
        print(json.dumps(final_validate(args.experiment), indent=2))


if __name__ == "__main__":
    main()
