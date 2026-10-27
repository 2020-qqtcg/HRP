"""E1 report from recorded finite-harness runs; no inferred pending results."""

import csv
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from hrp_miniwob.harnesses.miniwob_harnesses import PROFILES
from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks.miniwob_v2_report import paired_interval, read_json

ROOT = Path(__file__).resolve().parents[3]
KEYS = ("success_probability", "fixed_bank_energy_mse", "regularized_return", "ceiling", "gap", "mixed_J", "mixed_ceiling")


def prompt_target_comparison(experiment, recorded):
    """H0/H5 share complete state/action spaces, with legacy paths kept intact."""
    pair = ("H0", "H5")
    path = experiment / f"diagnostics/{recorded['H0']}_{recorded['H5']}_targets.json"
    if path.exists():
        result = read_json(path)
        return result | {"native_ceilings": {h: result["native_ceilings"][recorded[h]] for h in pair}}
    directories = [experiment / "runs" / f"exactpg_{recorded[h]}/checkpoint_000000/calibration" for h in pair]
    reports = [read_json(d / "metrics.json") for d in directories]
    if not all(reports):
        return None
    from hrp_miniwob.oracle import solve
    by_family = defaultdict(list)
    for row in reports[0]["rows"]:
        key = f"{row['family']}_{row['task_seed']}"
        # The task key is recorded by the policy filenames, independently of display names.
        matches = list((directories[0] / "policies").glob(f"*{row['task_seed']}.npz"))
        if len(matches) != 1:
            raise AssertionError(f"Ambiguous initial policy table: {key}")
        name = matches[0].name
        with np.load(directories[0] / "policies" / name) as first, np.load(directories[1] / "policies" / name) as second:
            for field in ("actions", "next_state", "rewards", "fixed_bank", "initial_distribution"):
                if not np.array_equal(first[field], second[field]):
                    raise AssertionError(f"H0/H5 supposedly shared {field} differs for {name}")
            optima = [solve(saved["next_state"], saved["rewards"], saved["reference"], .95, .1)
                      for saved in (first, second)]
            delta = .1 * (optima[0].log_policy - optima[1].log_policy)[first["fixed_bank"]]
            delta -= delta.mean(axis=1, keepdims=True)
            by_family[row["family"]].append(float(np.square(delta).mean()))
    result = {"comparison": "Same complete actions, dynamics, rewards, reachable bank and coverage; different system prompt and frozen-reference readout",
              "tasks": sum(map(len, by_family.values())),
              "optimal_centered_energy_mse": float(np.mean([np.mean(values) for values in by_family.values()])),
              "by_family": {family: float(np.mean(values)) for family, values in by_family.items()},
              "native_ceilings": {h: r["metrics"]["ceiling"] for h, r in zip(pair, reports, strict=True)}}
    path.parent.mkdir(exist_ok=True)
    write_json(path, result | {"native_ceilings": {recorded[h]: result["native_ceilings"][h] for h in pair}})
    return result


def write_report(experiment):
    experiment = Path(experiment).resolve()
    docs = ROOT / "environment/docs/v2" / experiment.name
    docs.mkdir(parents=True, exist_ok=True)
    pipeline = read_json(experiment / "pipeline.json") or {"status": "preparing", "jobs": {}}
    config = read_json(experiment / "config.json") or {}
    numbering = read_json(experiment / "harness_numbering.json")
    recorded = numbering["current_to_recorded"] if numbering else {h: h for h in config.get("harnesses", list(PROFILES))}
    harnesses = list(recorded)
    lines = ["# 多 harness Exact-PG：E1 实验报告", "",
             f"更新：{datetime.now(UTC).isoformat()}；流水线：{pipeline['status']}。", "",
             f"本轮实验接口：{', '.join(harnesses)}（{len(harnesses)} 组）。", "",
             ("本文使用连续新编号：H4 为原 H5（LM reviewer），H5 为原 H7（system prompt）。"
              "原 H4 计划执行器已取消。历史路径及 PLAN.md 保留旧编号，映射见 harness_numbering.json。" if numbering else ""), "",
             f"设置及判据见 [PLAN.md]({docs / 'PLAN.md'})。Qwen3.5-2B，seed 42，BF16，LoRA rank 16，恒定学习率 1e-5；"
             "Exact-PG batch 4 个任务、microbatch 8 个状态、1024 次更新，每 64 次完整评测。"
             "起点为 50% 原生＋50% 合法中间状态。所有新接口启用 non-reentrant 激活重计算以容纳长菜单；"
             "BF16 梯度的重复数值误差另见 GPU 验收记录。", "",
             "## 当前进度与校准结果", "",
             "所有准确率为原生起点成功概率，三任务族等权；MSE 使用本接口固定的全部合法可达状态。初始列均为该接口 update 0。", "",
             "| Harness | 状态 / 更新 | 完整评测步 | 初始准确率 | 初始 MSE | 初始 J | 当前准确率 | 当前 MSE | 当前 J | C |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    series, curves, runs = {}, [], {}
    for harness in harnesses:
        arm = f"exactpg_{recorded[harness]}"
        directory = experiment / "runs" / arm
        run = read_json(directory / "run.json") or {}
        runs[harness] = run
        state = pipeline["jobs"].get(f"train_{arm}", {}).get("status", "等待集成检查")
        entries = []
        for checkpoint in run.get("checkpoints", []):
            if "calibration" not in checkpoint:
                continue
            reports = {split: read_json(directory / checkpoint["directory"] / split / "metrics.json")
                       for split in ("calibration", "train_probe")}
            if not all(reports.values()):
                continue
            entries.append({"updates": checkpoint["updates"], **reports})
            for split, result in reports.items():
                curves.append({"harness": harness, "split": split, "updates": checkpoint["updates"],
                               "task_draws": checkpoint["task_draws"], **{key: result["metrics"][key] for key in KEYS}})
        series[harness] = entries
        updates = run.get("completed_updates", 0)
        if entries:
            first, last = [entry["calibration"]["metrics"] for entry in (entries[0], entries[-1])]
            lines.append(f"| {harness} | {state} / {updates} | {entries[-1]['updates']} | {100*first['success_probability']:.3f}% | "
                         f"{first['fixed_bank_energy_mse']:.6f} | {first['regularized_return']:.6f} | {100*last['success_probability']:.3f}% | "
                         f"{last['fixed_bank_energy_mse']:.6f} | {last['regularized_return']:.6f} | {last['ceiling']:.6f} |")
        else:
            progress = None
            if run.get("checkpoints"):
                progress = read_json(directory / run["checkpoints"][-1]["directory"] / "calibration/progress.json")
            detail = f"；初始化校准 {progress['tasks']}/{progress['total']}" if progress else ""
            lines.append(f"| {harness} | {state} / {updates}{detail} | — | — | — | — | — | — | — | — |")
    if curves:
        with (docs / "learning_curves.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
            writer.writeheader()
            writer.writerows(curves)
    lines += ["", "## 相对初始化的几何变化", "",
              "以任务为独立单位、族内配对 bootstrap 2000 次；这是任务抽样区间，不是训练 seed 间区间。", "",
              "| Harness | 完整评测步 | 校准 ΔMSE [95% CI] | 训练探针 ΔMSE [95% CI] |",
              "| --- | ---: | --- | --- |"]
    for harness, entries in series.items():
        if len(entries) < 2:
            continue
        values = [paired_interval(entries[-1][split], entries[0][split], experiment) for split in ("calibration", "train_probe")]
        formatted = [f"{delta:+.6f} [{lo:+.6f}, {hi:+.6f}]" for delta, lo, hi in values]
        lines.append(f"| {harness} | {entries[-1]['updates']} | {' | '.join(formatted)} |")
    comparison = prompt_target_comparison(experiment, recorded) if {"H0", "H5"} <= recorded.keys() else None
    if comparison:
        lines += ["", f"H0/H5 已核对相同完整状态、动作、P、r 和起点分布。DP 最优相对能量之间的 MSE 为 "
                  f"{comparison['optimal_centered_energy_mse']:.8f}；原生 ceiling：{comparison['native_ceilings']}。"
                  "这是目标差异测量，不是训练已拟合目标的证据。"]
    lines += ["", "## 校准选点后的留出评测", "",
              "只按校准原生准确率最高选点，平局取最早，初始化也参与。IID/OOD 不参与选择。", "",
              "| Harness | 所选步 | Split | 准确率 | MSE | J | C |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for harness in harnesses:
        choice = read_json(experiment / "runs" / f"exactpg_{recorded[harness]}" / "e1_selection.json")
        if not choice:
            continue
        for split in ("test", "ood"):
            result = read_json(experiment / "selected_eval" / f"exactpg_{recorded[harness]}" / split / "metrics.json")
            if result:
                m = result["metrics"]
                lines.append(f"| {harness} | {choice['updates']} | {split} | {100*m['success_probability']:.3f}% | "
                             f"{m['fixed_bank_energy_mse']:.6f} | {m['regularized_return']:.6f} | {m['ceiling']:.6f} |")
    lines += ["", "## 训练接口 × 部署接口", "",
              "交叉矩阵使用统一 update 1024，另含未训练模型。每列按部署接口重新读取完整菜单并使用该接口的 frozen reference/DP 目标。"
              "不同菜单不能直接按相同代码字母比较动作能量；矩阵中的每一格均对齐其部署接口。", ""]
    for split in ("test", "ood"):
        matrices = {h: read_json(experiment / "cross_eval" / f"exactpg_{split}" / recorded[h] / "matrix.json") for h in harnesses}
        if not any(matrices.values()):
            lines += [f"{split}：等待交叉评测。", ""]
            continue
        for key in KEYS[:4]:
            lines += [f"{split} / {key}", "", "| 训练接口 | " + " | ".join(harnesses) + " |",
                      "| --- | " + " | ".join(["---:"] * len(harnesses)) + " |"]
            for i, label in enumerate(("initial", *harnesses)):
                row = []
                for h in harnesses:
                    values = (matrices[h] or {}).get("rows", [])
                    # Resolve the recorded training interface from the checkpoint,
                    # since retired columns and changed labels invalidate indices.
                    match = next((r for r in values if
                                  (r["checkpoint"] is None if i == 0 else
                                   r["checkpoint"] is not None and
                                   Path(r["checkpoint"]).parents[2].name == f"exactpg_{recorded[label]}")), None)
                    row.append(f"{match[key]:.6f}" if match else "—")
                lines.append(f"| {label} | " + " | ".join(row) + " |")
            lines.append("")
    lines += ["## 成本与结论边界", "",
              "| Harness | 已更新 | 任务抽取数 | 训练阶段小时 | actor 前向状态数 | actor 梯度状态数 | reference 前向状态数 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for harness, run in runs.items():
        costs = run.get("accounting", {})
        lines.append(f"| {harness} | {run.get('completed_updates', 0)} | {run.get('task_draws', 0)} | "
                     f"{costs.get('wall_seconds', 0)/3600:.3f} | {costs.get('actor_forwards', 0)} | "
                     f"{costs.get('actor_gradient_observations', 0)} | {costs.get('reference_forwards', 0)} |")
    lines += ["", "训练阶段成本包含当步缺失 reference 的计算，排除 checkpoint 校准/探针评测；总进程起止时间另见 pipeline.json/run.json。"
              "相同更新数与任务数不等于相同计算量，不同控制器会改变合法状态数。", "",
              "本轮为 seed 42 的受控扩展，复用既有三任务内容划分；不是全新的盲测，也不能估计多 seed 训练稳定性。"
              "只有已记录的训练/校准固定状态 MSE、J/C/gap 和交叉评测共同支持时，才陈述几何向各自目标靠近；准确率上升本身不足够。"
              "不保证所有状态误差逐点下降，也不把有限预算、LoRA 下的残差称为达到全局最优。", ""]
    path = docs / "REPORT.md"
    path.write_text("\n".join(lines))
    return path
