"""Generate the complete local report for the frozen readout study."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_readout as study
from hrp_miniwob.benchmarks.miniwob_meta_harness import digest


def fmt(item, scale=1.0):
    return f"{scale * item['mean']:.6f} [{scale * item['lower']:.6f}, {scale * item['upper']:.6f}]"


def report(experiment=study.EXPERIMENT, docs=study.DOCS):
    experiment, docs = Path(experiment).resolve(), Path(docs).resolve()
    data_path = experiment / "analysis/readout_data.json"
    data = json.loads(data_path.read_text())
    if data["status"] != "complete":
        raise RuntimeError("Readout analysis is incomplete")
    docs.mkdir(parents=True, exist_ok=True)
    test = data["splits"]["test"]
    labels = {
        "base": "Qwen3.5-2B base",
        "h0_final": "E1 H0 final",
        "h3_adapt_final": "Figure 3 H3-adapt final",
    }
    lines = [
        "# 第五章：冻结模型的 readout 与呈现等价性实验",
        "",
        f"生成时间：{datetime.now(UTC).isoformat()}。主结果来自预先冻结的 576 个 test 任务，calibration 仅用于实现验收。",
        "",
        "## 实验范围",
        "",
        "本实验冻结三个模型条件，在 H0 的同一个语义控制图上只改变模型看到的 JSON 与单 token 动作 code。主对照把基础模型在规范 R0 下的语义概率固定为共同 reference `q0`，并传输到所有呈现，因此 P、reward、原生起点、预算、reference 与最优策略保持一致。actor 正常读取各自 prompt，没有 prior 或 logit 校正。",
        "",
        "- 任务族：`click-checkboxes`、`click-checkboxes-soft`、`ascending-numbers`，每族 calibration 24、test 192。",
        "- 模型：原始 Qwen3.5-2B、E1 H0 update 1024、从该 H0 checkpoint 在 H3 适应到 update 1024。后两个训练过程都使用 seed 42；三个模型条件不是三个训练 seed。",
        "- 控制参数：horizon 10、gamma 0.95、tau 0.1、显式成本 0、完整菜单 temperature 1、BF16 前向与 FP64 控制计算。",
        "- 区间：族内以任务为配对单位 bootstrap 2,000 次，再对三个任务族等权平均；区间不含训练 seed 不确定性。",
        "",
        "## 主对照结果",
        "",
        "表中 readout RMS 是固定完整可达状态上的 `sqrt(MSE(eta))`。ΔJ 与 Δsuccess 均为同一模型、同一任务上的 Rv−R0 配对差值。",
        "",
        "| 冻结模型 | 呈现 | target RMS | readout RMS | semantic TV | argmax 改变率 | ΔJ [95% CI] | Δsuccess [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id, label in labels.items():
        rows = test["models"][model_id]["main"]
        for rendering in study.RENDERINGS:
            row = rows[rendering]
            target_rms = row["target_mse"]["mean"] ** .5
            readout_rms = row["readout_mse"]["mean"] ** .5
            lines.append(
                f"| {label} | {rendering} | {target_rms:.6g} | {readout_rms:.6f} | "
                f"{row['semantic_tv']['mean']:.6f} | {100 * row['argmax_change_rate']['mean']:.3f}% | "
                f"{fmt(row['delta_regularized_return_vs_R0'])} | {fmt(row['delta_success_probability_vs_R0'], 100)} pp |"
            )

    lines += ["", "## 精确恒等式与等价性验收", "",
              "以下是所有 test 任务和 R0–R5 上的最坏误差。目标阈值为 1e-10；这些数值是实现一致性检查，不作为经验独立证据。", "",
              "| 模型 | ΔC 最大值 | 最优策略最大差 | gap 恒等式 | ΔJ+ΔM | e0+eta+g | correspondence pairwise 违反量 | L2 违反量 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model_id, label in labels.items():
        value = test["models"][model_id]["validation"]
        lines.append(
            f"| {label} | {value['max_target_equivalence_error']:.3e} | {value['max_optimal_policy_equivalence_error']:.3e} | "
            f"{value['max_gap_identity_error']:.3e} | {value['max_delta_j_plus_delta_m_error']:.3e} | "
            f"{value['max_deployment_identity_error']:.3e} | {value['max_pairwise_bound_violation']:.3e} | {value['max_l2_bound_violation']:.3e} |"
        )

    lines += ["", "共同 reference 主对照中的 target drift 为数值零；因此观察到的语义策略变化来自冻结模型对呈现的敏感性。回报变化由 drift 的方向及其改变的 occupancy 共同决定，readout MSE 大并不推出 J 必然下降。`M` 由被评测策略的 occupancy 与 `KL(pi || pi*)` 独立计算，没有使用 `C-J` 反推。", "",
              "## 数值重复与选择后改名", "", "| 模型 | 重复状态/次 | 五次前向最大 log-prob 差 | 最大 semantic TV | 最大 centered readout 差 | 选择后改名最大指标误差 |",
              "|---|---:|---:|---:|---:|---:|"]
    for model_id, label in labels.items():
        repeated = data["repeat_controls"][model_id]
        validation = test["models"][model_id]["validation"]
        lines.append(f"| {label} | {repeated['states_per_repeat']} | {repeated['maxima']['log_probability']:.3e} | {repeated['maxima']['semantic_tv']:.3e} | {repeated['maxima']['centered_readout']:.3e} | {validation['max_rename_metric_error']:.3e} |")
    lines += ["", "重复前向是 eval mode、dropout 关闭、相同批次和 padding 下的数值基线，不是动作采样误差或训练 seed 方差。选择后改名在 R0 选择完成后做确定性双射及逆解码，语义策略、target、J、M 与 success 应完全不变。CPU 验收中的错误列对齐用例必须失败；实际 test 的错误对齐检测率也记录在机器可读结果中。", "",
              "## 只改变目标的 oracle 阳性控制", "", "| 模型 | target MSE [95% CI] | readout MSE | max |g| [95% CI] | 末次决策反转误差上界 | J | M |", "|---|---:|---:|---:|---:|---:|---:|"]
    for model_id, label in labels.items():
        row = test["models"][model_id]["positive_control"]
        lines.append(f"| {label} | {fmt(row['target_mse'])} | {row['readout_mse']['mean']:.3e} | {fmt(row['g_max_abs'])} | {row['last_step_reversal_max_error']['upper']:.3e} | {fmt(row['regularized_return'])} | {fmt(row['mismatch'])} |")
    lines += ["", "该控制复用 R0 actor 概率和 prompt，只在剩余一次决策的状态改变 reference。它应产生非零 g、零 eta，并反转所有非平坦末次决策状态的最优 centered energy；它只验收读数能区分目标变化与行为变化。", "",
              "## Reference 随呈现变化的附加分析", "", "该分析把基础模型在各 Rv 下已经保存的语义分布用作自然 reference `q_v`，只做 CPU 重求解。它同时允许 target 与 actor readout 改变，不混入 g=0 主对照。", "",
              "| 冻结模型 | 呈现 | target RMS vs R0 | readout RMS vs R0 | C | J | M | success |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model_id, label in labels.items():
        rows = test["models"][model_id]["natural_reference"]
        for rendering in study.RENDERINGS:
            row = rows[rendering]
            lines.append(f"| {label} | {rendering} | {row['target_mse_vs_R0']['mean'] ** .5:.6f} | {row['readout_mse_vs_R0']['mean'] ** .5:.6f} | {row['ceiling']['mean']:.6f} | {row['regularized_return']['mean']:.6f} | {row['mismatch']['mean']:.6f} | {row['success_probability']['mean']:.6f} |")

    accounting = {}
    for split in study.SPLIT_QUOTAS:
        for model_id in study.MODEL_SPECS:
            for rendering in study.RENDERINGS:
                path = experiment / "evaluations" / split / model_id / rendering / "accounting.json"
                accounting[f"{split}/{model_id}/{rendering}"] = json.loads(path.read_text())
    lines += ["", "## 资源与可追溯产物", "", f"36 个主评测单元全部完成，其中 test 单元 18 个；原始任务概率表保存在 `{experiment / 'evaluations'}`。每个单元包含 identity、逐任务概率与 prompt hash、资源计数和完成哈希。", "",
              f"- 固定配置：`{experiment / 'config.json'}`", f"- 数据及排除清单：`{experiment / 'suite/manifest.json'}`、`{experiment / 'exclusions.json'}`", f"- CPU/GPU 验收：`{experiment / 'acceptance'}`", f"- 聚合数据：`{experiment / 'analysis/readout_data.json'}`", f"- 逐任务主指标：`{experiment / 'analysis/task_rows.jsonl'}`", f"- 含自然 reference 与控制的逐任务记录：`{experiment / 'analysis/task_details.jsonl'}`",  "",
              "## 证据边界", "", "实验只覆盖三个受控 MiniWoB++ 任务族和一个训练/适应 seed。完整单 token 菜单使 parser validity 成为结构性结果；它不测自由文本 parser 的鲁棒性。correspondence 界只覆盖已枚举菜单 token，不外推到完整词表或任意长输出。本轮没有有限 probe、独立回报预测、真实开放 agent 干预，也没有新增训练。"]
    report_path = docs / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n")
    write_json(docs / "accounting.json", accounting)
    write_json(docs / "manifest.json", {
        "status": "complete",
        "report_sha256": digest(report_path),
        "data_sha256": digest(data_path),
        "accounting_sha256": digest(docs / "accounting.json"),
    })
    print(report_path)


if __name__ == "__main__":
    report()
