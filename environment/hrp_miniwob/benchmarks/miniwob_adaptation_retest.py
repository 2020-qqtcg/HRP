"""Supplementary adaptation evaluation on the retained E1 IID task cohort."""

import argparse
import csv
import fcntl
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from hrp_miniwob.benchmarks import miniwob_adaptation as study
from hrp_miniwob.benchmarks.miniwob_adaptation_report import cells, evaluation_path
from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.v2 import FAMILIES
from hrp_miniwob.v2.suite import load_suite

NAME = 'old_e1_iid'


def prepare(experiment):
    output = experiment / NAME
    output.mkdir(parents=True, exist_ok=True)
    models = load_suite(study.SOURCE / 'suite', 'test')
    assert len(models) == len({m.task.fingerprint for m in models}) == 576
    assert all(sum(m.task.family == f for m in models) == 192 for f in FAMILIES)
    fresh = {m.task.fingerprint for m in load_suite(experiment / 'suite', 'test')}
    assert not fresh & {m.task.fingerprint for m in models}
    source = study.source_record()
    baselines = {}
    for harness in study.HARNESSES:
        directory = study.SOURCE / f'cross_eval/exactpg_test/{harness}'
        matrix_path = directory / 'matrix.json'
        assert study.digest(matrix_path) == study.read(directory / 'completion.json')['matrix_sha256']
        matrix = study.read(matrix_path)
        assert matrix['rows'][1]['checkpoint'] == source['adapter']
        saved = directory / 'adapted_0/harness_0'
        baselines[harness] = {'directory': str(saved), 'matrix_sha256': study.digest(matrix_path),
                              'metrics_sha256': study.digest(saved / 'metrics.json'),
                              'policies_sha256': {m.task.key: study.digest(saved / f'policies/{m.task.key}.npz') for m in models}}
    config = {'parent_config_sha256': study.digest(experiment / 'config.json'),
              'suite': str(study.SOURCE / 'suite'), 'suite_manifest_sha256': study.digest(study.SOURCE / 'suite/manifest.json'),
              'task_seeds': [m.task.seed for m in models], 'task_content_sha256': [m.task.fingerprint for m in models],
              'split': 'test', 'tasks': 576, 'seed': 42, 'checkpoint_updates': study.NODES,
              'harnesses': study.HARNESSES, 'source': source, 'baseline_records': baselines,
              'evaluation_code_sha256': study.digest(Path(__file__)),
              'measurement_sources_sha256': study.read(experiment / 'config.json')['source_sha256_at_preparation'],
              'scope': 'Supplementary retained IID cohort, already used to select H0 to H3; fresh test remains primary',
              'selection': 'All scheduled checkpoints; primary final budget unchanged; no training or checkpoint selection on this cohort',
              'physical_gpus': [1, 3], 'total_test_cells': 26, 'external_thread_messages': False}
    path = output / 'config.json'
    if path.exists():
        assert study.read(path) == config
    else:
        write_json(path, config)
        write_json(output / 'manifest.json', {'source': source, 'status': 'prepared', 'created_at_utc': datetime.now(UTC).isoformat(),
                   'config_sha256': study.digest(path), 'cohort': NAME, 'tasks': 576, 'primary_result': False})
    return config


def job_specs(experiment):
    specs = {}
    root = experiment / NAME
    source = study.read(experiment / 'proposal.json')['source']['adapter']
    for branch, update, deploy in cells():
        adapter = source if update == 0 else str(experiment / f'runs/seed42_{branch}/checkpoint_{update:06d}/actor/adapter')
        output = evaluation_path(root, branch, update, deploy)
        specs[f'legacy_{branch}_{update:06d}_{deploy}'] = {
            'kind': 'evaluate', 'branch': branch, 'update': update, 'deployment': deploy,
            'cohort': NAME, 'output': str(output),
            'command': [sys.executable, '-m', 'hrp_miniwob.benchmarks.miniwob_adaptation_retest', 'evaluate',
                        '--experiment', str(experiment), '--harness', deploy, '--adapter', adapter, '--output', str(output)]}
    return specs


def evaluate(experiment, harness, adapter_path, output):
    root = experiment / NAME
    config = study.read(root / 'config.json')
    assert study.digest(Path(__file__)) == config['evaluation_code_sha256']
    assert study.digest(experiment / 'config.json') == config['parent_config_sha256']
    assert study.digest(Path(config['suite']) / 'manifest.json') == config['suite_manifest_sha256']
    models = load_suite(Path(config['suite']), 'test')
    assert [m.task.seed for m in models] == config['task_seeds']
    with study.bindings():
        host = study.study.base.runner
        parent = host.read_config(experiment)
        identity = {'cohort': NAME, 'supplement_config_sha256': study.digest(root / 'config.json'),
                    'harness': harness, 'adapter': str(adapter_path), 'adapter_sha256': study.model_files(adapter_path),
                    'source_suite_sha256': config['suite_manifest_sha256'], 'split': 'test'}
        output.mkdir(parents=True, exist_ok=True)
        with (output / '.run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (output / 'evaluation.json').exists():
                assert study.read(output / 'evaluation.json') == identity
            write_json(output / 'evaluation.json', identity)
            if (output / 'completion.json').exists():
                assert study.read(output / 'completion.json')['metrics_sha256'] == study.digest(output / 'metrics.json')
                return
            started = time.monotonic()
            actor = host.LanguagePolicy(parent['actor_model'], adapter_path=str(adapter_path), trainable=False, device='cuda:0', rank=16)
            frozen = host.load_frozen(parent, 'cuda:0')
            bound = host.adapter_for(actor, host.Candidate(study.study.candidate_path(harness)), frozen, parent, experiment)
            try:
                result = study.evaluate_models(models, bound, output)
                audit = {}
                baseline = config['baseline_records'][harness]
                maximum_reference_error = maximum_actor_error = 0.
                for model in models:
                    old_path = Path(baseline['directory']) / f'policies/{model.task.key}.npz'
                    assert study.digest(old_path) == baseline['policies_sha256'][model.task.key]
                    new_path = output / f'tasks/{model.task.key}/policies/{model.task.key}.npz'
                    with np.load(old_path) as old, np.load(new_path) as new:
                        for key in ('next_state', 'rewards', 'fixed_bank'):
                            np.testing.assert_array_equal(old[key], new[key])
                        maximum_reference_error = max(maximum_reference_error, float(np.abs(np.log(old['reference']) - np.log(new['reference'])).max()))
                        if str(adapter_path) == config['source']['adapter']:
                            maximum_actor_error = max(maximum_actor_error, float(np.abs(old['log_policy'] - new['log_policy']).max()))
                assert maximum_reference_error <= 1e-3
                audit.update(max_reference_log_error_vs_E1=maximum_reference_error, task_graphs_identical=576)
                if str(adapter_path) == config['source']['adapter']:
                    assert maximum_actor_error <= 1e-3
                    old_metrics = study.read(Path(baseline['directory']) / 'metrics.json')['metrics']
                    differences = {key: result['metrics'][key] - old_metrics[key] for key in ('ceiling', 'regularized_return', 'kl_mismatch', 'success_probability', 'fixed_bank_energy_mse')}
                    assert max(abs(d) for d in differences.values()) < 1e-6
                    audit.update(max_actor_log_error_vs_E1=maximum_actor_error, aggregate_differences_vs_E1=differences)
                write_json(output / 'replay_audit.json', audit)
                write_json(output / 'completion.json', {'metrics_sha256': study.digest(output / 'metrics.json'),
                           'replay_audit_sha256': study.digest(output / 'replay_audit.json'), 'tasks': result['tasks'],
                           'wall_seconds': time.monotonic() - started, 'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES')})
            finally:
                bound.close()


def report(experiment):
    # The revised shared analysis supplies identical paired bootstrap rules to
    # both cohorts. Import here so finalization can install the reporting revision.
    from hrp_miniwob.benchmarks import miniwob_adaptation_report as analysis
    root = experiment / NAME
    data, rows = analysis.analyze(root)
    config = study.read(root / 'config.json')
    data.update(cohort=NAME, primary_result=False, parent_experiment=str(experiment),
                suite_manifest_sha256=config['suite_manifest_sha256'], claim_scope=config['scope'])
    if rows:
        with (root / 'paired_task_results.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    family_rows = []
    for key, value in data['cells'].items():
        branch, update, deployment = key.split(':')
        family_rows.extend({'training_interface': branch, 'additional_updates': int(update), 'deployment_interface': deployment,
                            'family': f, **metrics} for f, metrics in value['per_family'].items())
    if family_rows:
        with (root / 'per_family_results.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(family_rows[0])); writer.writeheader(); writer.writerows(family_rows)
    write_json(root / 'figure3_data.json', data)
    docs = analysis.DOCS / NAME
    docs.mkdir(parents=True, exist_ok=True)
    lines = ['# 旧 E1 IID 576 任务补充评测', '', f"状态：{data['status']}，完成 {data['completed_cells']}/26 单元。", '',
             '复用 Figure 3 同一源 checkpoint、两条单 seed 42 训练分支和所有既定评测节点，没有新增训练。旧 IID 已参与 H0→H3 候选选择，本结果属于补充复测，不替代新留出 test 的主比较。两个数据集分别按三任务族等权聚合，族内以任务为配对单位 bootstrap 2000 次，seed=4203。', '',
             '每个 checkpoint 的转移核、reward、固定状态库与旧 E1 概率表逐任务核对，reference 概率也逐表核对。冻结源 checkpoint 额外核对完整动作概率及旧聚合指标。复测审计见各 evaluation 的 replay_audit.json。', '',
             '## 同一接口在两个评测任务集上的 ceiling', '',
             'ceiling 依赖任务起点分布。这里比较的是同一接口在两个不同有限任务集合上的均值，不能把数据集间的变化归因于适应训练。', '',
             '| 数据集 | C(H0) | C(H3) | C(H3)−C(H0) [95% CI] | T：恢复原始搭配的 J |', '|---|---:|---:|---:|---:|']
    fresh = study.read(experiment / 'figure3_data.json')
    for name, record in [('新留出 test（主结果）', fresh), ('旧 E1 IID（补充）', data)]:
        if 'threshold' in record:
            lines.append(f"| {name} | {record['cells']['initial:0:H0']['ceiling']['mean']:.6f} | {record['cells']['initial:0:H3']['ceiling']['mean']:.6f} | {analysis.format_ci(record['contrasts']['frozen_ceiling_gain'])} | {analysis.format_ci(record['threshold'])} |")
    lines += ['', '## 全部预定节点', '', '| 训练分支 | 更新 | 部署 | 成功率 | J | M | 全状态 MSE |', '|---|---:|---|---:|---:|---:|---:|']
    for branch, update, deploy in cells():
        record = data['cells'].get(analysis.cell_key(branch, update, deploy))
        if record:
            lines.append(f"| {branch} | {update} | {deploy} | {record['success_probability']['mean']:.6f} | {record['regularized_return']['mean']:.6f} | {record['kl_mismatch']['mean']:.6f} | {record['fixed_bank_energy_mse']['mean']:.6f} |")
    if data['status'] == 'complete':
        final = data['final']['H3']
        lines += ['', '## 固定最终 1024 更新比较', '', '| 比较 | 均值 [95% CI] |', '|---|---:|']
        for key in ('recovery', 'ranking', 'vs_source_continuation', 'branch_difference_on_H3', 'A1', 'A2', 'Omega'):
            lines.append(f'| {key} | {analysis.format_ci(final[key])} |')
        for key in ('vs_initial_pair', 'vs_source_continuation'):
            if 'outcomes' in final:
                for metric in ('success_probability', 'discounted_task_reward', 'discounted_kl_penalty'):
                    lines.append(f"| {key}: {metric} | {analysis.format_ci(final['outcomes'][key][metric])} |")
        lines += ['', f"首次观察到平均 J 恢复初始搭配的节点：{data['first_observed_recovery_update']}。首次接口排序反转节点：{data['first_observed_ranking_reversal_update']}。这些条件必须分别解释，不能以排序反转替代恢复，也不能以成功率改善替代正则化回报改善。", '']
    lines += ['## 数据与审计', '', f'- 清单：`{root / "manifest.json"}`', f'- 原始概率与审计：`{root / "evaluations"}`',
              f'- 任务级数据：`{root / "paired_task_results.csv"}`', f'- 三族分项：`{root / "per_family_results.csv"}`',
              f'- 完整配对区间与曲线：`{root / "figure3_data.json"}`', '',
              '当前主论文图片仍使用新留出 test。本补充数据不替换主图中的 ceiling 或学习曲线。', '']
    (docs / 'REPORT.md').write_text('\n'.join(lines))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'evaluate', 'report'])
    parser.add_argument('--experiment', type=Path, default=study.DEFAULT)
    parser.add_argument('--harness', choices=study.HARNESSES)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == 'prepare':
        prepare(args.experiment)
    elif args.command == 'report':
        report(args.experiment)
    else:
        if any(v is None for v in (args.harness, args.adapter, args.output)):
            parser.error('evaluate requires --harness, --adapter and --output')
        evaluate(args.experiment, args.harness, args.adapter, args.output)


if __name__ == '__main__':
    main()
