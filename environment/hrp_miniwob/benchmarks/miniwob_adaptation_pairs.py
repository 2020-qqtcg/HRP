"""Registered source-to-target adaptation among H0-H5, scored on the E1 test cohort.

Labels follow the current h0_h5_v2 numbering (H1 program verify, H3 batched clicks).
The paper swaps H1 and H3, so every record also stores the paper label.
"""

import argparse
import fcntl
import hashlib
import itertools
import json
import os
import shutil
import sqlite3
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_adaptation as adaptation
from hrp_miniwob.benchmarks import miniwob_harnesses as study
from hrp_miniwob.benchmarks.miniwob_meta_harness import digest, model_files
from hrp_miniwob.harnesses.miniwob_harnesses import PROFILES
from hrp_miniwob.v2 import FAMILIES
from hrp_miniwob.v2.suite import load_suite

ROOT = study.ROOT
SOURCE = adaptation.SOURCE
FIGURE3 = adaptation.DEFAULT
DEFAULT = ROOT / 'environment/artifacts/v2/adaptation_pairs_exactpg_2026-09-24'
DOCS = ROOT / 'environment/docs/v2/adaptation_pairs_exactpg_2026-09-24'
# report() rewrites only the text between these markers.
RESULTS = ROOT / 'docs/figure4实验.md'
RESULTS_MARKERS = ('<!-- results:begin -->', '<!-- results:end -->')
NODES = adaptation.NODES
read = adaptation.read
PAPER = {'H0': 'H0', 'H1': 'H3', 'H2': 'H2', 'H3': 'H1', 'H4': 'H4', 'H5': 'H5'}
# Launch order. The core tier uses sources H0, H2 and H3 and contains one predicted failure.
PAIRS = [('H3', 'H0'), ('H0', 'H5'), ('H3', 'H5'), ('H2', 'H5'), ('H3', 'H2'),
         ('H1', 'H3'), ('H4', 'H3'), ('H4', 'H1')]
CORE = ('H0', 'H2', 'H3')
EXISTING = ('H0', 'H3')
# H0 continuation reuses the Figure 3 branch, which has the same source, seed and training suite.
CONTINUATIONS = ['H3', 'H2', 'H1', 'H4']
METRICS = ('regularized_return', 'ceiling', 'kl_mismatch', 'success_probability')


def harness_numbering():
    return read(SOURCE / 'harness_numbering.json')['current_to_recorded']


def source_hashes():
    paths = [Path(__file__), ROOT / 'environment/hrp_miniwob/benchmarks/miniwob_adaptation.py',
             ROOT / 'environment/hrp_miniwob/orchestration/run_miniwob_adaptation_pairs.py']
    return adaptation.BASE_SOURCES() | {str(p.relative_to(ROOT)): digest(p) for p in paths}


@contextmanager
def bindings(training=False):
    """Figure 3 bindings with this experiment's sealed source list."""
    old = study.source_hashes
    study.source_hashes = source_hashes
    try:
        with study.bindings(), study.base.bindings(training=training):
            host = study.base.runner
            old_eval, host.evaluate_models = host.evaluate_models, adaptation.evaluate_models
            try:
                yield host
            finally:
                host.evaluate_models = old_eval
    finally:
        study.source_hashes = old


def e1_adapter(harness):
    return SOURCE / f'runs/exactpg_{harness_numbering()[harness]}/checkpoint_001024/actor/adapter'


def e1_record(checkpoint, deployment):
    """E1 test record of the update-1024 checkpoint trained under `checkpoint` and deployed under `deployment`."""
    directory = SOURCE / 'cross_eval/exactpg_test' / harness_numbering()[deployment]
    assert digest(directory / 'matrix.json') == read(directory / 'completion.json')['matrix_sha256']
    row = next(r for r in read(directory / 'matrix.json')['rows'] if r['checkpoint'] == str(e1_adapter(checkpoint)))
    return directory / row['weights'] / 'harness_0'


def source_record(harness):
    run_path = SOURCE / f'runs/exactpg_{harness_numbering()[harness]}/run.json'
    run = read(run_path)
    entry = next(c for c in run['checkpoints'] if c['updates'] == 1024)
    directory = run_path.parent / entry['directory']
    assert run['status'] == 'complete' and directory / 'actor/adapter' == e1_adapter(harness)
    for relative, expected in entry['files_sha256'].items():
        assert digest(directory / relative) == expected, relative
    return {'paper_label': PAPER[harness], 'recorded_label': harness_numbering()[harness], 'adapter': str(e1_adapter(harness)),
            'updates': 1024, 'source_training_seed': 42, 'run_sha256': digest(run_path),
            'adapter_sha256': model_files(e1_adapter(harness)), 'checkpoint_files_sha256': entry['files_sha256']}


def arrays(path, keys=None):
    rows = {(r['family'], r['content_sha256']): r for r in read(path)['rows']}
    keys = keys or sorted(rows)
    assert sorted(rows) == keys and all(sum(k[0] == f for k in keys) == 192 for f in FAMILIES)
    return keys, {m: np.array([rows[k][m] for k in keys]) for m in METRICS}


def bootstrap(keys):
    rng = np.random.default_rng(4203)
    return [rng.choice(np.array([i for i, k in enumerate(keys) if k[0] == f]), size=(2000, 192)) for f in FAMILIES]


def summarize(values, draws):
    samples = np.mean([values[d].mean(axis=1) for d in draws], axis=0)
    low, high = np.quantile(samples, [.025, .975])
    return {'mean': float(values.mean()), 'lower': float(low), 'upper': float(high)}


def label(pair):
    return f'{pair[0]}->{pair[1]}'


def preregistration():
    """Predictions from recorded E1 test cells only. No run of the eight pairs exists when this is written."""
    keys, E = None, {}
    for s, t in itertools.product(PROFILES, PROFILES):
        keys, E[s, t] = arrays(e1_record(s, t) / 'metrics.json', keys)
    draws = bootstrap(keys)
    J = {k: v['regularized_return'] for k, v in E.items()}
    C = {t: E['H0', t]['ceiling'] for t in PROFILES}
    for (s, t), v in E.items():
        np.testing.assert_allclose(v['ceiling'], C[t], atol=1e-10, rtol=0)
    rejected = [(s, t) for s, t in itertools.permutations(PROFILES, 2) if J[s, t].mean() < J[s, s].mean()]
    ruled_out = [(s, t) for s, t in rejected if (C[t] - J[s, s]).mean() <= 0]
    candidates = [p for p in rejected if p not in ruled_out]
    assert sorted(candidates) == sorted(PAIRS + [EXISTING])

    def pair(s, t):
        margin = summarize(J[t, t] - J[s, s], draws)
        return {'code': [s, t], 'paper': [PAPER[s], PAPER[t]],
                'source_return_J_ss': summarize(J[s, s], draws),
                'frozen_return_change_J_st_minus_J_ss': summarize(J[s, t] - J[s, s], draws),
                'ceiling_gain_C_t_minus_C_s': summarize(C[t] - C[s], draws),
                'initial_target_mismatch_M_st': summarize(E[s, t]['kl_mismatch'], draws),
                'threshold_T': summarize(C[t] - J[s, s], draws),
                'target_residual_r_t': summarize(E[t, t]['kl_mismatch'], draws),
                'predicted_margin_T_minus_r_t': margin,
                'required_mismatch_reduction_fraction': float((J[s, s] - J[s, t]).mean() / E[s, t]['kl_mismatch'].mean()),
                'predicted_ranking_J_tt_minus_J_ts': summarize(J[t, t] - J[t, s], draws),
                'prediction': 'recovery' if margin['mean'] > 0 else 'no recovery',
                'close_call': bool(margin['lower'] <= 0 <= margin['upper'])}

    observed = read(FIGURE3 / 'old_e1_iid/figure3_data.json')['final']['H3']
    return {
        'status': 'registered before any run of the eight new pairs',
        'created_at_utc': datetime.now(UTC).isoformat(),
        'labels': {'current_code': list(PROFILES), 'paper': PAPER, 'E1_recorded': harness_numbering()},
        'data': {'cohort': 'E1 test split, 576 tasks, 192 per family, update-1024 checkpoints of seed 42',
                 'records_sha256': {label((s, t)): digest(e1_record(s, t) / 'metrics.json')
                                    for s, t in itertools.product(PROFILES, PROFILES)}},
        'definitions': {
            'J_st': 'regularized return of the E1 checkpoint trained under s, deployed under t',
            'T': 'C_t - J_ss. Recovery requires M(adapted, t) < T, and T <= 0 rules recovery out because M >= 0',
            'r_t': 'mismatch of the E1 checkpoint trained under t, deployed under t (reachable residual after 1,024 updates)',
            'predicted_margin': 'T - r_t = J_tt - J_ss',
            'predicted_ranking': 'J_tt - J_ts, the E1 t-trained checkpoint under t minus under s',
            'statistics': 'means weight families equally; 95% paired task bootstrap within family, 2,000 draws, seed 4203'},
        'selection': {'rule': 'frozen comparison rejects t (J_st < J_ss) and T > 0',
                      'frozen_accepted': [label(p) for p in itertools.permutations(PROFILES, 2) if p not in rejected],
                      'ruled_out_by_ceiling': {label(p): summarize(C[p[1]] - J[p[0], p[0]], draws)['mean'] for p in ruled_out},
                      'candidates': [label(p) for p in candidates]},
        'pairs': [pair(s, t) for s, t in PAIRS],
        'existing_pair': pair(*EXISTING) | {'observed_before_registration': {
            k: observed[k] for k in ('recovery', 'ranking', 'vs_source_continuation')}},
        'hypotheses': [
            {'id': 'P1', 'role': 'primary', 'endpoint': 'recovery = J(theta_s->t at 1,024, t) - J_ss',
             'rule': 'observed mean recovery > 0 exactly for pairs predicted to recover; score = correct signs out of 8',
             'baselines': 'frozen rule predicts no recovery for all 8; ceiling rule (T > 0) predicts recovery for all 8'},
            {'id': 'P2', 'role': 'primary', 'endpoint': 'Spearman(predicted margin, observed recovery) over the 8 pairs',
             'rule': 'positive, one-sided exact permutation p < 0.05'},
            {'id': 'S1', 'role': 'secondary', 'endpoint': 'gain over continuation = J(theta_s->t, t) - J(theta_s->s, s) at 1,024',
             'rule': 'sign agrees with the registered prediction'},
            {'id': 'S2', 'role': 'secondary', 'endpoint': 'ranking = J(theta_s->t, t) - J(theta_s->t, s) at 1,024',
             'rule': 'sign agrees with the predicted ranking and Spearman > 0'},
            {'id': 'X1', 'role': 'exploratory', 'endpoint': 'first scheduled update with positive mean recovery',
             'rule': 'among recovering pairs, later for a larger required mismatch-reduction fraction'}],
        'reporting': ['Classify each pair by the sign of the mean, and separately as positive, inconclusive or negative by its 95% interval.',
                      'Report P1 on all 8 pairs and on the pairs that are not close calls.',
                      'Report the existing H0->H3 pair separately. It was observed before registration.',
                      'Record any departure in deviations.json. Registered predictions never change.']}


def cell_path(experiment, source, target, update, deployment):
    name = f'initial_{source}' if update == 0 else f'seed42_{source}_to_{target}/checkpoint_{update:06d}'
    return experiment / 'evaluations' / name / deployment


def namespace(candidate_sha256, config, sources):
    """Frozen-decoder cache directory name, as computed by miniwob_meta_harness.adapter_for."""
    identity = {'candidate': candidate_sha256, 'models': config['model_sha256'], 'reference': config['reference_model'],
                'reviewer': config['reviewer_model'], 'source': sources}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def prepare(experiment):
    experiment = study.base.require_v2_path(experiment)
    if (experiment / 'manifest.json').exists():
        manifest = read(experiment / 'manifest.json')
        for name in ('config', 'proposal', 'preregistration'):
            assert digest(experiment / f'{name}.json') == manifest[f'{name}_sha256'], name
        return manifest
    experiment.mkdir(parents=True, exist_ok=True)
    registered = preregistration()
    write_json(experiment / 'preregistration.json', registered)
    # E1 training tasks in E1 order give the same task draws as Figure 3's H0 branch, whose suite task files were removed.
    # E1's 72-task search split (24 per family) only records in-training calibration curves.
    origin = read(SOURCE / 'suite/manifest.json')
    entries = ([e for e in origin['entries'] if e['split'] == 'train']
               + [e | {'split': 'calibration'} for e in origin['entries'] if e['split'] == 'search'])
    assert ([(e['seed'], e['sha256']) for e in entries if e['split'] == 'train']
            == [(e['seed'], e['sha256']) for e in read(FIGURE3 / 'suite/manifest.json')['entries'] if e['split'] == 'train'])
    (experiment / 'suite/tasks').mkdir(parents=True, exist_ok=True)
    for entry in entries:
        shutil.copyfile(SOURCE / 'suite' / entry['json'], experiment / 'suite' / entry['json'])
        assert digest(experiment / 'suite' / entry['json']) == entry['sha256']
    write_json(experiment / 'suite/manifest.json', {
        'version': 'v2', 'status': 'complete', 'profile': 'adaptation_pairs', 'horizon': origin['horizon'],
        'families': origin['families'], 'entries': entries, 'source_suite_sha256': digest(SOURCE / 'suite/manifest.json'),
        'validation': 'E1 task files and browser audits; calibration is the E1 search split'})
    write_json(experiment / 'splits.json', {s: [e['seed'] for e in entries if e['split'] == s]
                                            for s in ('search', 'train', 'calibration', 'test', 'ood')})
    for relative in ('train_public.json', 'train_probe.json'):
        shutil.copyfile(SOURCE / relative, experiment / relative)
    hashes = {key: digest(experiment / relative) for relative, key in [
        ('suite/manifest.json', 'suite_manifest_sha256'), ('splits.json', 'splits_sha256'),
        ('train_public.json', 'train_public_sha256'), ('train_probe.json', 'train_probe_sha256')]}
    cohort = load_suite(SOURCE / 'suite', 'test')
    used = {m.task.fingerprint for split in ('train', 'calibration') for m in load_suite(experiment / 'suite', split)}
    assert len({m.task.fingerprint for m in cohort}) == 576 and not used & {m.task.fingerprint for m in cohort}
    run, evaluation = FIGURE3 / 'runs/seed42_H0', FIGURE3 / 'old_e1_iid/evaluations/seed42_H0/checkpoint_001024/H0'
    assert read(run / 'run.json')['status'] == 'complete' and read(FIGURE3 / 'proposal.json')['source']['adapter'] == str(e1_adapter('H0'))
    assert read(evaluation / 'completion.json')['metrics_sha256'] == digest(evaluation / 'metrics.json')
    # H4 reuses E1's archived reviewer decisions, so its control problem matches the recorded E1 cells.
    e1_config, sources = read(SOURCE / 'config.json'), source_hashes()
    config = e1_config | hashes | {
        'harness_protocol': 'h0_h5_v2', 'harnesses': list(PROFILES),
        'action_decoder': 'complete parameterized menu; one reversible A-Z/a-z token per action',
        'controller': 'explicit verification result or automatic gate',
        'budget_unit': 'one actor decision; a batch may execute two native clicks; gamma applies per actor decision',
        'claim_scope': 'Registered source-to-target adaptation among H0-H5 on three native task families',
        'adaptation_protocol': 'pairwise_registered_v1', 'source_sha256_at_preparation': sources,
        'split_counts': {s: len(v) for s, v in read(experiment / 'splits.json').items()}}
    old = (SOURCE / 'frozen' / namespace(read(e1_record('H4', 'H4').parents[1] / 'matrix.json')['rows'][0]['candidate_sha256'],
                                         e1_config, e1_config['source_sha256_at_preparation']) / 'fixed_microbatch8/reviewer.sqlite')
    new = (experiment / 'frozen' / namespace(study.adapter.Candidate(study.candidate_path('H4')).sha256, config, sources)
           / 'fixed_microbatch8/reviewer.sqlite')
    new.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(old, new)
    with sqlite3.connect(new) as db:
        reviews = db.execute('SELECT COUNT(*) FROM probabilities').fetchone()[0]
    proposal = {
        'design': 'registered source-to-target adaptation; update-1024 E1 sources; paired continuation per source',
        'pairs': [list(p) for p in PAIRS], 'paper_labels': PAPER, 'core_sources': list(CORE), 'continuations': CONTINUATIONS,
        'sources': {s: source_record(s) for s in sorted({s for s, _ in PAIRS})},
        'training': {'method': 'exactpg', 'updates': 1024, 'tasks_per_update': 4, 'seed': 42, 'checkpoint_updates': NODES,
                     'initialization': 'source LoRA weights; fresh AdamW moments', 'settings': 'E1 and Figure 3 optimizer settings',
                     'data': 'E1 1,440 training tasks in E1 order (paired draws); calibration = E1 search split, 72 tasks'},
        'evaluation': {'cohort': 'E1 test', 'suite': str(SOURCE / 'suite'), 'suite_manifest_sha256': digest(SOURCE / 'suite/manifest.json'),
                       'task_seeds': [m.task.seed for m in cohort], 'target_updates': NODES, 'source_updates': [0, 1024],
                       'continuation_updates': [1024]},
        'reused': {'H0': {'run': str(run), 'run_sha256': digest(run / 'run.json'), 'evaluation': str(evaluation),
                          'metrics_sha256': digest(evaluation / 'metrics.json')}},
        'reviewer_decisions': {'from': str(old), 'to': str(new), 'rows': reviews, 'sha256': digest(new)},
        'preregistration_sha256': digest(experiment / 'preregistration.json')}
    write_json(experiment / 'proposal.json', proposal)
    config['proposal_sha256'] = digest(experiment / 'proposal.json')
    write_json(experiment / 'config.json', config)
    for relative in sources:
        target = experiment / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    manifest = {'status': 'sealed', 'created_at_utc': datetime.now(UTC).isoformat(),
                **{f'{name}_sha256': digest(experiment / f'{name}.json') for name in ('config', 'proposal', 'preregistration')},
                'primary_cohort': 'E1 test, 576 tasks', 'bootstrap_seed': 4203, 'bootstrap_samples': 2000,
                'uncertainty': 'paired task bootstrap within family; source and adaptation seed 42 only'}
    write_json(experiment / 'manifest.json', manifest)
    write_preregistration_docs(registered, manifest)
    return manifest


def write_preregistration_docs(registered, manifest):
    def ci(value):
        return f"{value['mean']:+.4f} [{value['lower']:+.4f}, {value['upper']:+.4f}]"

    def row(p):
        return (f"| {p['paper'][0]}→{p['paper'][1]} | {p['code'][0]}→{p['code'][1]} | {p['frozen_return_change_J_st_minus_J_ss']['mean']:+.4f} "
                f"| {p['threshold_T']['mean']:.4f} | {p['initial_target_mismatch_M_st']['mean']:.4f} | {p['required_mismatch_reduction_fraction']:.0%} "
                f"| {p['target_residual_r_t']['mean']:.4f} | {ci(p['predicted_margin_T_minus_r_t'])} "
                f"| {'恢复' if p['prediction'] == 'recovery' else '不恢复'}{'（接近零）' if p['close_call'] else ''} "
                f"| {p['predicted_ranking_J_tt_minus_J_ts']['mean']:+.4f} |")

    existing = registered['existing_pair']
    lines = ['# 8 对 harness 适配实验：预注册预测', '',
             f"登记时间 {registered['created_at_utc']}。本文件与 `preregistration.json` 在任何一对新适配运行之前写入，"
             f"sha256 为 `{manifest['preregistration_sha256']}`，同时记录在实验目录的 `manifest.json` 中。预测以后不再修改，偏离预案的操作写入 `deviations.json`。", '',
             '## 编号', '',
             '代码使用 h0_h5_v2 编号，论文把 H1 与 H3 对调。下表先列论文编号。', '',
             '| 含义 | 论文 | 代码 | E1 记录 |', '|---|---|---|---|']
    meanings = {'H0': '直接执行', 'H1': '程序 verify', 'H2': '自动 gate', 'H3': '两次点击 batch', 'H4': '冻结 LM reviewer', 'H5': '改写 system prompt'}
    lines += [f"| {meanings[h]} | {PAPER[h]} | {h} | {harness_numbering()[h]} |" for h in PROFILES]
    lines += ['', '## 数据与定义', '',
              '- 全部数值来自 E1 在 E1 test 集（576 个任务，每族 192 个）上的交叉评测，六个 checkpoint 都取 seed 42 的 update 1,024。',
              '- J_st：在 s 下训练的 checkpoint 部署到 t 的正则化回报。C_t：t 的 ceiling。M_st = C_t − J_st。',
              '- T = C_t − J_ss。适配后的 pair 超过初始 pair 当且仅当 M(θ_{s→t}, t) < T。因为 M ≥ 0，T ≤ 0 时任何适配都不能恢复。',
              '- r_t：在 t 下训练 1,024 次更新后的残余 mismatch（E1 对角格）。预测余量 T − r_t = J_tt − J_ss。余量为正预测恢复。',
              '- 需消掉的比例 = (J_ss − J_st) / M_st。预测排序 = J_tt − J_ts。',
              '- 均值三族等权。区间为族内配对任务 bootstrap（2,000 次，seed 4203）的 95% 区间。', '',
              '## 筛选', '',
              f"规则：frozen 比较拒绝 t（J_st < J_ss）且 T > 0。30 个有向切换中，frozen 接受 {len(registered['selection']['frozen_accepted'])} 个，"
              f"frozen 拒绝 {len(registered['selection']['ruled_out_by_ceiling']) + len(registered['selection']['candidates'])} 个。"
              f"拒绝的切换里 {len(registered['selection']['ruled_out_by_ceiling'])} 个满足 T ≤ 0，理论上不可能恢复，不运行。"
              f"余下 {len(registered['selection']['candidates'])} 个候选，其中论文 H0→H1 已完成，新增 8 对。", '',
              '## 登记的预测', '',
              '| 论文 | 代码 | frozen 回报变化 | T | M_st | 需消掉比例 | r_t | 预测余量 T − r_t [95% CI] | 预测 | 预测排序 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---|---:|']
    lines += [row(p) for p in registered['pairs']]
    observed = existing['observed_before_registration']
    lines += ['', '已完成的论文 H0→H1（代码 H0→H3）在登记前已观察到，单独报告，不计入检验：', '',
              '| 论文 | 代码 | frozen 回报变化 | T | M_st | 需消掉比例 | r_t | 预测余量 T − r_t [95% CI] | 预测 | 预测排序 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---|---:|', row(existing), '',
              f"实测：恢复 {ci(observed['recovery'])}，排序 {ci(observed['ranking'])}，相对续训 {ci(observed['vs_source_continuation'])}。", '',
              '## 假设与判定', '',
              '- **P1（主）**：1,024 次更新后，恢复差 J(θ_{s→t}, t) − J_ss 的均值为正，当且仅当预测为"恢复"。计分为 8 对中符号正确的对数，与两条基线比较：frozen 规则（全部不恢复）和 ceiling 规则（T > 0，全部恢复）。另报告去掉"接近零"对之后的结果。',
              '- **P2（主）**：8 对的预测余量与实测恢复差的 Spearman 相关为正，单侧精确置换检验 p < 0.05。',
              '- **S1（次）**：相对续训的增益 J(θ_{s→t}, t) − J(θ_{s→s}, s) 的符号与预测一致。',
              '- **S2（次）**：排序 J(θ_{s→t}, t) − J(θ_{s→t}, s) 的符号与预测排序一致，且两者 Spearman 相关为正。',
              '- **X1（探索）**：在恢复的 pair 中，首次恢复的节点随需消掉比例增大而推后。', '',
              '每对同时按均值符号分类，并按 95% 区间分为显著为正、不确定、显著为负三类。', '',
              '## 实验设置', '',
              '- 源 checkpoint：E1 各 harness 的 update 1,024（seed 42），不按 calibration 另选。',
              '- 每条分支 1,024 次 Exact-PG 更新，每次 4 个任务，seed 42，AdamW 矩重置，其余优化设置与 E1 和 Figure 3 相同。',
              '- 训练使用 E1 的 1,440 个训练任务，顺序与 Figure 3 相同，所以所有分支（含 Figure 3 的 H0 续训分支）逐次抽到相同任务，H0 续训可以直接复用。训练中的 calibration 曲线使用 E1 的 search 集（72 个任务，每族 24 个），不参与任何选择。',
              '- 评测：目标 harness 在更新 0/32/64/128/256/512/1,024，源 harness 在 0 和 1,024，续训分支在 1,024。',
              '- H4 复用 E1 存档的 reviewer 判定。每个初始格与 E1 记录逐任务核对状态图、reward、reference 与 actor 概率，核对通过后才开始训练。', '']
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / 'PREREGISTRATION.md').write_text('\n'.join(lines))


def train(experiment, source, harness):
    proposal = read(experiment / 'proposal.json')
    assert [source, harness] in proposal['pairs'] or (source == harness and source in proposal['continuations'])
    output = experiment / 'runs' / f'seed42_{source}_to_{harness}'
    args = argparse.Namespace(experiment=experiment, output=output, candidate=study.candidate_path(harness), device='cuda:0',
        method='exactpg', seed=42, batch_size=4, updates=1024, checkpoint_interval=64,
        checkpoint_updates=NODES, rollout_episodes=None, limit=None, resume=True,
        initial_adapter=Path(proposal['sources'][source]['adapter']))
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a') as lock, bindings(training=True):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study.base.train(args)
        study.base.select_checkpoint(output)


def replay_audit(proposal, harness, adapter_path, models, output, result):
    """Compare graphs and reference with E1. An E1 source checkpoint must also replay its E1 actor probabilities."""
    source = next((s for s, r in proposal['sources'].items() if r['adapter'] == str(adapter_path)), None)
    baseline = e1_record(source or 'H0', harness)
    reference_error = actor_error = 0.
    for model in models:
        with (np.load(baseline / f'policies/{model.task.key}.npz') as old,
              np.load(output / f'tasks/{model.task.key}/policies/{model.task.key}.npz') as new):
            for key in ('next_state', 'rewards', 'fixed_bank'):
                np.testing.assert_array_equal(old[key], new[key])
            reference_error = max(reference_error, float(np.abs(np.log(old['reference']) - np.log(new['reference'])).max()))
            if source:
                actor_error = max(actor_error, float(np.abs(old['log_policy'] - new['log_policy']).max()))
    assert reference_error <= 1e-3, reference_error
    audit = {'baseline': str(baseline), 'task_graphs_identical': len(models), 'max_reference_log_error_vs_E1': reference_error}
    if source:
        previous = read(baseline / 'metrics.json')['metrics']
        differences = {k: result['metrics'][k] - previous[k] for k in METRICS + ('fixed_bank_energy_mse',)}
        assert actor_error <= 1e-3 and max(map(abs, differences.values())) < 1e-6, (actor_error, differences)
        audit.update(max_actor_log_error_vs_E1=actor_error, aggregate_differences_vs_E1=differences)
    return audit


def evaluate(experiment, harness, adapter_path, output):
    proposal = read(experiment / 'proposal.json')
    models = load_suite(SOURCE / 'suite', 'test')
    assert digest(SOURCE / 'suite/manifest.json') == proposal['evaluation']['suite_manifest_sha256']
    assert [m.task.seed for m in models] == proposal['evaluation']['task_seeds']
    with bindings() as host:
        config = host.read_config(experiment)
        identity = {'config_sha256': digest(experiment / 'config.json'), 'cohort': 'E1 test', 'harness': harness,
                    'adapter': str(adapter_path), 'adapter_sha256': model_files(adapter_path)}
        output.mkdir(parents=True, exist_ok=True)
        with (output / '.run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (output / 'evaluation.json').exists():
                assert read(output / 'evaluation.json') == identity
            write_json(output / 'evaluation.json', identity)
            if (output / 'completion.json').exists():
                assert read(output / 'completion.json')['metrics_sha256'] == digest(output / 'metrics.json')
                return
            started = time.monotonic()
            actor = host.LanguagePolicy(config['actor_model'], adapter_path=str(adapter_path), trainable=False, device='cuda:0', rank=16)
            frozen = host.load_frozen(config, 'cuda:0')
            bound = host.adapter_for(actor, host.Candidate(study.candidate_path(harness)), frozen, config, experiment)
            try:
                result = adaptation.evaluate_models(models, bound, output)
                write_json(output / 'replay_audit.json', replay_audit(proposal, harness, adapter_path, models, output, result))
                write_json(output / 'completion.json', {'metrics_sha256': digest(output / 'metrics.json'),
                           'replay_audit_sha256': digest(output / 'replay_audit.json'), 'tasks': result['tasks'],
                           'wall_seconds': time.monotonic() - started, 'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES')})
            finally:
                bound.close()


def spearman(x, y):
    """Rank correlation with average ranks for ties, and a one-sided exact permutation p-value."""
    def ranks(values):
        values = np.asarray(values, dtype=float)
        result = np.empty(len(values))
        result[values.argsort()] = np.arange(len(values))
        for value in np.unique(values):
            result[values == value] = result[values == value].mean()
        return result

    rx, ry = ranks(x), ranks(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return {'rho': None, 'p_one_sided': None, 'pairs': len(x)}
    rho = float(np.corrcoef(rx, ry)[0, 1])
    values = np.array([np.corrcoef(rx, np.array(p))[0, 1] for p in itertools.permutations(ry)])
    return {'rho': rho, 'p_one_sided': float(np.mean(values >= rho - 1e-12)), 'pairs': len(x)}


def report(experiment):
    proposal, registered = read(experiment / 'proposal.json'), read(experiment / 'preregistration.json')
    keys, cells = None, {}
    for path in sorted((experiment / 'evaluations').glob('**/completion.json')):
        assert read(path)['metrics_sha256'] == digest(path.parent / 'metrics.json')
        keys, cells[path.parent] = arrays(path.parent / 'metrics.json', keys)
    reused = proposal['reused']['H0']
    assert digest(Path(reused['evaluation']) / 'metrics.json') == reused['metrics_sha256']
    if cells:
        keys, cells[cell_path(experiment, 'H0', 'H0', 1024, 'H0')] = arrays(Path(reused['evaluation']) / 'metrics.json', keys)
    draws = bootstrap(keys) if keys else None
    results = []
    for p in registered['pairs']:
        s, t = p['code']
        get = lambda target, update, deployment: cells.get(cell_path(experiment, s, target, update, deployment))
        start, frozen = get(t, 0, s), get(t, 0, t)
        record = {'code': [s, t], 'paper': p['paper'], 'prediction': p['prediction'], 'close_call': p['close_call'],
                  'predicted_margin': p['predicted_margin_T_minus_r_t']['mean'],
                  'predicted_ranking': p['predicted_ranking_J_tt_minus_J_ts']['mean'], 'curve': []}
        if start is None or frozen is None:
            results.append(record)
            continue
        threshold = frozen['ceiling'] - start['regularized_return']
        assert abs(threshold.mean() - p['threshold_T']['mean']) < 1e-5, 'Initial cells do not replay the registered E1 values'
        for update in NODES[1:]:
            adapted = get(t, update, t)
            if adapted is not None:
                record['curve'].append({'updates': update, 'M_t': summarize(adapted['kl_mismatch'], draws),
                                        'recovery': summarize(adapted['regularized_return'] - start['regularized_return'], draws)})
        final, under_source, continuation = get(t, 1024, t), get(t, 1024, s), get(s, 1024, s)
        if final is not None:
            recovery = record['curve'][-1]['recovery']
            record.update(recovery=recovery, observed='recovery' if recovery['mean'] > 0 else 'no recovery',
                          interval='positive' if recovery['lower'] > 0 else ('negative' if recovery['upper'] < 0 else 'inconclusive'),
                          first_recovery_update=next((c['updates'] for c in record['curve'] if c['recovery']['mean'] > 0), None),
                          success_change=summarize(final['success_probability'] - start['success_probability'], draws))
        if final is not None and under_source is not None:
            record['ranking'] = summarize(final['regularized_return'] - under_source['regularized_return'], draws)
        if final is not None and continuation is not None:
            record['gain_over_continuation'] = summarize(final['regularized_return'] - continuation['regularized_return'], draws)
        results.append(record)
    done = [r for r in results if 'observed' in r]
    tests = {'completed_pairs': len(done)}
    if done:
        tests['P1'] = {'correct': sum(r['observed'] == r['prediction'] for r in done),
                       'frozen_rule_correct': sum(r['observed'] == 'no recovery' for r in done),
                       'ceiling_rule_correct': sum(r['observed'] == 'recovery' for r in done),
                       'correct_excluding_close_calls': sum(r['observed'] == r['prediction'] for r in done if not r['close_call']),
                       'pairs_excluding_close_calls': sum(not r['close_call'] for r in done)}
    if len(done) >= 3:
        tests['P2'] = spearman([r['predicted_margin'] for r in done], [r['recovery']['mean'] for r in done])
    gained = [r for r in done if 'gain_over_continuation' in r]
    if gained:
        tests['S1'] = {'correct': sum((r['gain_over_continuation']['mean'] > 0) == (r['prediction'] == 'recovery') for r in gained), 'pairs': len(gained)}
    ranked = [r for r in done if 'ranking' in r]
    if ranked:
        tests['S2'] = {'correct': sum((r['ranking']['mean'] > 0) == (r['predicted_ranking'] > 0) for r in ranked), 'pairs': len(ranked)}
        if len(ranked) >= 3:
            tests['S2'] |= spearman([r['predicted_ranking'] for r in ranked], [r['ranking']['mean'] for r in ranked])
    fraction = {tuple(p['code']): p['required_mismatch_reduction_fraction'] for p in registered['pairs']}
    recovered = [r for r in done if r['first_recovery_update'] is not None]
    if len(recovered) >= 3:
        tests['X1'] = spearman([fraction[tuple(r['code'])] for r in recovered], [r['first_recovery_update'] for r in recovered])
    draws_by_run = {}
    for run in sorted((experiment / 'runs').glob('seed42_*/run.json')) + [Path(reused['run']) / 'run.json']:
        if read(run)['status'] == 'complete':
            updates = {r['updates']: r['task_seeds'] for r in map(json.loads, (run.parent / 'updates.jsonl').read_text().splitlines())}
            draws_by_run[str(run.parent)] = [updates[u] for u in range(1, 1025)]
    assert len({json.dumps(v) for v in draws_by_run.values()}) <= 1, 'Training task draws are not paired'
    data = {'status': 'complete' if len(done) == len(PAIRS) and len(gained) == len(PAIRS) else 'partial',
            'created_at_utc': datetime.now(UTC).isoformat(), 'preregistration_sha256': digest(experiment / 'preregistration.json'),
            'evaluated_cells': len(cells), 'paired_training_runs': len(draws_by_run), 'pairs': results, 'tests': tests}
    write_json(experiment / 'pairs_data.json', data)

    def ci(value):
        return f"{value['mean']:+.4f} [{value['lower']:+.4f}, {value['upper']:+.4f}]" if value else '—'

    def correlation(key, name):
        value = tests.get(key)
        if value is None:
            return f'- {name}：至少 3 对完成后计算。'
        if value['rho'] is None:
            return f"- {name}：{value['pairs']} 对的取值相同，无法计算相关。"
        return f"- {name}：Spearman ρ = {value['rho']:+.3f}，单侧置换 p = {value['p_one_sided']:.4f}，{value['pairs']} 对。"

    names = {'recovery': '恢复', 'no recovery': '不恢复'}
    intervals = {'positive': '显著为正', 'inconclusive': '不确定', 'negative': '显著为负'}
    lines = ['## 结果（自动生成）', '',
             f"`miniwob_adaptation_pairs report` 于 {data['created_at_utc']} 写入，调度器每完成一个作业刷新一次。"
             f"状态 {data['status']}：已完成 {len(done)}/8 对，{len(cells)} 个评测格，{len(draws_by_run)} 条已完成训练分支的任务抽样逐次一致。"
             f"机器可读数据见 `{experiment / 'pairs_data.json'}`。", '',
             '| 论文 | 代码 | 预测 | 预测余量 | 实测恢复 [95% CI] | 区间 | 首次恢复更新 | 排序 [95% CI] | 相对续训 [95% CI] | 成功率变化 |',
             '|---|---|---|---:|---:|---|---:|---:|---:|---:|']
    for r in results:
        first = (r['first_recovery_update'] or '未恢复') if 'observed' in r else '—'
        success = f"{100 * r['success_change']['mean']:+.1f} pp" if 'success_change' in r else '—'
        lines.append(f"| {r['paper'][0]}→{r['paper'][1]} | {r['code'][0]}→{r['code'][1]} | {names[r['prediction']]}{'（接近零）' if r['close_call'] else ''} "
                     f"| {r['predicted_margin']:+.4f} | {ci(r.get('recovery'))} | {intervals.get(r.get('interval'), '—')} | {first} "
                     f"| {ci(r.get('ranking'))} | {ci(r.get('gain_over_continuation'))} | {success} |")
    lines += ['', '恢复差 J(θ_{s→t}, t) − J_ss 在各适配更新数上的均值：', '',
              '| 论文 | ' + ' | '.join(str(u) for u in NODES[1:]) + ' |', '|---|' + '---:|' * len(NODES[1:])]
    for r in results:
        curve = {c['updates']: c['recovery']['mean'] for c in r['curve']}
        lines.append(f"| {r['paper'][0]}→{r['paper'][1]} | " + ' | '.join(f'{curve[u]:+.4f}' if u in curve else '—' for u in NODES[1:]) + ' |')
    lines += ['', '### 预注册检验', '']
    if 'P1' in tests:
        p1 = tests['P1']
        lines.append(f"- P1（主）：{p1['correct']}/{len(done)} 对的恢复方向与预测一致。frozen 规则 {p1['frozen_rule_correct']}/{len(done)}，"
                     f"ceiling 规则 {p1['ceiling_rule_correct']}/{len(done)}。去掉接近零的对后 {p1['correct_excluding_close_calls']}/{p1['pairs_excluding_close_calls']}。")
    else:
        lines.append('- P1（主）：尚无完成的 pair。')
    lines.append(correlation('P2', 'P2（主）预测余量与实测恢复'))
    lines.append(f"- S1（次）：相对续训增益的方向 {tests['S1']['correct']}/{tests['S1']['pairs']} 与预测一致。" if 'S1' in tests
                 else '- S1（次）：尚无续训结果。')
    if 'S2' in tests:
        lines.append(f"- S2（次）：排序方向 {tests['S2']['correct']}/{tests['S2']['pairs']} 与预测一致。")
    lines.append(correlation('S2', 'S2（次）预测排序与实测排序'))
    lines.append(correlation('X1', 'X1（探索）需消掉比例与首次恢复更新'))
    text = RESULTS.read_text()
    begin, end = text.index(RESULTS_MARKERS[0]) + len(RESULTS_MARKERS[0]), text.index(RESULTS_MARKERS[1])
    RESULTS.write_text(text[:begin] + '\n' + '\n'.join(lines) + '\n' + text[end:])
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'evaluate', 'report'])
    parser.add_argument('--experiment', type=Path, default=DEFAULT)
    parser.add_argument('--source', choices=PROFILES)
    parser.add_argument('--harness', choices=PROFILES)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'train' and (args.source is None or args.harness is None):
        parser.error('train requires --source and --harness')
    if args.command == 'evaluate' and any(v is None for v in (args.harness, args.adapter, args.output)):
        parser.error('evaluate requires --harness, --adapter and --output')
    torch.set_num_threads(2)
    experiment = study.base.require_v2_path(args.experiment)
    if args.command == 'prepare':
        print(json.dumps(prepare(experiment), indent=2))
    elif args.command == 'train':
        train(experiment, args.source, args.harness)
    elif args.command == 'evaluate':
        evaluate(experiment, args.harness, args.adapter, args.output)
    else:
        data = report(experiment)
        print(json.dumps({'status': data['status'], 'tests': data['tests'], 'report': str(RESULTS)}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
