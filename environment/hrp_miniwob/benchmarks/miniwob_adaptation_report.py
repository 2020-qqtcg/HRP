"""Paired task inference and reproducible reports for bounded harness adaptation."""

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from hrp_miniwob.benchmarks import miniwob_adaptation as study
from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.v2 import FAMILIES

METRICS = ('ceiling', 'regularized_return', 'kl_mismatch', 'success_probability',
           'fixed_bank_energy_mse', 'expected_actor_decisions_native', 'expected_native_clicks_native',
           'discounted_task_reward', 'discounted_cost', 'discounted_kl_penalty')
DOCS = study.ROOT / 'environment/docs/v2/figure3_h0_h3_exactpg_2026-09-14'


def evaluation_path(experiment, branch, update, deployment):
    name = 'initial' if update == 0 else f'seed42_{branch}/checkpoint_{update:06d}'
    return experiment / 'evaluations' / name / deployment


def cells():
    return [('initial', 0, h) for h in study.HARNESSES] + [
        (branch, update, deploy) for update in study.NODES[1:]
        for branch in study.HARNESSES for deploy in study.HARNESSES]


def cell_key(branch, update, deployment):
    return f'{branch}:{update}:{deployment}'


def collect(experiment):
    records, sources = {}, {}
    for branch, update, deploy in cells():
        output = evaluation_path(experiment, branch, update, deploy)
        if not (output / 'completion.json').exists():
            continue
        path = output / 'metrics.json'
        assert study.read(output / 'completion.json')['metrics_sha256'] == study.digest(path)
        sources[str(path)] = study.digest(path)
        record = study.read(path)
        assert record['tasks'] == len(record['rows']) == 576
        assert max(max(r['gap_identity_error'], r['mixed_gap_identity_error']) for r in record['rows']) < 1e-8
        records[cell_key(branch, update, deploy)] = record
    return records, sources


def analyze(experiment):
    records, sources = collect(experiment)
    manifest = study.read(experiment / 'manifest.json')
    data = {'status': 'complete' if len(records) == 26 else 'partial', 'completed_cells': len(records), 'required_cells': 26,
            'created_at_utc': datetime.now(UTC).isoformat(), 'experiment': str(experiment),
            'config_sha256': study.digest(experiment / 'config.json'), 'raw_sources_sha256': sources,
            'source': manifest['source'], 'adaptation_seed': 42, 'source_training_seed': 42,
            'analysis_revision': '2026-09-15_success_and_return', 'analysis_code_sha256': study.digest(Path(__file__)),
            'bootstrap': {'unit': 'native task content, paired across checkpoints and interfaces, stratified by family',
                          'samples': 2000, 'seed': 4203, 'interval': '95% percentile, pointwise', 'training_seed_uncertainty': False},
            'nodes': study.NODES, 'cells': {}, 'contrasts': {}, 'curves': {}}
    if not records:
        return data, []
    first = next(iter(records.values()))
    keys = sorted((r['family'], r['content_sha256']) for r in first['rows'])
    assert len(keys) == len(set(keys)) == 576
    rng = np.random.default_rng(4203)
    draws = [rng.choice(np.array([i for i, k in enumerate(keys) if k[0] == family]), size=(2000, 192)) for family in FAMILIES]
    assert all(sum(k[0] == family for k in keys) == 192 for family in FAMILIES)

    def summarize(values):
        samples = np.mean([values[d].mean(axis=1) for d in draws], axis=0)
        low, high = np.quantile(samples, [.025, .975])
        return {'mean': float(values.mean()), 'lower': float(low), 'upper': float(high)}

    def compare_outcomes(target, baseline):
        differences = {m: target[m] - baseline[m] for m in
                       ('success_probability', 'regularized_return', 'discounted_task_reward',
                        'discounted_cost', 'discounted_kl_penalty')}
        residual = differences['regularized_return'] - (differences['discounted_task_reward']
                   - differences['discounted_cost'] - differences['discounted_kl_penalty'])
        np.testing.assert_allclose(residual, 0, atol=1e-10, rtol=0)
        return {m: summarize(values) for m, values in differences.items()} | {
            'max_task_return_decomposition_error': float(np.abs(residual).max())}

    arrays, raw_rows = {}, []
    for key, record in records.items():
        rows = {(r['family'], r['content_sha256']): r for r in record['rows']}
        assert set(rows) == set(keys)
        arrays[key] = {metric: np.array([rows[k][metric] for k in keys]) for metric in METRICS}
        data['cells'][key] = {metric: summarize(value) for metric, value in arrays[key].items()}
        data['cells'][key]['per_family'] = record['per_family']
        branch, update, deploy = key.split(':')
        raw_rows.extend({'source_training_seed': 42, 'adaptation_seed': 42, 'training_interface': branch,
                         'additional_updates': int(update), 'deployment_interface': deploy, **row} for row in record['rows'])
        np.testing.assert_allclose([arrays[key][m].mean() for m in METRICS], [record['metrics'][m] for m in METRICS], atol=1e-12)
    if not all(f'initial:0:{h}' in arrays for h in study.HARNESSES):
        return data, raw_rows
    initial = {h: arrays[f'initial:0:{h}'] for h in study.HARNESSES}
    for key, value in arrays.items():
        np.testing.assert_allclose(value['ceiling'], initial[key.split(':')[2]]['ceiling'], atol=1e-10, rtol=0)
    threshold = initial['H3']['ceiling'] - initial['H0']['regularized_return']
    data['threshold'] = summarize(threshold)
    data['contrasts']['frozen_ceiling_gain'] = summarize(initial['H3']['ceiling'] - initial['H0']['ceiling'])
    data['contrasts']['frozen_return_difference'] = summarize(initial['H3']['regularized_return'] - initial['H0']['regularized_return'])
    data['contrasts']['frozen_mismatch_increase'] = summarize(initial['H3']['kl_mismatch'] - initial['H0']['kl_mismatch'])
    data['frozen_outcomes'] = compare_outcomes(initial['H3'], initial['H0'])
    data['max_gap_identity_error'] = max(r['gap_identity_error'] for r in raw_rows)
    for branch in study.HARNESSES:
        curve = []
        for update in study.NODES:
            key = cell_key(branch if update else 'initial', update, 'H3')
            if key not in arrays:
                continue
            value = arrays[key]
            recovery = value['regularized_return'] - initial['H0']['regularized_return']
            np.testing.assert_allclose(recovery, threshold - value['kl_mismatch'], atol=1e-8, rtol=0)
            item = {'updates': update, 'M_H3': summarize(value['kl_mismatch']),
                    'J_H3': summarize(value['regularized_return']), 'recovery': summarize(recovery),
                    'M_reduction_H3': summarize(initial['H3']['kl_mismatch'] - value['kl_mismatch']),
                    'outcomes': {'vs_initial_pair': compare_outcomes(value, initial['H0'])}}
            h0key = cell_key(branch if update else 'initial', update, 'H0')
            if h0key in arrays:
                h0 = arrays[h0key]
                a1 = h0['regularized_return'] - initial['H0']['regularized_return']
                a2 = value['regularized_return'] - initial['H3']['regularized_return']
                item.update(J_H0=summarize(h0['regularized_return']), ranking=summarize(value['regularized_return'] - h0['regularized_return']),
                            A1=summarize(a1), A2=summarize(a2), Omega=summarize(a2 - a1))
                item['outcomes'].update(interface_ranking=compare_outcomes(value, h0),
                                        A1=compare_outcomes(h0, initial['H0']), A2=compare_outcomes(value, initial['H3']))
            source_h0, source_h3 = cell_key('H0', update, 'H0'), cell_key('H0', update, 'H3')
            if branch == 'H3' and source_h0 in arrays and source_h3 in arrays:
                item.update(vs_source_continuation=summarize(value['regularized_return'] - arrays[source_h0]['regularized_return']),
                            branch_difference_on_H3=summarize(value['regularized_return'] - arrays[source_h3]['regularized_return']))
                item['outcomes'].update(vs_source_continuation=compare_outcomes(value, arrays[source_h0]),
                                        same_H3_branch_difference=compare_outcomes(value, arrays[source_h3]))
            curve.append(item)
        data['curves'][branch] = curve
    if data['status'] == 'complete':
        data['final'] = {b: data['curves'][b][-1] for b in study.HARNESSES}
        data['first_observed_recovery_update'] = next((r['updates'] for r in data['curves']['H3'] if r['recovery']['mean'] > 0), None)
        data['first_pointwise_CI_recovery_update'] = next((r['updates'] for r in data['curves']['H3'] if r['recovery']['lower'] > 0), None)
        data['first_observed_ranking_reversal_update'] = next((r['updates'] for r in data['curves']['H3'] if r['ranking']['mean'] > 0), None)
        final = data['final']['H3']
        ceiling_difference = data['contrasts']['frozen_ceiling_gain']
        data['conclusions'] = {
            'ceiling': ('ceiling 差值区间包含零，不能确认 H3 ceiling 更高，也不构成两接口 ceiling 等价的证据。'
                        if ceiling_difference['lower'] <= 0 <= ceiling_difference['upper'] else
                        ('任务区间支持 H3 ceiling 更高。' if ceiling_difference['lower'] > 0 else '任务区间支持 H3 ceiling 更低。')),
            'frozen_swap': ('冻结切换后 mismatch 增大、正则化回报下降，任务配对区间支持这一方向。'
                           if data['contrasts']['frozen_return_difference']['upper'] < 0 and data['contrasts']['frozen_mismatch_increase']['lower'] > 0
                           else '冻结切换的回报与 mismatch 方向需按各自任务配对区间判断。'),
            'finite_adaptation': '预定最终预算恢复了初始搭配，配对区间下界大于零。' if final['recovery']['lower'] > 0
                else ('预定最终预算均值恢复初始搭配，但区间包含零。' if final['recovery']['mean'] > 0 else '预定最终预算未观察到均值超过初始搭配。'),
            'ranking': '最终 H3 优于 H0，任务配对区间下界大于零。' if final['ranking']['lower'] > 0
                else ('最终均值偏好 H3，但区间包含零。' if final['ranking']['mean'] > 0 else '最终均值仍偏好 H0。'),
            'source_continuation': '新搭配超过同预算源接口续训，任务配对区间下界大于零。' if final['vs_source_continuation']['lower'] > 0
                else ('新搭配均值超过源接口续训，但区间包含零。' if final['vs_source_continuation']['mean'] > 0 else '新搭配均值未超过同预算源接口续训。')}
    return data, raw_rows


def format_ci(value):
    return f"{value['mean']:.6f} [{value['lower']:.6f}, {value['upper']:.6f}]"


def report(experiment, docs=DOCS):
    docs.mkdir(parents=True, exist_ok=True)
    data, rows = analyze(experiment)
    if rows:
        with (experiment / 'paired_task_results.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    family_rows = []
    for key, values in data['cells'].items():
        branch, update, deploy = key.split(':')
        family_rows.extend({'source_training_seed': 42, 'adaptation_seed': 42, 'training_interface': branch,
                            'additional_updates': int(update), 'deployment_interface': deploy,
                            'family': family, **metrics} for family, metrics in values['per_family'].items())
    if family_rows:
        with (experiment / 'per_family_results.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(family_rows[0]))
            writer.writeheader()
            writer.writerows(family_rows)
    data['training'] = {}
    data['calibration_accounting'] = {}
    training_rows = []
    for branch in study.HARNESSES:
        path = experiment / f'runs/seed42_{branch}/run.json'
        if path.exists():
            run = study.read(path)
            data['training'][branch] = {'status': run['status'], 'completed_updates': run.get('completed_updates', 0),
                                        'accounting': run['accounting'], 'attempts': run['attempts']}
            updates = path.parent / 'updates.jsonl'
            # Latest retry for an update is the accepted trajectory after resume.
            by_update = {r['updates']: r for r in (json.loads(line) for line in updates.read_text().splitlines())} if updates.exists() else {}
            training_rows.append(by_update)
            calibration = [study.read(path.parent / c['directory'] / 'calibration/metrics.json')['accounting']
                           for c in run['checkpoints'] if 'calibration' in c]
            data['calibration_accounting'][branch] = {k: sum(row[k] for row in calibration) for k in calibration[0]} if calibration else {}
    if data['status'] == 'complete':
        assert len(training_rows) == 2 and all(r['status'] == 'complete' and r['completed_updates'] == 1024 for r in data['training'].values())
        assert all(set(r) == set(range(1, 1025)) for r in training_rows)
        assert all(training_rows[0][u]['task_seeds'] == training_rows[1][u]['task_seeds'] for u in range(1, 1025))
        data['paired_training_task_draws_verified'] = 4096
    scheduler = experiment / 'scheduler.json'
    if scheduler.exists():
        data['resource_jobs'] = study.read(scheduler).get('jobs', {})
        data['recorded_gpu_hours'] = sum(j.get('gpu_hours', 0) + sum(a.get('gpu_hours', 0) for a in j.get('previous_attempts', [])) for j in data['resource_jobs'].values())
    write_json(experiment / 'figure3_data.json', data)
    source = data['source']
    lines = ['# Figure 3: H0 → H3 有界适应实验', '', f"状态：**{data['status']}**，完成 {data['completed_cells']}/26 个完整 test 评测单元。", '',
             '## 固定设计', '', f"源模型：`{source['adapter']}`，E1 H0 calibration 所选 update 1024。源训练 seed=42，适应 seed=42，只有一个训练随机种子。", '',
             '- 两支从同一源 LoRA 权重初始化，均重置 AdamW 的一阶、二阶矩和 scheduler。',
             '- 每支额外 1024 次 Exact-PG 更新，每次 4 个任务，共 4096 次任务抽取。两支使用配对的任务随机序列。',
             '- Qwen3.5-2B，LoRA rank=16、alpha=32、dropout=0，基座与前向 BF16，LoRA/master moments FP32，动作归一化与 DP FP64。',
             '- AdamW lr=1e-5 常数，betas=(0.9,0.999)，eps=1e-8，weight_decay=0，grad_clip=1。microbatch=8 个状态，非重入 activation checkpointing。',
             '- 训练目标为每任务在 50% 原生起点与 50% 合法中间状态覆盖混合分布下的 KL 正则回报。Exact-PG 枚举动作并用该策略的精确访问权重求梯度。',
             '- 三个原生任务族：click-checkboxes、click-checkboxes-soft、ascending-numbers。复用 1440 个训练任务。新 calibration=72，test=576，分别每族 24/192。',
             '- 新评测实例逐一在浏览器验证成功路径，按 seed 与内容排除保留的旧数据。清单见 exclusions.json、suite/manifest.json 和 preparation.json。',
             '- 节点 0、32、64、128、256、512、1024。主结果使用预先固定的 1024，保留完整曲线，额外最佳 checkpoint 仅按 calibration 选择。',
             '- 两接口 gamma=0.95、tau=0.1、显式点击与提交 cost=0，最多 10 个 actor 决策。reference 是冻结基础 Qwen3.5-2B，在各自接口 prompt 下计算，绝不重设为源/续训/适应模型。',
             '- H0 单步执行，H3 保留单步并提供有序两次点击 batch，原生终止时停止执行后续点击。H3 的一次决策可能执行两次点击，折扣按 actor 决策施加。',
             '- 完整动作菜单经可逆单 token 代码归一化，采样温度为 1，理论目标温度 tau=0.1。Exact-PG 与精确评测不采样轨迹。', '',
             '## 测量和统计', '',
             'test 评测均从共同原生任务起点出发。C 由 soft DP 求解，J 由被测策略的独立 Bellman 策略评估求得。M 使用同一被测策略的完整动作概率和前向访问分布，计算 tau × Σ_s visits(s) KL(π||π*)，不以 C−J 构造 M。终止态零损失，非终止 occupancy 不重新归一化。', '',
             '检查 J=C−M 的浮点残差属于实现一致性验证，不能作为有限训练一定成功的经验证据。固定状态 energy MSE 单独保留，不代入恢复阈值。H0/H3 动作集不同，不计算未对齐的跨接口能量向量差。', '',
             '95% 区间按 native task content 配对，族内 bootstrap 2000 次，随机种子 4203，再以三个族等权平均。所有曲线、接口和差值共用重采样索引。区间为逐点区间，不含训练 seed 方差，不是整条曲线同时置信带。测试结果不用于改预算、换接口或选择 checkpoint。', '']
    lines += ['由一般恒等式 J=C−M，恢复初始搭配满足 J(θt,H3)>J(θ0,H0) 当且仅当 M(θt,H3)<T，其中 T=C(H3)−J(θ0,H0)。这一比较不要求 C(H3)>C(H0)，T 也不等于 ceiling gain。', '',
              '本次新 test 的 ceiling 差值区间包含零，不能写成 H3 ceiling 更高，也不能断言等价。因此本实验检验一般冻结切换与有限适应机制，不能称为正文中显式假设 ΔC>0 的更高 ceiling 特例的经验验证。两个固定接口和 reference 下的 ceiling 不随训练改变。', '']
    if 'threshold' in data:
        lines += [f"固定恢复阈值 T=C(H3)−J(θ0,H0)：{format_ci(data['threshold'])}。", '', '| 冻结切换比较 | 均值 [95% CI] |', '|---|---:|']
        lines += [f'| {name} | {format_ci(value)} |' for name, value in data['contrasts'].items()]
        lines += ['', '## 完整原生 test 曲线', '', '| 训练支路 | 额外更新 | 部署 | 成功率 | J | M | C | 全状态 energy MSE |', '|---|---:|---|---:|---:|---:|---:|---:|']
        for branch, update, deploy in cells():
            key = cell_key(branch, update, deploy)
            if key not in data['cells']:
                continue
            r = data['cells'][key]
            lines.append(f"| {branch} | {update} | {deploy} | {r['success_probability']['mean']:.6f} | {r['regularized_return']['mean']:.6f} | {r['kl_mismatch']['mean']:.6f} | {r['ceiling']['mean']:.6f} | {r['fixed_bank_energy_mse']['mean']:.6f} |")
    if data['status'] == 'complete':
        final = data['final']['H3']
        lines += ['', '## 预定最终预算比较', '', '| 比较 | 均值 [95% CI] |', '|---|---:|']
        names = {'M_reduction_H3': '目标接口 mismatch 的下降', 'recovery': '超过初始搭配 J(θt,H3)−J(θ0,H0)',
                 'ranking': '更新后排序 J(θt,H3)−J(θt,H0)', 'vs_source_continuation': '超过源续训 J(θt,H3)−J(θs,H0)',
                 'branch_difference_on_H3': '同一部署 H3 下分支差 J(θt,H3)−J(θs,H3)', 'A1': 'A1：目标适应后 H0 的绝对变化',
                 'A2': 'A2：目标适应后 H3 的绝对变化', 'Omega': 'Omega=A2−A1'}
        lines += [f'| {name} | {format_ci(final[key])} |' for key, name in names.items()]
        frozen = data['contrasts']
        lines += ['', '## 实测解读', '',
                  *data['conclusions'].values(), '',
                  f"冻结切换：ceiling 差为 {frozen['frozen_ceiling_gain']['mean']:+.6f}，冻结模型回报差为 {frozen['frozen_return_difference']['mean']:+.6f}。这两个量分别检验接口潜力与当前模型的实际利用效果。", '',
                  f"有限适应：最终恢复差为 {final['recovery']['mean']:+.6f}。均值首次跨过恢复阈值的已测节点为 {data['first_observed_recovery_update']}，恢复差的逐点 CI 下界首次大于零的节点为 {data['first_pointwise_CI_recovery_update']}。未观察到的节点记为 None，不能解释为永远不能恢复。", '',
                  f"更新后接口排序差为 {final['ranking']['mean']:+.6f}，H0 与 H3 的绝对变化分别为 {final['A1']['mean']:+.6f}、{final['A2']['mean']:+.6f}。排序变化必须结合两边绝对变化解读。", '',
                  f"与相同更新预算下的 H0 续训相比，新搭配回报差为 {final['vs_source_continuation']['mean']:+.6f}。同一 H3 部署下目标适应相对源续训的差为 {final['branch_difference_on_H3']['mean']:+.6f}。", '',
                  f"最大独立 gap 恒等式残差：{data['max_gap_identity_error']:.3g}。已核对两支全部 4096 次任务抽取逐更新一致。"]
    lines += ['', '## 任务成功率的配对比较', '',
              '成功率为无折扣任务成功概率。下表以百分点报告相对初始搭配、同预算 H0 续训原搭配的任务配对差值及 95% CI。512 等节点均为中期结果，主比较固定为 1024。', '',
              '| H3 适应更新 | 结果阶段 | 相对初始 H0 搭配 (百分点) | 相对同预算 H0 续训原搭配 (百分点) |', '|---:|---|---:|---:|']
    def success_ci(outcome):
        return format_ci({k: 100 * v for k, v in outcome['success_probability'].items()})
    for row in data['curves'].get('H3', []):
        outcomes = row['outcomes']
        continuation = success_ci(outcomes['vs_source_continuation']) if 'vs_source_continuation' in outcomes else '—'
        stage = '预定最终结果' if row['updates'] == 1024 else ('冻结切换' if row['updates'] == 0 else '中期')
        lines.append(f"| {row['updates']} | {stage} | {success_ci(outcomes['vs_initial_pair'])} | {continuation} |")
    lines += ['', '## 成功率与正则化回报的区别', '',
              'J = discounted_task_reward − discounted_cost − discounted_kl_penalty。显式 cost 固定为零，记录的 KL penalty 已包含 tau 与沿轨迹折扣。成功率不扣除 KL 惩罚，也不包含完成时机的折扣，因此成功率更高并不必然意味着 J 更高。以下直接汇总已有逐任务记录，没有新增模型前向或改变目标。', '',
              '| 模型／更新／部署 | 成功率 | 折扣任务奖励 | 折扣 KL 惩罚 | J |', '|---|---:|---:|---:|---:|']
    summary_cells = [('initial', 0, h) for h in study.HARNESSES] + [(b, u, h) for u in (512, 1024) for b in study.HARNESSES for h in study.HARNESSES]
    for branch, update, deploy in summary_cells:
        r = data['cells'].get(cell_key(branch, update, deploy))
        if r:
            lines.append(f"| {branch}/{update}/{deploy} | {r['success_probability']['mean']:.6f} | {r['discounted_task_reward']['mean']:.6f} | {r['discounted_kl_penalty']['mean']:.6f} | {r['regularized_return']['mean']:.6f} |")
    lines += ['', '| H3 适应更新 | 配对基线 | Δ 折扣任务奖励 [95% CI] | Δ 折扣 KL 惩罚 [95% CI] | Δ J [95% CI] |', '|---:|---|---:|---:|---:|']
    for row in data['curves'].get('H3', []):
        for key, label in [('vs_initial_pair', '初始 H0 搭配'), ('vs_source_continuation', '同预算 H0 续训原搭配')]:
            outcome = row['outcomes'].get(key)
            if outcome:
                lines.append(f"| {row['updates']} | {label} | {format_ci(outcome['discounted_task_reward'])} | {format_ci(outcome['discounted_kl_penalty'])} | {format_ci(outcome['regularized_return'])} |")
    lines += ['', '三任务族分项保存在 figure3_data.json 的 cells/per_family，并导出 per_family_results.csv。全 test 三族等权聚合始终是主结果，分项不用于重新选择任务或解释为新的主比较。', '',
              '## 计算量与限制', '',
              'updates.jsonl 保存每次任务 ID、模型前向、反向状态数、microbatch 数、prompt tokens、reference 请求和耗时。训练 run.json/accounting 汇总优化阶段，scheduler.json 逐进程记录含模型加载、校准和 test 的占卡时长。重试时报告实际所有尝试，不把重做计算算作新增训练预算。', '',
              'Exact-PG 更新不执行浏览器 rollout，实际训练环境交互为零。评测的 expected_actor_decisions_native 与 expected_native_clicks_native 是无折扣访问分布下的解析期望，不能当作真实执行调用次数或样本效率。任务生成的实际原生成功重放步数在 suite manifest 中记录。模型前向 token 与预期执行成本分别报告。', '',
              '源接口来自旧 IID 的探索性选择。本次以新留出任务验证该特定 H0→H3 案例，不构成所有接口修改的普遍结论。只有源训练/适应 seed 42，不估计训练随机性。H3 扩大菜单且一决策可含两次点击，匹配更新数/任务抽取不等于匹配 FLOPs、token 或原生点击。', '',
              '与论文现有占位协议的差异：本实验不以前置的有限 probe/prediction sweep 为条件，不报告未实际进行的独立轨迹预测验证或多 seed 区间。这里的 J 和成功率由已验证有限状态转移核的精确策略评估得到。论文正文和图注由发送任务验收后统一修改。', '',
              '## 文件', '', f'- Manifest：`{experiment / "manifest.json"}`', f'- 原始任务级结果：`{experiment / "paired_task_results.csv"}`',
              f'- 机器可读图表数据：`{experiment / "figure3_data.json"}`', f'- 运行日志：`{experiment / "logs"}`',
              f'- GPU 接受检查：`{experiment / "acceptance_gpu.json"}`', f'- 源码封存：`{experiment / "source_snapshot"}`', '']
    lines += ['## 已记录训练计算量', '', '| 支路 | 完成 updates | 优化耗时 (小时) | actor 状态前向 | 含梯度的状态前向 | actor prompt tokens | reference prompt tokens |', '|---|---:|---:|---:|---:|---:|---:|']
    for branch, run in data['training'].items():
        c = run['accounting']
        lines.append(f"| {branch} | {run['completed_updates']} | {c.get('wall_seconds', 0)/3600:.3f} | {c.get('actor_forwards', 0)} | {c.get('actor_gradient_observations', 0)} | {c.get('actor_prompt_tokens', 0)} | {c.get('reference_prompt_tokens', 0)} |")
    lines += ['', f"已结束进程累计占卡时间（包括加载、校准、test 和记录的重试）：{data.get('recorded_gpu_hours', 0):.3f} GPU 小时。优化阶段计数与校准计数分别保留，未完成的中间任务/更新可能尚未落盘，不能把成功记录的 token 数当作所有失败计算的精确总量。", '',
              '| 模型 | 部署 | 每任务期望 actor 决策 | 每任务期望原生点击 |', '|---|---|---:|---:|']
    for branch, update in [('initial', 0), ('H0', 1024), ('H3', 1024)]:
        for h in study.HARNESSES:
            row = data['cells'].get(cell_key(branch, update, h))
            if row:
                lines.append(f"| {branch} | {h} | {row['expected_actor_decisions_native']['mean']:.4f} | {row['expected_native_clicks_native']['mean']:.4f} |")
    lines.append('')
    (docs / 'REPORT.md').write_text('\n'.join(lines))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=study.DEFAULT)
    parser.add_argument('--docs', type=Path, default=DOCS)
    args = parser.parse_args()
    data = report(args.experiment, args.docs)
    print(json.dumps({'status': data['status'], 'cells': data['completed_cells'], 'report': str(args.docs / 'REPORT.md')}))


if __name__ == '__main__':
    main()
