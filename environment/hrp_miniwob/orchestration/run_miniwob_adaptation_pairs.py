"""Run the registered adaptation pairs on the listed GPUs. Without --execute, write the plan and time estimate."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil

from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_adaptation_pairs as study


def recorded_hours():
    """Hours per job on one RTX 4090 (48 GiB), from the E1 and Figure 3 records."""
    jobs = study.read(study.SOURCE / 'pipeline.json')['jobs']

    def span(job):
        return (datetime.fromisoformat(job['finished_at_utc']) - datetime.fromisoformat(job['started_at_utc'])).total_seconds() / 3600

    # Figure 3 branches: whole process time over optimizer time (checkpoint calibration and model loading).
    runs = [study.read(study.FIGURE3 / f'runs/seed42_{b}/run.json') for b in ('H0', 'H3')]
    overhead = sum(r['last_attempt_wall_seconds'] / r['accounting']['wall_seconds'] for r in runs) / len(runs)
    hours = {}
    for harness, recorded in study.harness_numbering().items():
        evaluation = span(jobs[f'cross_{recorded}_test']) / 7
        hours[harness] = {'train': study.read(study.SOURCE / f'runs/exactpg_{recorded}/run.json')['accounting']['wall_seconds'] / 3600 * overhead,
                          'evaluate': evaluation, 'first_evaluate': span(jobs[f'selected_exactpg_{recorded}_test'])}
    return hours, overhead


def job_specs(experiment, scope):
    proposal = study.read(experiment / 'proposal.json')
    base = [sys.executable, '-m', 'hrp_miniwob.benchmarks.miniwob_adaptation_pairs']
    pairs = [tuple(p) for p in proposal['pairs']]
    tiers = {'core': [p for p in pairs if p[0] in proposal['core_sources']]}
    if scope == 'all':
        tiers['extended'] = [p for p in pairs if p[0] not in proposal['core_sources']]
    specs = {}

    def evaluation(tier, source, target, update, deployment):
        adapter = (proposal['sources'][source]['adapter'] if update == 0
                   else str(experiment / f'runs/seed42_{source}_to_{target}/checkpoint_{update:06d}/actor/adapter'))
        output = study.cell_path(experiment, source, target, update, deployment)
        key = f'eval_initial_{source}_{deployment}' if update == 0 else f'eval_{source}_to_{target}_{update:06d}_{deployment}'
        specs[key] = {'kind': 'evaluate', 'tier': tier, 'source': source, 'target': target, 'update': update,
                      'deployment': deployment, 'output': str(output),
                      'command': base + ['evaluate', '--experiment', str(experiment), '--harness', deployment,
                                         '--adapter', adapter, '--output', str(output)]}

    for tier, members in tiers.items():
        sources = list(dict.fromkeys(s for s, _ in members))
        for source in sources:
            for deployment in [source] + [t for s, t in members if s == source]:
                evaluation(tier, source, source, 0, deployment)
        branches = members + [(s, s) for s in sources if s in proposal['continuations']]
        for source, target in branches:
            specs[f'train_{source}_to_{target}'] = {
                'kind': 'train', 'tier': tier, 'source': source, 'target': target,
                'output': str(experiment / f'runs/seed42_{source}_to_{target}'),
                'command': base + ['train', '--experiment', str(experiment), '--source', source, '--harness', target]}
            updates = study.NODES[1:] if source != target else [1024]
            for update in updates:
                evaluation(tier, source, target, update, target)
            if source != target:
                evaluation(tier, source, target, 1024, source)
    return specs


def complete(spec):
    path = Path(spec['output'])
    if spec['kind'] == 'train':
        return (path / 'run.json').exists() and study.read(path / 'run.json')['status'] == 'complete' and (path / 'e1_selection.json').exists()
    return (path / 'completion.json').exists() and study.read(path / 'completion.json')['metrics_sha256'] == study.digest(path / 'metrics.json')


def ready(spec, specs):
    if spec['kind'] == 'train':
        # A branch starts only after every initial cell of its source has replayed E1.
        return all(complete(s) for s in specs.values() if s['kind'] == 'evaluate' and s['update'] == 0 and s['source'] == spec['source'])
    if spec['update'] == 0:
        return True
    path = Path(specs[f"train_{spec['source']}_to_{spec['target']}"]['output']) / 'run.json'
    return path.exists() and any(c['updates'] == spec['update'] and 'calibration' in c for c in study.read(path)['checkpoints'])


def plan(experiment, specs):
    hours, overhead = recorded_hours()
    seen, rows, totals = set(), [], {}
    for key, spec in specs.items():
        harness = spec['target'] if spec['kind'] == 'train' else spec['deployment']
        estimate = hours[harness]['train' if spec['kind'] == 'train' else 'evaluate']
        if spec['kind'] == 'evaluate' and harness not in seen:
            seen.add(harness)
            estimate = hours[harness]['first_evaluate']
        rows.append({'job': key, 'tier': spec['tier'], 'kind': spec['kind'], 'hours': estimate})
        group = totals.setdefault(spec['tier'], {'train': 0., 'evaluate': 0.})
        group[spec['kind']] += estimate
    total = sum(r['hours'] for r in rows)
    lines = ['# 8 对 harness 适配实验：运行计划与单卡耗时', '',
             f'由 `python -m hrp_miniwob.orchestration.run_miniwob_adaptation_pairs` 生成，共 {len(rows)} 个作业。耗时按 E1 与 Figure 3 在同一台机器单张 RTX 4090（48 GiB）上的记录估算：',
             f'训练 = E1 同一 harness 1,024 次更新的优化时长 × {overhead:.3f}（Figure 3 两条分支的整进程/优化时长之比，含 checkpoint calibration 与加载）。',
             '评测 = E1 在同一 test 集上的交叉评测时长 / 7。每个 harness 的第一次评测另含 reference 缓存填充，取 E1 选点评测时长。', '',
             '| harness（代码） | 论文 | 训练 1,024 更新 (h) | 评测 576 任务 (h) | 首次评测 (h) |', '|---|---|---:|---:|---:|']
    lines += [f"| {h} | {study.PAPER[h]} | {v['train']:.2f} | {v['evaluate']:.2f} | {v['first_evaluate']:.2f} |" for h, v in hours.items()]
    lines += ['', '| 层级 | 训练 (h) | 评测 (h) | 合计 (h) | 合计 (天) |', '|---|---:|---:|---:|---:|']
    for tier, group in totals.items():
        lines.append(f"| {tier} | {group['train']:.1f} | {group['evaluate']:.1f} | {sum(group.values()):.1f} | {sum(group.values()) / 24:.1f} |")
    lines += [f'| 全部 | {sum(g["train"] for g in totals.values()):.1f} | {sum(g["evaluate"] for g in totals.values()):.1f} | {total:.1f} | {total / 24:.1f} |', '',
              '单卡按下表顺序串行执行。多卡时调度器按同一顺序填充空闲卡。', '', '| 顺序 | 作业 | 层级 | 估计 (h) |', '|---:|---|---|---:|']
    lines += [f"| {i} | {r['job']} | {r['tier']} | {r['hours']:.2f} |" for i, r in enumerate(rows, 1)]
    study.DOCS.mkdir(parents=True, exist_ok=True)
    (study.DOCS / 'PLAN.md').write_text('\n'.join(lines) + '\n')
    return {'jobs': len(rows), 'single_gpu_hours': total, 'single_gpu_days': total / 24, 'tiers': totals, 'plan': str(study.DOCS / 'PLAN.md')}


def alive(job):
    try:
        process = psutil.Process(job['pid'])
        return process.create_time() == job['process_created'] and process.status() != psutil.STATUS_ZOMBIE and process.cmdline() == job['command']
    except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
        return False


def free_gpus(gpus):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True)
    return [int(i) for i, used in (line.split(',') for line in result.stdout.splitlines()) if int(i) in gpus and int(used) < 512]


def execute(experiment, specs, gpus):
    study.prepare(experiment)
    logs = experiment / 'logs'
    logs.mkdir(exist_ok=True)
    path = experiment / 'scheduler.json'
    with (experiment / '.scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = study.read(path) if path.exists() else {'started_at_utc': datetime.now(UTC).isoformat(), 'jobs': {}}
        state.update(status='running', supervisor_pid=os.getpid(), gpus=gpus)
        children = {}
        while True:
            failed, finished = [], []
            for key, job in state['jobs'].items():
                if job['status'] != 'running':
                    continue
                if (children[key].poll() is not None) if key in children else not alive(job):
                    job.update(status='complete' if complete(specs[key]) else 'failed', finished_at_utc=datetime.now(UTC).isoformat(),
                               gpu_hours=(time.time() - job['started_epoch']) / 3600)
                    (failed if job['status'] == 'failed' else finished).append(key)
            if failed:
                state.update(status='failed', failed_jobs=failed)
                write_json(path, state)
                raise RuntimeError(f'Jobs failed: {failed}. Logs: {logs}. Rerun the launcher to resume.')
            if finished:
                study.report(experiment)
            if all(complete(spec) for spec in specs.values()):
                study.report(experiment)
                state.update(status='complete', finished_at_utc=datetime.now(UTC).isoformat())
                write_json(path, state)
                print(json.dumps({'status': 'complete', 'report': str(study.RESULTS)}, ensure_ascii=False), flush=True)
                return
            occupied = {job['gpu'] for job in state['jobs'].values() if job['status'] == 'running'}
            available = [g for g in free_gpus(gpus) if g not in occupied]
            for key, spec in specs.items():
                if not available:
                    break
                if complete(spec) or state['jobs'].get(key, {}).get('status') == 'running' or not ready(spec, specs):
                    continue
                gpu = available.pop(0)
                env = os.environ | {'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false'}
                with (logs / f'{key}.log').open('a') as log:
                    child = subprocess.Popen(spec['command'], cwd=study.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                previous = state['jobs'].get(key)
                state['jobs'][key] = {'status': 'running', 'pid': child.pid, 'process_created': psutil.Process(child.pid).create_time(),
                                      'gpu': gpu, 'command': spec['command'], 'started_epoch': time.time(),
                                      'started_at_utc': datetime.now(UTC).isoformat(), 'log': str(logs / f'{key}.log'),
                                      'previous_attempts': (previous.get('previous_attempts', []) + [
                                          {k: v for k, v in previous.items() if k != 'previous_attempts'}]) if previous else []}
                children[key] = child
                print(json.dumps({'started': key, 'gpu': gpu, 'pid': child.pid}), flush=True)
            state['updated_at_utc'] = datetime.now(UTC).isoformat()
            write_json(path, state)
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=study.DEFAULT)
    parser.add_argument('--scope', choices=['core', 'all'], default='all', help='core: sources H0, H2, H3 (paper H0, H2, H1)')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    experiment = study.study.base.require_v2_path(args.experiment)
    study.prepare(experiment)
    specs = job_specs(experiment, args.scope)
    if args.execute:
        execute(experiment, specs, args.gpus)
    else:
        print(json.dumps(plan(experiment, specs), indent=2))


if __name__ == '__main__':
    main()
