"""Run the two-branch adaptation experiment on idle GPUs 0, 1, 2, 3."""

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

from hrp_miniwob.benchmarks import miniwob_adaptation as study
from hrp_miniwob.benchmarks.miniwob_adaptation_report import cells, evaluation_path, report, DOCS
from hrp_miniwob.adapters.native_miniwob import write_json

GPUS = [0, 1, 2, 3]


def job_specs(experiment):
    specs = {}
    base = [sys.executable, '-m', 'hrp_miniwob.benchmarks.miniwob_adaptation']
    source = study.read(experiment / 'proposal.json')['source']['adapter']
    for branch, update, deploy in cells():
        key = f'eval_{branch}_{update:06d}_{deploy}'
        adapter = source if update == 0 else str(experiment / f'runs/seed42_{branch}/checkpoint_{update:06d}/actor/adapter')
        output = evaluation_path(experiment, branch, update, deploy)
        specs[key] = {'kind': 'evaluate', 'branch': branch, 'update': update, 'deployment': deploy, 'output': str(output),
                      'command': base + ['evaluate', '--experiment', str(experiment), '--harness', deploy, '--adapter', adapter, '--output', str(output)]}
    for branch in study.HARNESSES:
        specs[f'train_{branch}'] = {'kind': 'train', 'branch': branch, 'output': str(experiment / f'runs/seed42_{branch}'),
                                   'command': base + ['train', '--experiment', str(experiment), '--harness', branch, '--seed', '42']}
    return specs


def complete(spec):
    path = Path(spec['output'])
    if spec['kind'] == 'train':
        return (path / 'run.json').exists() and study.read(path / 'run.json')['status'] == 'complete' and (path / 'e1_selection.json').exists()
    return (path / 'completion.json').exists() and study.read(path / 'completion.json')['metrics_sha256'] == study.digest(path / 'metrics.json')


def ready(spec, specs, experiment):
    if spec['kind'] == 'train':
        return all(complete(s) for s in specs.values() if s['kind'] == 'evaluate' and s['update'] == 0)
    if spec['update'] == 0:
        return True
    path = experiment / f"runs/seed42_{spec['branch']}/run.json"
    if not path.exists():
        return False
    run = study.read(path)
    return any(c['updates'] == spec['update'] and 'calibration' in c for c in run['checkpoints'])


def alive(job):
    try:
        process = psutil.Process(job['pid'])
        return process.create_time() == job['process_created'] and process.status() != psutil.STATUS_ZOMBIE and process.cmdline() == job['command']
    except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
        return False


def free_gpus():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True)
    return [int(line.split(',')[0]) for line in result.stdout.splitlines()
            if int(line.split(',')[0]) in GPUS and int(line.split(',')[1]) < 512]


def finish(experiment):
    data = report(experiment)
    assert data['status'] == 'complete'
    if (experiment / 'old_e1_iid/manifest.json').exists():
        from hrp_miniwob.benchmarks import miniwob_adaptation_retest
        supplementary = miniwob_adaptation_retest.report(experiment)
        assert supplementary['status'] == 'complete'
        with (DOCS / 'REPORT.md').open('a') as stream:
            stream.write(f'\n旧 E1 IID 的 26 个补充评测单元也已完成。报告：{DOCS / "old_e1_iid/REPORT.md"}。\n')
    write_json(experiment / 'completion_summary.json', {
        'status': 'complete', 'completed_test_cells': data['completed_cells'],
        'report': str(DOCS / 'REPORT.md'), 'data_sha256': study.digest(experiment / 'figure3_data.json'),
        'conclusions': data['conclusions'], 'final': data['final']})
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=study.DEFAULT)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    specs = job_specs(experiment)
    if not args.execute:
        print(json.dumps({'gpus': GPUS, 'jobs': specs}, indent=2))
        return
    assert study.read(experiment / 'acceptance_gpu.json')['status'] == 'passed'
    assert study.read(experiment / 'acceptance_cpu.json')['status'] == 'passed'
    config = study.read(experiment / 'config.json')
    expected_sources = config['source_sha256_at_preparation']
    revision = experiment / 'report_revision_applied.json'
    if revision.exists() and all(complete(spec) for spec in specs.values()):
        expected_sources = expected_sources | study.read(revision)['replacement_sha256']
    assert expected_sources == study.source_hashes(), 'Sealed measurement source changed'
    logs = experiment / 'logs'
    logs.mkdir(exist_ok=True)
    path = experiment / 'scheduler.json'
    with (experiment / '.scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = study.read(path) if path.exists() else {'status': 'running', 'started_at_utc': datetime.now(UTC).isoformat(), 'jobs': {}}
        state['supervisor_pid'] = os.getpid()
        children = {}
        while True:
            changed, failed = False, []
            for key, job in state['jobs'].items():
                if job['status'] != 'running':
                    continue
                if key in children:
                    finished = children[key].poll() is not None
                else:
                    finished = not alive(job)
                if finished:
                    job.update(status='complete' if complete(specs[key]) else 'failed', finished_at_utc=datetime.now(UTC).isoformat(),
                               wall_seconds=time.time() - job['started_epoch'])
                    job['gpu_hours'] = job['wall_seconds'] / 3600
                    changed = True
                    if job['status'] == 'failed':
                        failed.append(key)
            if failed:
                state.update(status='failed', failed_jobs=failed)
                write_json(path, state)
                raise RuntimeError(f'Jobs failed: {failed}')
            if all(complete(spec) for spec in specs.values()):
                state.update(status='evaluations_complete', finished_at_utc=datetime.now(UTC).isoformat())
                write_json(path, state)
                finish(experiment)
                state['status'] = 'complete'
                write_json(path, state)
                print(json.dumps({'status': 'complete', 'report': str(DOCS / 'REPORT.md')}), flush=True)
                return
            available = free_gpus()
            occupied = {job['gpu'] for job in state['jobs'].values() if job['status'] == 'running'}
            available = [g for g in available if g not in occupied]
            # Training takes priority once both initial deployments have completed.
            ordered = sorted(specs, key=lambda key: (specs[key]['kind'] != 'train', specs[key].get('update', -1), key))
            for key in ordered:
                spec = specs[key]
                if complete(spec) or state['jobs'].get(key, {}).get('status') == 'running' or not ready(spec, specs, experiment):
                    continue
                if not available:
                    break
                preferred = 1 if spec['branch'] == 'H0' else 2
                gpu = preferred if spec['kind'] == 'train' and preferred in available else (3 if spec['kind'] == 'evaluate' and 3 in available else available[0])
                env = os.environ | {'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false'}
                with (logs / f'{key}.log').open('a') as log:
                    child = subprocess.Popen(spec['command'], cwd=study.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                previous = state['jobs'].get(key)
                attempts = (previous.get('previous_attempts', []) + [{k: v for k, v in previous.items() if k != 'previous_attempts'}]) if previous else []
                state['jobs'][key] = {'status': 'running', 'pid': child.pid, 'process_created': psutil.Process(child.pid).create_time(),
                                      'gpu': gpu, 'command': spec['command'], 'started_epoch': time.time(),
                                      'started_at_utc': datetime.now(UTC).isoformat(), 'previous_attempts': attempts, 'log': str(logs / f'{key}.log')}
                children[key] = child
                available.remove(gpu)
                changed = True
                print(json.dumps({'started': key, 'gpu': gpu, 'pid': child.pid}), flush=True)
            state.update(status='running', updated_at_utc=datetime.now(UTC).isoformat())
            write_json(path, state)
            if changed:
                report(experiment)
            time.sleep(30)


if __name__ == '__main__':
    main()
