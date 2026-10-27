"""Schedule eight training branches and 79 H100 evaluations on at most eight GPUs."""

import argparse
import fcntl
import json
import math
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime

from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_adaptation_pairs as study
from hrp_miniwob.benchmarks import miniwob_adaptation_pairs_hpc3 as hardware
from hrp_miniwob.orchestration import run_miniwob_adaptation_pairs as scheduler


def specs_for(experiment):
    specs = {
        key: spec for key, spec in scheduler.job_specs(experiment, 'all').items()
        if not (spec['source'] == spec['target'] and
                (spec['kind'] == 'train' or spec['update'] == 1024))
    }
    assert len(specs) == 77
    base = [sys.executable, '-m', 'hrp_miniwob.benchmarks.miniwob_adaptation_pairs']
    for source, deployment in hardware.baseline_cells():
        key = f'eval_initial_{source}_{deployment}'
        if key in specs:
            continue
        output = study.cell_path(experiment, source, source, 0, deployment)
        specs[key] = {'kind': 'evaluate', 'source': source, 'target': source, 'update': 0,
            'deployment': deployment, 'output': str(output), 'baseline': True,
            'command': base + ['evaluate', '--experiment', str(experiment), '--harness', deployment,
                              '--adapter', str(study.e1_adapter(source)), '--output', str(output)]}
    output = study.cell_path(experiment, 'H0', 'H0', 1024, 'H0')
    specs['eval_reused_H0_H0'] = {'kind': 'evaluate', 'source': 'H0', 'target': 'H0', 'update': 1024,
        'deployment': 'H0', 'output': str(output), 'baseline': True,
        'command': base + ['evaluate', '--experiment', str(experiment), '--harness', 'H0',
            '--adapter', str(study.FIGURE3 / 'runs/seed42_H0/checkpoint_001024/actor/adapter'), '--output', str(output)]}
    assert len(specs) == 87
    return specs


def group_ready(ready, slots, evaluations_per_gpu):
    """A training task owns its GPU, while up to two evaluations share one."""
    ready = list(ready)
    groups = []
    while ready and len(groups) < slots:
        first = ready.pop(0)
        group = [first]
        if first[1]['kind'] == 'evaluate' and evaluations_per_gpu == 2:
            peer = next((i for i, (_, spec) in enumerate(ready) if spec['kind'] == 'evaluate'), None)
            if peer is not None:
                group.append(ready.pop(peer))
        groups.append(group)
    return groups


def execute(experiment, evaluations_per_gpu=2, max_gpus=8):
    specs = specs_for(experiment)
    hours, _ = scheduler.recorded_hours()
    # Unlock the longest training branches early, then fill remaining slots with evaluations.
    def priority(item):
        key, spec = item
        if spec['kind'] == 'train':
            return (0, -hours[spec['target']]['train'], key)
        if spec['update'] == 0:
            longest = max((hours[s['target']]['train'] for s in specs.values()
                           if s['kind'] == 'train' and s['source'] == spec['source']), default=0)
            return (1, -longest, key)
        if spec.get('baseline'):
            return (1, 0, key)
        return (2, -spec['update'], key)

    ordered = sorted(specs.items(), key=priority)
    logs = experiment / 'logs'
    logs.mkdir(exist_ok=True)
    path = experiment / 'slurm_scheduler.json'
    with (experiment / '.scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = study.read(path) if path.exists() else {'jobs': {}, 'started_at_utc': datetime.now(UTC).isoformat()}
        state.update(status='running', max_gpus=max_gpus, total_jobs=len(specs),
                     evaluations_per_gpu=evaluations_per_gpu, training_activation_checkpointing=False)
        while True:
            active = {key: job for key, job in state['jobs'].items() if job['status'] == 'submitted'}
            if active:
                ids = ','.join(sorted({job['slurm_id'] for job in active.values()}))
                output = subprocess.check_output(
                    ['sacct', '-X', '-n', '-P', '-j', ids, '-o', 'JobID,State,ExitCode'], text=True)
                accounting = {parts[0]: parts[1:] for line in output.splitlines() if (parts := line.split('|')) and len(parts) >= 3}
                finished = False
                for key, job in active.items():
                    info = accounting.get(job['slurm_id'])
                    if not info:
                        continue
                    status, exit_code = info[:2]
                    job['slurm_state'] = status
                    if status in {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED'}:
                        continue
                    completed = scheduler.complete(specs[key])
                    if not completed and status == 'COMPLETED' and exit_code == '0:0':
                        # Shared result files can become visible after Slurm records success.
                        now = datetime.now(UTC)
                        first_seen = job.setdefault('completion_pending_since_utc', now.isoformat())
                        if (now - datetime.fromisoformat(first_seen)).total_seconds() < 120:
                            continue
                    # One member of a two-task allocation may finish successfully
                    # even if its companion fails. Preserve that completed result.
                    job.update(status='complete' if completed else 'failed',
                               finished_at_utc=datetime.now(UTC).isoformat(), exit_code=exit_code)
                    finished = True
                if finished:
                    hardware.report(experiment)
            failed = [key for key, job in state['jobs'].items() if job['status'] == 'failed']
            if failed:
                # Let already submitted work finish while stopping new submissions.
                state.update(status='failed', failed_jobs=failed)
                write_json(path, state)
                raise RuntimeError(f'Slurm jobs failed: {failed}. See {logs}')
            active_count = len({job['slurm_id'] for job in state['jobs'].values() if job['status'] == 'submitted'})
            if active_count == 0 and all(scheduler.complete(spec) for spec in specs.values()):
                hardware.report(experiment)
                state.update(status='complete', finished_at_utc=datetime.now(UTC).isoformat())
                write_json(path, state)
                print(json.dumps({'status': 'complete', 'jobs': len(specs)}), flush=True)
                return
            validation = experiment / 'validation/checkpoint_20260925/checkpoint.json'
            training_validated = validation.exists() and study.read(validation)['status'] == 'passed'
            ready = [(key, spec) for key, spec in ordered
                     if not scheduler.complete(spec) and state['jobs'].get(key, {}).get('status') != 'submitted'
                     and (spec['kind'] != 'train' or training_validated)
                     and (spec.get('baseline') or scheduler.ready(spec, specs))]
            for group in group_ready(ready, max_gpus - active_count, evaluations_per_gpu):
                key = group[0][0]
                estimates = [hours[spec['target'] if spec['kind'] == 'train' else spec['deployment']]
                             ['train' if spec['kind'] == 'train' else 'first_evaluate'] for _, spec in group]
                estimate = sum(estimates)
                minutes = min(7 * 24 * 60, max(120, math.ceil(estimate * 2 * 60 + 60)))
                command = [sys.executable, '-u', '-m', 'hrp_miniwob.orchestration.run_miniwob_adaptation_pairs_hpc3',
                           '--experiment', str(experiment), '--jobs', *(k for k, _ in group)]
                job_id = subprocess.check_output([
                    'sbatch', '--parsable', '--partition=acd_ue', '--nodes=1', '--ntasks=1',
                    '--cpus-per-task=8', '--gres=gpu:1', '--mem=64G', f'--time={minutes}',
                    f'--job-name=f4-{key}', f'--chdir={study.ROOT}',
                    f'--output={logs}/{key}.slurm-%j.log', '--wrap=' + shlex.join(command),
                ], text=True).strip().split(';')[0]
                for lane, (member, _) in enumerate(group):
                    state['jobs'][member] = {'status': 'submitted', 'slurm_id': job_id,
                        'submitted_at_utc': datetime.now(UTC).isoformat(), 'allocation_gpus': 1,
                        'lane': lane, 'allocation_tasks': [k for k, _ in group]}
                active_count += 1
                write_json(path, state)
                print(json.dumps({'submitted': [k for k, _ in group], 'slurm_id': job_id,
                                  'active_or_pending_gpus': active_count}), flush=True)
            state['updated_at_utc'] = datetime.now(UTC).isoformat()
            write_json(path, state)
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--evaluations-per-gpu', type=int, choices=[1, 2], default=2)
    parser.add_argument('--max-gpus', type=int, choices=range(1, 9), default=8)
    args = parser.parse_args()
    study.prepare(study.DEFAULT)
    if args.execute:
        execute(study.DEFAULT, args.evaluations_per_gpu, args.max_gpus)
    else:
        specs = specs_for(study.DEFAULT)
        print(json.dumps({'jobs': len(specs), 'training': sum(s['kind'] == 'train' for s in specs.values()),
                          'max_gpus': args.max_gpus, 'gpus_per_slurm_job': 1,
                          'evaluations_per_gpu': args.evaluations_per_gpu,
                          'training_activation_checkpointing': False}, indent=2))


if __name__ == '__main__':
    main()
