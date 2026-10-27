"""Run one training task or two evaluation tasks in one Slurm GPU allocation."""

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_adaptation_pairs as study
from hrp_miniwob.benchmarks import miniwob_adaptation_pairs_hpc3 as hardware
from hrp_miniwob.orchestration.run_miniwob_adaptation_pairs_slurm import specs_for


class H100Policy(study.study.adapter.LanguagePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation_checkpointing=False, **kwargs)
        if any(p.requires_grad for p in self.model.parameters()):
            self.metadata['activation_checkpointing'] = 'disabled on H100 after gradient and memory validation'


def run_task(experiment, key):
    spec = specs_for(experiment)[key]
    output = Path(spec['output'])
    output.mkdir(parents=True, exist_ok=True)
    if spec['kind'] == 'train':
        validation = experiment / 'validation/checkpoint_20260925/checkpoint.json'
        result = study.read(validation)
        assert result['status'] == 'passed' and result['harnesses'][spec['target']]['status'] == 'passed'
        study.study.adapter.LanguagePolicy = H100Policy
    else:
        validation = experiment / 'validation/evaluation_20260925/evaluation.json'
        assert study.read(validation)['status'] == 'passed'
    write_json(output / 'hpc3_runtime.json', {
        'started_at_utc': datetime.now(UTC).isoformat(), 'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'gpu': torch.cuda.get_device_name(0), 'lane': int(os.environ['HRP_GPU_LANE']),
        'activation_checkpointing': False if spec['kind'] == 'train' else 'not applicable',
        'validation': str(validation), 'validation_sha256': study.digest(validation),
        'runtime_source_sha256': study.digest(Path(__file__)),
        'sealed_source_sha256': study.source_hashes(),
    })
    if spec['kind'] == 'train':
        study.train(experiment, spec['source'], spec['target'])
    else:
        command = spec['command']
        adapter = Path(command[command.index('--adapter') + 1])
        hardware.evaluate(experiment, spec['deployment'], adapter, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=study.DEFAULT)
    parser.add_argument('--jobs', nargs='+', required=True)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.child:
        assert len(args.jobs) == 1
        run_task(args.experiment, args.jobs[0])
        return
    specs = specs_for(args.experiment)
    assert 1 <= len(args.jobs) <= 2
    assert len(args.jobs) == 1 or all(specs[k]['kind'] == 'evaluate' for k in args.jobs)
    children, logs = [], []
    try:
        for lane, key in enumerate(args.jobs):
            log = (args.experiment / f'logs/{key}.slurm-{os.environ["SLURM_JOB_ID"]}.task.log').open('a')
            logs.append(log)
            command = ['bash', str(study.ROOT / 'scripts/miniwob_slurm_worker.sh'), sys.executable,
                       '-u', '-m', __spec__.name, '--experiment', str(args.experiment), '--child', '--jobs', key]
            children.append(subprocess.Popen(command, env=os.environ | {'HRP_GPU_LANE': str(lane)},
                                              stdout=log, stderr=subprocess.STDOUT))
        statuses = [child.wait() for child in children]
        if any(statuses):
            raise RuntimeError(f'Allocation task exit codes: {dict(zip(args.jobs, statuses))}')
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
