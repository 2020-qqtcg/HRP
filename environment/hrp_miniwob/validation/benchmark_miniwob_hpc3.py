"""Discarded H100 timing and numerical checks for Figure 4 execution changes."""

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_adaptation_pairs as study
from hrp_miniwob.validation.check_miniwob_harnesses_gpu import checkpoint_gradient_check


def checkpoint(args):
    from hrp_miniwob.adapters.miniwob_v2_batches import exact_batch_update
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    os.environ.update(WORLD_SIZE='1', RANK='0', LOCAL_WORLD_SIZE='1', LOCAL_RANK='0',
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    report = {'status': 'running', 'scope': 'discarded optimization validation', 'harnesses': {}}
    with study.bindings(training=True) as host:
        from hrp_miniwob.adapters.verl_miniwob import make_worker
        config = host.read_config(args.experiment)
        actor = make_worker(config['actor_model'], 'actor', 'cuda:0', 42, 1e-5, 16)
        frozen = host.load_frozen(config, 'cuda:0')
        training = study.load_suite(args.experiment / 'suite', 'train')
        for harness in args.harnesses:
            actor.engine.optimizer.state.clear()
            actor.engine.updates = 0
            actor.engine.initialize_adapter(study.e1_adapter('H4'))
            row = report['harnesses'][harness] = {'status': 'running', 'stress': []}
            write_json(args.output / 'checkpoint.json', report)
            with tempfile.TemporaryDirectory(prefix='hrp-checkpoint-') as cache:
                bound = host.MetaAdapter(actor.engine.policy, host.Candidate(study.study.candidate_path(harness)),
                                         *frozen, Path(cache))
                try:
                    row['gradient_check'] = checkpoint_gradient_check(actor, bound, training)
                    # Scan every training task's largest public controller state.
                    # The selected prompt is longest after the actual tokenizer.
                    longest = {}
                    for model in training:
                        graph = bound.graph(model, with_reference=False)
                        state = graph.state(graph.index(10, graph.task.memories - 1, model.task.physical_states - 1))
                        prompt, codes = bound.messages(state)
                        length = len(actor.engine.policy.tokenizer.encode(actor.engine.policy.chat_text(prompt), add_special_tokens=False))
                        if model.task.family not in longest or length > longest[model.task.family][0]:
                            longest[model.task.family] = (length, model, prompt, codes)
                        bound.graphs.clear()
                    for length, model, prompt, codes in longest.values():
                        item = {'task': model.task.key, 'tokens': length, 'modes': {}}
                        for enabled in (True, False):
                            if enabled:
                                actor.engine.policy.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
                            else:
                                actor.engine.policy.model.gradient_checkpointing_disable()
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                            timings = []
                            for repeat in range(4):
                                actor.engine.optimizer_zero_grad()
                                torch.cuda.synchronize()
                                tick = time.monotonic()
                                lp = actor.engine.policy.batch_log_probabilities([prompt] * 8, [codes] * 8)
                                (lp.exp() * torch.arange(len(codes), device=lp.device)).sum(-1).mean().backward()
                                torch.cuda.synchronize()
                                if repeat:
                                    timings.append(time.monotonic() - tick)
                                del lp
                            item['modes'][str(enabled)] = {'seconds': timings, 'median_seconds': float(np.median(timings)),
                                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                                'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}
                            actor.engine.optimizer_zero_grad()
                        item['speedup'] = item['modes']['True']['median_seconds'] / item['modes']['False']['median_seconds']
                        row['stress'].append(item)
                        write_json(args.output / 'checkpoint.json', report)
                    # A real, discarded update exercises the unchanged replay assertion.
                    chosen = [min((m for m in training if m.task.family == f), key=lambda m: len(m.task.actions))
                              for f in study.FAMILIES]
                    chosen.append(next(m for m in training if m.task.family == study.FAMILIES[0] and m not in chosen))
                    tick = time.monotonic()
                    row['discarded_update'] = exact_batch_update(actor, chosen, bound)
                    row['discarded_update']['seconds'] = time.monotonic() - tick
                    row['status'] = 'passed'
                except Exception as error:
                    row.update(status='failed', error=repr(error))
                    raise
                finally:
                    bound.close()
                    write_json(args.output / 'checkpoint.json', report)
                print(json.dumps({'harness': harness, 'status': row['status'], 'speedups': [r['speedup'] for r in row['stress']]}), flush=True)
    report['status'] = 'passed'
    write_json(args.output / 'checkpoint.json', report)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def evaluate_child(args):
    models = study.load_suite(study.SOURCE / 'suite', 'test')
    selected = []
    for family in study.FAMILIES:
        group = sorted((m for m in models if m.task.family == family), key=lambda m: (len(m.task.actions), m.task.seed))
        selected.extend([group[len(group) // 2], group[-1]])
    with study.bindings() as host:
        config = host.read_config(args.experiment)
        actor = host.LanguagePolicy(config['actor_model'], adapter_path=str(study.e1_adapter('H3')),
                                    trainable=False, device='cuda:0', rank=16)
        frozen = host.load_frozen(config, 'cuda:0')
        bound = host.adapter_for(actor, host.Candidate(study.study.candidate_path(args.harnesses[0])), frozen, config, args.experiment)
        try:
            # Warm each process's kernels and every reference-cache entry before timing.
            study.adaptation.evaluate_models(selected, bound, args.output / 'warmup')
            (args.output / 'ready').touch()
            deadline = time.monotonic() + 900
            while not (args.output.parent / 'go').exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('Benchmark start barrier')
                time.sleep(.1)
            torch.cuda.synchronize()
            started = time.monotonic()
            metrics = study.adaptation.evaluate_models(selected, bound, args.output / 'measured')
            torch.cuda.synchronize()
            write_json(args.output / 'timing.json', {'seconds': time.monotonic() - started,
                'tasks': metrics['tasks'], 'task_seeds': [m.task.seed for m in selected],
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30})
        finally:
            bound.close()


def evaluation(args):
    report = {'status': 'running', 'scope': 'six E1 test tasks per process, discarded performance comparison', 'phases': {}}
    for name, count in (('single', 1), ('dual', 2)):
        phase = args.output / name
        phase.mkdir()
        children, logs = [], []
        for lane in range(count):
            output = phase / str(lane)
            log = (phase / f'{lane}.log').open('w')
            logs.append(log)
            command = ['bash', str(study.ROOT / 'scripts/miniwob_slurm_worker.sh'), sys.executable, '-m', __spec__.name,
                       '--mode', 'eval-child', '--experiment', str(args.experiment), '--output', str(output),
                       '--harnesses', args.harnesses[0]]
            children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                             env=os.environ | {'HRP_GPU_LANE': str(lane)}))
        try:
            deadline = time.monotonic() + 1200
            while not all((phase / str(i) / 'ready').exists() for i in range(count)):
                if any(c.poll() is not None for c in children):
                    raise RuntimeError(f'{name} benchmark child exited before warmup. See {phase}')
                if time.monotonic() > deadline:
                    raise TimeoutError('Benchmark warmup')
                time.sleep(1)
            tick = time.monotonic()
            (phase / 'go').touch()
            for child in children:
                if child.wait(timeout=1200):
                    raise RuntimeError(f'{name} benchmark child failed. See {phase}')
            timings = [study.read(phase / str(i) / 'timing.json') for i in range(count)]
            report['phases'][name] = {'processes': count, 'timings': timings, 'including_exit_seconds': time.monotonic() - tick,
                                      'tasks_per_second': 6 * count / max(t['seconds'] for t in timings)}
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for log in logs:
                log.close()
        write_json(args.output / 'evaluation.json', report)
    maximum = 0.
    for path in (args.output / 'single/0/measured/tasks').glob('*/policies/*.npz'):
        relative = path.relative_to(args.output / 'single/0/measured')
        for lane in range(2):
            with np.load(path) as baseline, np.load(args.output / f'dual/{lane}/measured' / relative) as parallel:
                for key in ('log_policy', 'reference'):
                    maximum = max(maximum, float(np.abs(baseline[key] - parallel[key]).max()))
    assert maximum <= 1e-3, maximum
    report.update(status='passed', max_probability_array_difference=maximum,
                  throughput_speedup=report['phases']['dual']['tasks_per_second'] / report['phases']['single']['tasks_per_second'])
    write_json(args.output / 'evaluation.json', report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('checkpoint', 'evaluation', 'eval-child'), required=True)
    parser.add_argument('--experiment', type=Path, default=study.DEFAULT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--harnesses', nargs='+', default=['H1', 'H0', 'H2', 'H3', 'H5'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    {'checkpoint': checkpoint, 'evaluation': evaluation, 'eval-child': evaluate_child}[args.mode](args)


if __name__ == '__main__':
    main()
