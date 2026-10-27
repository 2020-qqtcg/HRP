"""Check source replay, weights-only initialization and resumed Exact-PG."""

import argparse
import copy
import json
import os
import socket
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from hrp_miniwob.benchmarks import miniwob_adaptation as experiment
from hrp_miniwob.adapters import miniwob_harnesses as adapter
from hrp_miniwob.adapters.miniwob_v2_batches import exact_batch_update
from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.v2 import FAMILIES
from hrp_miniwob.v2.suite import load_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    source = experiment.source_record()
    started = time.monotonic()
    report = {'status': 'running', 'source': source, 'checks': [],
              'scope': 'Discarded source replay and optimizer acceptance. No held-out task is used.',
              'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    os.environ.update(WORLD_SIZE='1', RANK='0', LOCAL_WORLD_SIZE='1', LOCAL_RANK='0',
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    write_json(args.output, report)
    try:
        with tempfile.TemporaryDirectory(prefix='hrp-adaptation-acceptance-') as tmp, experiment.study.bindings(), experiment.study.base.bindings(training=True):
            from hrp_miniwob.adapters.verl_miniwob import make_worker
            model_path = experiment.read(experiment.SOURCE / 'config.json')['actor_model']
            actor = make_worker(model_path, 'actor', 'cuda:0', 42, 1e-5, 16)
            actor.engine.initialize_adapter(source['adapter'])
            assert not actor.engine.optimizer.state and actor.engine.updates == 0
            assert all(p.dtype == torch.float32 for p in actor.engine.parameters)
            assert actor.engine.policy.metadata['dtype'] == 'bfloat16'
            frozen = adapter.LanguagePolicy(model_path, trainable=False, device='cuda:0', rank=16)
            iid = load_suite(experiment.SOURCE / 'suite', 'test')
            train = load_suite(experiment.SOURCE / 'suite', 'train')
            batch = [min((m for m in train if m.task.family == f), key=lambda m: (m.task.physical_states, len(m.task.actions))) for f in FAMILIES]
            batch.append(batch[0])
            for harness in experiment.HARNESSES:
                bound = adapter.Adapter(actor.engine.policy, adapter.Candidate(experiment.study.candidate_path(harness)), frozen, frozen, str(Path(tmp) / harness))
                try:
                    checked = []
                    for family in FAMILIES:
                        model = next(m for m in iid if m.task.family == family)
                        graph = bound.graph(model, with_reference=False)
                        path = experiment.SOURCE / f'cross_eval/exactpg_test/{harness}/adapted_0/harness_0/policies/{model.task.key}.npz'
                        with np.load(path) as saved:
                            np.testing.assert_array_equal(graph.next_state, saved['next_state'])
                            np.testing.assert_array_equal(graph.task_rewards, saved['rewards'])
                            bank = saved['fixed_bank']
                            indices = bank[np.linspace(0, len(bank) - 1, min(24, len(bank)), dtype=int)]
                            rows = [bound.messages(graph.state(int(i))) for i in indices]
                            with torch.no_grad():
                                lp = actor.engine.policy.batch_log_probabilities([r[0] for r in rows], [r[1] for r in rows]).cpu().numpy()
                                ref = frozen.batch_log_probabilities([r[0] for r in rows], [r[1] for r in rows]).cpu().numpy()
                            error = float(np.max(np.abs(lp - saved['log_policy'][indices])))
                            ref_error = float(np.max(np.abs(ref - np.log(saved['reference'][indices]))))
                            assert max(error, ref_error) <= 1e-3, (harness, family, error, ref_error)
                            checked.append({'task': model.task.key, 'states': len(indices), 'actor_max_log_error': error, 'reference_max_log_error': ref_error})
                    report['checks'].append({'harness': harness, 'saved_probability_replay': checked})
                    write_json(args.output, report)
                    print(json.dumps(report['checks'][-1]), flush=True)
                finally:
                    bound.close()
            # Reset both weights and moments by constructing each branch afresh.
            del actor
            torch.cuda.empty_cache()
            for harness in experiment.HARNESSES:
                actor = make_worker(model_path, 'actor', 'cuda:0', 42, 1e-5, 16)
                actor.engine.initialize_adapter(source['adapter'])
                bound = adapter.Adapter(actor.engine.policy, adapter.Candidate(experiment.study.candidate_path(harness)), frozen, frozen, str(Path(tmp) / harness))
                try:
                    first = exact_batch_update(actor, batch, bound)
                    initial_weights = [p.detach().cpu().clone() for p in actor.engine.parameters]
                    initial_optimizer = copy.deepcopy(actor.engine.optimizer.state_dict())
                    checkpoint = Path(tmp) / f'{harness}_checkpoint'
                    actor.save_checkpoint(str(checkpoint), global_step=1)
                    before_probabilities = [bound.measure(m)[1].copy() for m in batch]
                    second = exact_batch_update(actor, batch, bound)
                    expected = [p.detach().cpu().clone() for p in actor.engine.parameters]
                    expected_gradient = torch.cat([p.grad.detach().cpu().flatten() for p in actor.engine.parameters]).double()
                    actor.load_checkpoint(str(checkpoint), del_local_after_load=False)
                    assert actor.engine.updates == 1 and actor.engine.optimizer.state
                    load_error = max(float((p.detach().cpu() - e).abs().max()) for p, e in zip(actor.engine.parameters, initial_weights, strict=True))
                    assert load_error == 0, load_error
                    loaded_optimizer = actor.engine.optimizer.state_dict()
                    assert initial_optimizer['param_groups'] == loaded_optimizer['param_groups']
                    for k, values in initial_optimizer['state'].items():
                        for name, value in values.items():
                            assert torch.equal(value.cpu(), loaded_optimizer['state'][k][name].cpu()), (k, name)
                    before_resumed = [bound.measure(m)[1] for m in batch]
                    replay_error = max(float(np.abs(a-b).max()) for a, b in zip(before_probabilities, before_resumed, strict=True))
                    assert replay_error <= 1e-3, replay_error
                    resumed = exact_batch_update(actor, batch, bound)
                    error = max(float((p.detach().cpu() - e).abs().max()) for p, e in zip(actor.engine.parameters, expected, strict=True))
                    resumed_gradient = torch.cat([p.grad.detach().cpu().flatten() for p in actor.engine.parameters]).double()
                    # BF16 CUDA backward has small repeat noise even without serialization.
                    # A full task gradient can amplify cancellation noise relative to
                    # E1's single-microbatch checkpoint test. Compare the full-gradient
                    # direction with a control that never reads a checkpoint.
                    with torch.no_grad():
                        for parameter, value in zip(actor.engine.parameters, initial_weights, strict=True):
                            parameter.copy_(value)
                    actor.engine.optimizer.load_state_dict(copy.deepcopy(initial_optimizer))
                    actor.engine.updates = 1
                    baseline = exact_batch_update(actor, batch, bound)
                    baseline_gradient = torch.cat([p.grad.detach().cpu().flatten() for p in actor.engine.parameters]).double()
                    denominator = expected_gradient.norm().clamp_min(1e-12)
                    baseline_error = float((baseline_gradient - expected_gradient).norm() / denominator)
                    resume_gradient_error = max(float((resumed_gradient - g).norm() / denominator) for g in (expected_gradient, baseline_gradient))
                    cosine = float(torch.nn.functional.cosine_similarity(resumed_gradient, expected_gradient, dim=0))
                    tolerance = max(1e-5, 2 * baseline_error)
                    report['resume_diagnostics'] = {'harness': harness, 'load_error': load_error, 'optimizer_state_exact': True,
                        'replay_error': replay_error, 'next_parameter_error': error, 'first': first, 'second': second, 'resumed': resumed,
                        'baseline': baseline, 'baseline_gradient_relative_error': baseline_error,
                        'resume_gradient_relative_error': resume_gradient_error, 'gradient_tolerance': tolerance, 'gradient_cosine': cosine}
                    write_json(args.output, report)
                    assert resume_gradient_error <= tolerance and cosine >= .999, report['resume_diagnostics']
                    assert actor.engine.updates == 2
                    report['checks'].append({'harness': harness, 'weights_only_initialization': 'exact tensors, empty optimizer, update zero',
                                             'first_update': first, 'second_update': second, 'resumed_update': resumed,
                                             'resume_max_parameter_error': error, 'formal_training_updates': 0,
                                             'resume_gradient_relative_error': resume_gradient_error,
                                             'baseline_gradient_relative_error': baseline_error, 'gradient_tolerance': tolerance,
                                             'gradient_cosine': cosine,
                                             'criterion': 'exact weights, moments and forwards; full-gradient cosine >=0.999 and relative error at most twice in-memory BF16 repeat noise, floor 1e-5; not bitwise optimizer determinism'})
                    write_json(args.output, report)
                    print(json.dumps({'harness': harness, 'resume_max_parameter_error': error}), flush=True)
                finally:
                    bound.close()
                del actor
                torch.cuda.empty_cache()
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['wall_seconds'] = time.monotonic() - started
        write_json(args.output, report)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
