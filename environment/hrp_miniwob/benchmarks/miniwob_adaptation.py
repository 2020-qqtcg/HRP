"""Paired, bounded adaptation after a harness revision, with fresh held-out tasks."""

import argparse
import fcntl
import json
import os
import shutil
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from hrp_miniwob.benchmarks import miniwob_harnesses as study
from hrp_miniwob.benchmarks.miniwob_meta_harness import digest, model_files
from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.v2 import FAMILIES
from hrp_miniwob.v2.browser import Browser, Task
from hrp_miniwob.v2.model import compile_task
from hrp_miniwob.v2.suite import is_ood, verify_success_path
from src.hrp.initial_distribution import discounted_visits

ROOT = study.ROOT
SOURCE = ROOT / 'environment/artifacts/v2/qwen35_2b_h0_h7_exactpg_seed42_2026-09-11'
DEFAULT = ROOT / 'environment/artifacts/v2/figure3_h0_h3_exactpg_2026-09-14'
NODES = [0, 32, 64, 128, 256, 512, 1024]
SEEDS = [42]
HARNESSES = ['H0', 'H3']
BASE_SOURCES = study.source_hashes


def read(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    paths = [Path(__file__), ROOT / 'environment/hrp_miniwob/orchestration/run_miniwob_adaptation.py',
             ROOT / 'environment/hrp_miniwob/benchmarks/miniwob_adaptation_report.py',
             ROOT / 'environment/hrp_miniwob/validation/check_miniwob_adaptation.py',
             ROOT / 'environment/hrp_miniwob/validation/check_miniwob_v2_batches.py']
    return BASE_SOURCES() | {str(p.relative_to(ROOT)): digest(p) for p in paths}


@contextmanager
def bindings(training=False):
    old_sources = study.source_hashes
    study.source_hashes = source_hashes
    try:
        with study.bindings(), study.base.bindings(training=training):
            host = study.base.runner
            old_eval = host.evaluate_models
            host.evaluate_models = evaluate_models
            try:
                yield
            finally:
                host.evaluate_models = old_eval
    finally:
        study.source_hashes = old_sources


def source_record():
    selection = read(SOURCE / 'runs/exactpg_H0/e1_selection.json')
    run_path = SOURCE / 'runs/exactpg_H0/run.json'
    run = read(run_path)
    assert selection['updates'] == 1024 and run['status'] == 'complete'
    assert digest(run_path) == selection['run_sha256']
    entry = next(c for c in run['checkpoints'] if c['updates'] == 1024)
    directory = run_path.parent / entry['directory']
    for relative, expected in entry['files_sha256'].items():
        assert digest(directory / relative) == expected, relative
    return {'experiment': str(SOURCE), 'checkpoint': str(directory), 'adapter': selection['adapter'],
            'source_training_seed': 42, 'updates': 1024, 'run_sha256': digest(run_path),
            'selection_sha256': digest(run_path.parent / 'e1_selection.json'),
            'adapter_sha256': model_files(Path(selection['adapter'])),
            'checkpoint_files_sha256': entry['files_sha256'], 'config_sha256': digest(SOURCE / 'config.json')}


def prepare_data(experiment):
    """Resume native task generation, excluding all retained V2 suite contents."""
    experiment.mkdir(parents=True, exist_ok=True)
    proposal = {'source': source_record(), 'adaptation_seeds': SEEDS, 'checkpoint_updates': NODES,
                'updates_per_branch': 1024, 'batch_size': 4, 'optimizer_state': 'reset for both branches',
                'harnesses': HARNESSES, 'new_calibration_per_family': 24, 'new_test_per_family': 192,
                'start_seed': 29100000, 'physical_gpus': [1, 2, 3], 'reserved_e1_gpu': 0,
                'training_distribution': 'unchanged E1 tasks and 0.5 native + 0.5 legal-prefix coverage',
                'evaluation_distribution': 'common native IID task starts; exact policy evaluation',
                'selection': 'predeclared final budget primary; all nodes retained; calibration-only optional selection'}
    proposal_path = experiment / 'proposal.json'
    if proposal_path.exists():
        assert read(proposal_path) == proposal
    else:
        write_json(proposal_path, proposal)
    if (experiment / 'suite/manifest.json').exists():
        return
    exclusion_path = experiment / 'exclusions.json'
    if not exclusion_path.exists():
        contents, seeds, manifests = set(), set(), {}
        for path in sorted((ROOT / 'environment/artifacts').rglob('manifest.json')):
            if any(part in {'source_snapshot', 'runtime_source', 'source', 'vendor'} for part in path.parts) or experiment in path.parents:
                continue
            record = read(path)
            if record.get('version') == 'v2' and 'entries' in record:
                manifests[str(path)] = digest(path)
                for entry in record['entries']:
                    contents.add(entry['content_sha256'])
                    seeds.add(entry['seed'])
            # V1 checkbox public contents use a different hash serialization.
            elif isinstance(record.get('tasks'), list):
                for entry in record['tasks']:
                    if not isinstance(entry, dict) or 'json' not in entry:
                        continue
                    payload = read(path.parent / entry['json'])
                    task = payload.get('task', {})
                    if 'labels' in task and 'instruction' in task:
                        native = Task(task['seed'], 'click-checkboxes', task['instruction'], {'labels': task['labels']}, 0)
                        contents.add(native.fingerprint)
                        seeds.add(task['seed'])
                        manifests[str(path)] = digest(path)
        write_json(exclusion_path, {'manifest_sha256': manifests, 'content_sha256': sorted(contents), 'seeds': sorted(seeds)})
    exclusions = read(exclusion_path)
    forbidden, forbidden_seeds = set(exclusions['content_sha256']), set(exclusions['seeds'])
    for path, expected in exclusions['manifest_sha256'].items():
        assert digest(path) == expected, path
    suite = experiment / 'suite'
    (suite / 'tasks').mkdir(parents=True, exist_ok=True)
    origin = read(SOURCE / 'suite/manifest.json')
    training = [entry for entry in origin['entries'] if entry['split'] == 'train']
    for entry in training:
        source = SOURCE / 'suite' / entry['json']
        assert digest(source) == entry['sha256']
        target = suite / entry['json']
        if not target.exists():
            shutil.copyfile(source, target)
    journal = experiment / 'new_tasks.jsonl'
    fresh = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    seen = {e['content_sha256'] for e in fresh}
    assert not seen & forbidden and len(seen) == len(fresh)
    with Browser(horizon=10) as browser:
        for index, family in enumerate(FAMILIES):
            counts = Counter(e['split'] for e in fresh if e['family'] == family)
            existing = [e['seed'] for e in fresh if e['family'] == family]
            seed = max(existing) + 1 if existing else proposal['start_seed'] + index * 1000000
            scanned = 0
            while counts['calibration'] < 24 or counts['test'] < 192:
                if scanned >= 100000:
                    raise RuntimeError(f'Fresh IID task scan exhausted for {family}')
                task = browser.reset(seed, family).task
                seed += 1
                scanned += 1
                if task.seed in forbidden_seeds or task.fingerprint in forbidden | seen or is_ood(task):
                    continue
                model = compile_task(task, 10)
                steps = verify_success_path(browser, model)
                split = 'calibration' if counts['calibration'] < 24 else 'test'
                relative = f'tasks/{task.key}.json'
                write_json(suite / relative, asdict(task))
                entry = {'family': family, 'seed': task.seed, 'split': split, 'json': relative,
                         'sha256': digest(suite / relative), 'content_sha256': task.fingerprint,
                         'verified_success_steps': steps, 'actions': len(task.actions), 'physical_states': task.physical_states}
                with journal.open('a') as stream:
                    stream.write(json.dumps(entry) + '\n')
                    stream.flush()
                fresh.append(entry)
                seen.add(task.fingerprint)
                counts[split] += 1
                write_json(experiment / 'preparation.json', {'status': 'running', 'family': family, 'counts': dict(counts), 'new_tasks': len(fresh)})
            print(json.dumps({'family': family, 'fresh_tasks': dict(counts)}), flush=True)
    entries = training + fresh
    assert len(fresh) == 648 and len({e['content_sha256'] for e in entries}) == len(entries)
    assert not {e['seed'] for e in fresh} & forbidden_seeds
    manifest = {'version': 'v2', 'status': 'complete', 'profile': 'figure3', 'horizon': 10,
                'families': list(FAMILIES), 'entries': entries, 'source_suite_sha256': digest(SOURCE / 'suite/manifest.json'),
                'exclusions_sha256': digest(exclusion_path), 'validation': 'Every fresh task has native success replay; native kernels unchanged from E1'}
    write_json(suite / 'manifest.json', manifest)
    partitions = {s: [e['seed'] for e in entries if e['split'] == s] for s in ('search', 'train', 'calibration', 'test', 'ood')}
    write_json(experiment / 'splits.json', partitions)
    shutil.copyfile(SOURCE / 'train_public.json', experiment / 'train_public.json')
    shutil.copyfile(SOURCE / 'train_probe.json', experiment / 'train_probe.json')
    write_json(experiment / 'preparation.json', {'status': 'complete', 'tasks': {s: len(v) for s, v in partitions.items()},
               'old_content_overlap': 0, 'old_seed_overlap': 0, 'old_manifests': len(exclusions['manifest_sha256'])})


def seal(experiment):
    assert read(experiment / 'preparation.json')['status'] == 'complete'
    config = read(SOURCE / 'config.json')
    config.update(source_sha256_at_preparation=source_hashes(), harnesses=HARNESSES,
                  figure3=read(experiment / 'proposal.json'), adaptation_protocol='h0_h3_paired_v1')
    for relative, key in [('suite/manifest.json', 'suite_manifest_sha256'), ('splits.json', 'splits_sha256'),
                          ('train_public.json', 'train_public_sha256'), ('train_probe.json', 'train_probe_sha256')]:
        config[key] = digest(experiment / relative)
    config['split_counts'] = {s: len(v) for s, v in read(experiment / 'splits.json').items()}
    path = experiment / 'config.json'
    if path.exists():
        assert read(path) == config, 'Sealed Figure 3 configuration changed'
        return
    for relative in config['source_sha256_at_preparation']:
        target = experiment / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    write_json(path, config)
    write_json(experiment / 'manifest.json', {'status': 'sealed', 'created_at_utc': datetime.now(UTC).isoformat(),
               'config_sha256': digest(path), 'proposal_sha256': digest(experiment / 'proposal.json'),
               'source': source_record(), 'primary_split': 'test', 'bootstrap_seed': 4203, 'bootstrap_samples': 2000,
               'uncertainty': 'paired task bootstrap within family only; source and adaptation seed 42; no training-seed uncertainty'})


def evaluate_models(models, bound, output):
    """Reuse exact E1 metrics, adding resumable task rows and execution accounting."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in models:
        task_output = output / 'tasks' / model.task.key
        if (task_output / 'metrics.json').exists() and (task_output / 'resource.json').exists():
            report = read(task_output / 'metrics.json')
            assert report['rows'][0]['content_sha256'] == model.task.fingerprint
        else:
            started = time.monotonic()
            before = bound.policy.forward_observations, bound.policy.forward_prompt_tokens, bound.frozen.forwards, bound.frozen.prompt_tokens
            report = study.adapter.evaluate_models([model], bound, task_output)
            graph = bound.graph(model)
            with np.load(task_output / 'policies' / f'{model.task.key}.npz') as table:
                policy = np.exp(table['log_policy'])
            start = np.zeros(len(graph.next_state))
            start[graph.initial_state] = 1
            visits = discounted_visits(graph.next_state, policy, start, 1.)
            clicks = np.zeros_like(policy)
            for s in range(len(graph.next_state)):
                physical = graph.state(s).physical
                for a, action in enumerate(graph.task.actions):
                    if action == 'submit' or action.startswith('click:'):
                        clicks[s, a] = 1
                    elif action.startswith('batch:'):
                        first = 'click:' + action[6:].split(',')[0]
                        clicks[s, a] = 1 + int(model.physical_next[physical, model.task.actions.index(first)] >= 0)
            write_json(task_output / 'resource.json', {'wall_seconds': time.monotonic() - started,
                       'actor_forwards': bound.policy.forward_observations - before[0],
                       'actor_prompt_tokens': bound.policy.forward_prompt_tokens - before[1],
                       'reference_forwards': bound.frozen.forwards - before[2],
                       'reference_prompt_tokens': bound.frozen.prompt_tokens - before[3],
                       'expected_actor_decisions_native': float(visits.sum()),
                       'expected_native_clicks_native': float(visits @ (policy * clicks).sum(1)),
                       'accounting_scope': 'exact expected task execution; no rollout browser calls for this measurement'})
        row = report['rows'][0] | read(task_output / 'resource.json')
        row.pop('accounting_scope')
        rows.append(row)
        write_json(output / 'progress.json', {'tasks': len(rows), 'total': len(models)})
    keys = [key for key in rows[0] if key not in ('task_seed', 'family', 'content_sha256')]
    families = {f: {k: float(np.mean([r[k] for r in rows if r['family'] == f])) for k in keys} for f in FAMILIES}
    report = {'tasks': len(rows), 'harness': bound.candidate.profile, 'rows': rows, 'per_family': families,
              'metrics': {k: float(np.mean([f[k] for f in families.values()])) for k in keys},
              'aggregation': 'equal task within family, equal family', 'mismatch': 'independent occupancy-weighted KL(policy || optimal_policy)',
              'accounting': {k: sum(r[k] for r in rows) for k in ('wall_seconds', 'actor_forwards', 'actor_prompt_tokens', 'reference_forwards', 'reference_prompt_tokens')}}
    write_json(output / 'metrics.json', report)
    return report


def train(experiment, harness, seed):
    proposal = read(experiment / 'proposal.json')
    assert harness in HARNESSES and seed in proposal['adaptation_seeds']
    output = experiment / 'runs' / f'seed{seed}_{harness}'
    args = argparse.Namespace(experiment=experiment, output=output, candidate=study.candidate_path(harness), device='cuda:0',
        method='exactpg', seed=seed, batch_size=4, updates=1024, checkpoint_interval=64,
        checkpoint_updates=NODES, rollout_episodes=None, limit=None, resume=True,
        initial_adapter=Path(proposal['source']['adapter']))
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a') as lock, bindings(training=True):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study.base.train(args)
        study.base.select_checkpoint(output)


def evaluate(experiment, harness, adapter_path, output):
    with bindings():
        host = study.base.runner
        config = host.read_config(experiment)
        identity = {'config_sha256': digest(experiment / 'config.json'), 'harness': harness,
                    'adapter_sha256': model_files(adapter_path), 'adapter': str(adapter_path), 'split': 'test'}
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
                result = evaluate_models(host.models_for(experiment, 'test'), bound, output)
                write_json(output / 'completion.json', {'metrics_sha256': digest(output / 'metrics.json'),
                           'wall_seconds': time.monotonic() - started, 'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
                           'model_decisions_are_exact_expectations': True, 'tasks': result['tasks']})
            finally:
                bound.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare-data', 'seal', 'train', 'evaluate'])
    parser.add_argument('--experiment', type=Path, default=DEFAULT)
    parser.add_argument('--harness', choices=HARNESSES)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'train' and (args.harness is None or args.seed is None):
        parser.error('train requires --harness and --seed')
    if args.command == 'evaluate' and any(v is None for v in (args.harness, args.adapter, args.output)):
        parser.error('evaluate requires --harness, --adapter and --output')
    torch.set_num_threads(2)
    experiment = study.base.require_v2_path(args.experiment)
    if args.command == 'prepare-data':
        prepare_data(experiment)
    elif args.command == 'seal':
        seal(experiment)
    elif args.command == 'train':
        train(experiment, args.harness, args.seed)
    else:
        evaluate(experiment, args.harness, args.adapter, args.output)


if __name__ == '__main__':
    main()
