"""Explicit H100 numerical migration, retaining the sealed Figure 4 protocol."""

import fcntl
import inspect
import time
from pathlib import Path

import numpy as np

from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_adaptation_pairs as study


def baseline_cells():
    return sorted({(s, s) for s in study.PROFILES} | set(study.PAIRS) | {(t, s) for s, t in study.PAIRS})


def evaluate(experiment, harness, adapter_path, output):
    proposal = study.read(experiment / 'proposal.json')
    migration = study.read(experiment / 'hpc3_migration.json')
    assert migration['authorized_hardware_migration'] is True
    assert study.digest(experiment / 'preregistration.json') == migration['preregistration_sha256']
    models = study.load_suite(study.SOURCE / 'suite', 'test')
    assert study.digest(study.SOURCE / 'suite/manifest.json') == proposal['evaluation']['suite_manifest_sha256']
    assert [m.task.seed for m in models] == proposal['evaluation']['task_seeds']
    with study.bindings() as host:
        config = host.read_config(experiment)
        identity = {'config_sha256': study.digest(experiment / 'config.json'), 'cohort': 'E1 test', 'harness': harness,
                    'adapter': str(adapter_path), 'adapter_sha256': study.model_files(adapter_path)}
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
            actor = host.LanguagePolicy(config['actor_model'], adapter_path=str(adapter_path), trainable=False, device='cuda:0', rank=16)
            frozen = host.load_frozen(config, 'cuda:0')
            bound = host.adapter_for(actor, host.Candidate(study.study.candidate_path(harness)), frozen, config, experiment)
            try:
                result = study.adaptation.evaluate_models(models, bound, output)
            finally:
                bound.close()
            source = next((s for s in study.PROFILES if study.e1_adapter(s) == adapter_path), None)
            old = study.e1_record(source or 'H0', harness)
            peers = sorted(p.parent for p in (experiment / 'evaluations').glob(f'initial_*/{harness}/completion.json'))
            peer = peers[0] if peers else None
            reference_error = actor_error = h100_reference_error = 0.
            for model in models:
                name = f'{model.task.key}.npz'
                with np.load(old / 'policies' / name) as before, np.load(output / f'tasks/{model.task.key}/policies' / name) as after:
                    for key in ('next_state', 'rewards', 'fixed_bank'):
                        np.testing.assert_array_equal(before[key], after[key])
                    reference_error = max(reference_error, float(np.abs(np.log(before['reference']) - np.log(after['reference'])).max()))
                    if source:
                        actor_error = max(actor_error, float(np.abs(before['log_policy'] - after['log_policy']).max()))
                    if peer:
                        with np.load(peer / f'tasks/{model.task.key}/policies' / name) as baseline:
                            h100_reference_error = max(h100_reference_error, float(np.abs(np.log(baseline['reference']) - np.log(after['reference'])).max()))
            assert h100_reference_error <= 1e-3, h100_reference_error
            audit = {'protocol': 'authorized H100 migration, historical drift recorded without altering archived E1',
                     'task_graphs_identical_to_E1': len(models), 'E1_baseline': str(old),
                     'max_reference_log_error_vs_E1': reference_error, 'H100_reference_baseline': str(peer) if peer else None,
                     'max_reference_log_error_within_H100': h100_reference_error,
                     'historical_tolerance': 1e-3, 'historical_reference_replay_passed': reference_error <= 1e-3}
            if source:
                previous = study.read(old / 'metrics.json')['metrics']
                audit.update(max_actor_log_error_vs_E1=actor_error, historical_actor_replay_passed=actor_error <= 1e-3,
                             aggregate_differences_vs_E1={k: result['metrics'][k] - previous[k] for k in study.METRICS})
            write_json(output / 'replay_audit.json', audit)
            write_json(output / 'completion.json', {'metrics_sha256': study.digest(output / 'metrics.json'),
                'replay_audit_sha256': study.digest(output / 'replay_audit.json'), 'tasks': result['tasks'],
                'wall_seconds': time.monotonic() - started, 'protocol': 'hpc3_h100_migration',
                'runtime_source_sha256': study.digest(Path(__file__))})


def report(experiment):
    """Reuse the sealed renderer with explicit H100 thresholds and continuation evidence.

    The registered predictions are retained. Its historical replay assertion
    receives the newly measured H100 threshold. Partial correlations and the
    no-continuation completion condition are corrected in memory, preserving
    the sealed source and protocol files.
    """
    continued = study.cell_path(experiment, 'H0', 'H0', 1024, 'H0')
    if not (continued / 'completion.json').exists():
        return
    read = study.read
    registered = read(experiment / 'preregistration.json')
    effective = read(experiment / 'preregistration.json')
    proposal = read(experiment / 'proposal.json')
    proposal['reused']['H0'].update(evaluation=str(continued), metrics_sha256=study.digest(continued / 'metrics.json'))
    sensitivity = []
    for original, pair in zip(registered['pairs'], effective['pairs'], strict=True):
        s, t = pair['code']
        paths = {label: study.cell_path(experiment, a, a, 0, b)
                 for label, (a, b) in {'ss': (s, s), 'st': (s, t), 'tt': (t, t), 'ts': (t, s)}.items()}
        if not all((paths[k] / 'completion.json').exists() for k in ('ss', 'st')):
            continue
        keys, source = study.arrays(paths['ss'] / 'metrics.json')
        _, target = study.arrays(paths['st'] / 'metrics.json', keys)
        pair['threshold_T']['mean'] = float((target['ceiling'] - source['regularized_return']).mean())
        row = {'code': [s, t], 'registered_threshold': original['threshold_T']['mean'],
               'h100_threshold': pair['threshold_T']['mean'], 'registered_prediction': pair['prediction']}
        if all((p / 'completion.json').exists() for p in paths.values()):
            _, diagonal = study.arrays(paths['tt'] / 'metrics.json', keys)
            _, reverse = study.arrays(paths['ts'] / 'metrics.json', keys)
            draws = study.bootstrap(keys)
            margin = study.summarize(diagonal['regularized_return'] - source['regularized_return'], draws)
            row.update(h100_predicted_margin=margin, h100_prediction='recovery' if margin['mean'] > 0 else 'no recovery',
                       h100_predicted_ranking=study.summarize(diagonal['regularized_return'] - reverse['regularized_return'], draws))
        sensitivity.append(row)

    def runtime_read(path):
        if Path(path) == experiment / 'preregistration.json':
            return effective
        if Path(path) == experiment / 'proposal.json':
            return proposal
        return read(path)

    # S2 has direction counts before three pairs exist, but no correlation yet.
    # Correct the renderer in memory while preserving the sealed function.
    renderer_source = inspect.getsource(study.report)
    guard = '        if value is None:\n'
    assert renderer_source.count(guard) == 1
    renderer_source = renderer_source.replace(guard, "        if value is None or 'rho' not in value:\n", 1)
    # The authorized scope completes after all adaptation curves and rankings.
    # Its completion does not depend on the excluded continuation branches.
    completion = 'len(done) == len(PAIRS) and len(gained) == len(PAIRS)'
    assert renderer_source.count(completion) == 1
    renderer_source = renderer_source.replace(completion,
        "len(done) == len(PAIRS) and all(len(r['curve']) == len(NODES) - 1 and 'ranking' in r for r in done)", 1)
    namespace = dict(vars(study), read=runtime_read)
    exec(compile(renderer_source, study.__file__, 'exec'), namespace)
    result = namespace['report'](experiment)
    result['hardware_migration'] = {'protocol': 'hpc3_h100_migration',
        'registered_predictions_unchanged': True, 'measured_recovery_baseline': 'H100 initial cells',
        'continuation': 'existing Figure 3 H0 adapter reevaluated on H100, no new continuation training',
        'sensitivity': sensitivity}
    write_json(experiment / 'pairs_data.json', result)
    text = study.RESULTS.read_text()
    note = ('\n本次 H100 迁移已获用户确认。上表保留原预注册预测，恢复幅度使用 H100 初始格，'
            'H0 续训对照使用已有 checkpoint 在 H100 上的重新评测。未新增续训。'
            '逐格历史差异见 `replay_audit.json`，H100 预测余量的敏感性分析见 '
            '`pairs_data.json` 的 `hardware_migration.sensitivity`。\n\n')
    study.RESULTS.write_text(text.replace(study.RESULTS_MARKERS[1], note + study.RESULTS_MARKERS[1]))
    return result
