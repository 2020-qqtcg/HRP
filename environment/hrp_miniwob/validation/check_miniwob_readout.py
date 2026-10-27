"""Acceptance checks for the readout-equivalence experiment."""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from hrp_miniwob.adapters.miniwob_readout import (
    RENDERINGS,
    ReadoutRenderer,
    align_encoded_columns,
    decode_code,
    encoded_column_order,
    mapping_record,
)
from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_readout as study
from hrp_miniwob.oracle import solve
from hrp_miniwob.suite import reachable_states
from hrp_miniwob.v2.browser import Browser


def normalized_observation(rendered):
    observation = json.loads(rendered.messages[1]["content"])
    actions = observation.pop("actions")
    if isinstance(actions, list):
        action_map = {row["code"]: row["action"] for row in actions}
    else:
        action_map = actions
    semantic = [action_map[code] for code in rendered.semantic_codes]
    return dict(observation), semantic


def cpu_acceptance(experiment):
    experiment = Path(experiment).resolve()
    models = study.load_models(experiment, "calibration")
    representatives = [next(model for model in models if model.task.family == family) for family in study.FAMILIES]
    checks = 0
    maximum = defaultdict_float()
    execution_checks = 0
    for model in representatives:
        state = model.state(model.initial_state)
        canonical = ReadoutRenderer("R0").render(state)
        canonical_observation, canonical_actions = normalized_observation(canonical)
        n = len(model.task.actions)
        for rendering in RENDERINGS:
            record = mapping_record(n, rendering)
            if sorted(record["semantic_to_code_index"]) != list(range(n)):
                raise AssertionError("Renderer code assignment is not a bijection")
            rendered = ReadoutRenderer(rendering).render(state)
            observation, actions = normalized_observation(rendered)
            if observation != canonical_observation or actions != canonical_actions:
                raise AssertionError(f"{rendering} does not reversibly preserve public information")
            if rendered.messages[0] != canonical.messages[0]:
                raise AssertionError("A presentation variant changed the H0 system message")
            for code, action in zip(rendered.semantic_codes, model.task.actions, strict=True):
                if decode_code(state, rendering, code) != action:
                    raise AssertionError("Displayed mapping and deployed decoder disagree")

            rng = np.random.default_rng(7000 + n)
            reference = rng.dirichlet(np.ones(n), size=len(model.next_state))
            order = encoded_column_order(n, rendering)
            result = solve(model.next_state[:, order], model.task_rewards[:, order], reference[:, order], study.GAMMA, study.TAU)
            aligned = align_encoded_columns(result.policy, rendering)
            baseline = solve(model.next_state, model.task_rewards, reference, study.GAMMA, study.TAU)
            maximum["semantic_optimal_probability"] = max(maximum["semantic_optimal_probability"], float(np.max(np.abs(aligned - baseline.policy))))
            maximum["value"] = max(maximum["value"], float(np.max(np.abs(result.value - baseline.value))))
            maximum["reference_pushforward"] = max(maximum["reference_pushforward"], float(np.max(np.abs(reference[:, order][:, np.asarray(record["semantic_to_code_index"])] - reference))))
            checks += 1

        # Browser execution is checked after decoding every initial-state action
        # under every rendering.  This validates effects, not just output names.
        with Browser(horizon=study.HORIZON) as browser:
            for rendering in RENDERINGS:
                rendered = ReadoutRenderer(rendering).render(state)
                for column, code in enumerate(rendered.semantic_codes):
                    restored = browser.reset(model.task.seed, model.task.family)
                    action = decode_code(state, rendering, code)
                    result = browser.step(action)
                    expected_next = int(model.next_state[model.initial_state, column])
                    actual_next = -1 if result.state.terminated else (result.state.remaining - 1) * model.task.physical_states + result.state.physical
                    if actual_next != expected_next or result.reward != model.task_rewards[model.initial_state, column]:
                        raise AssertionError("Decoded code changed native execution semantics")
                    if restored.task != model.task:
                        raise AssertionError("Native task changed during execution acceptance")
                    execution_checks += 1

    # Mapping control: inverse renaming recovers columns exactly, while applying
    # the visible permutation without decoding must be caught.
    synthetic = np.arange(20, dtype=float).reshape(4, 5)
    permutation = np.asarray(mapping_record(5, "R4")["semantic_to_code_index"])
    renamed = np.empty_like(synthetic)
    renamed[:, permutation] = synthetic
    recovered = renamed[:, permutation]
    if not np.array_equal(recovered, synthetic):
        raise AssertionError("Post-selection inverse rename failed")
    wrong_error = float(np.max(np.abs(renamed - synthetic)))
    if wrong_error == 0:
        raise AssertionError("The erroneous column-alignment failure case was not detected")

    # Known target-only change in a one-step control reverses centered energy.
    rewards = np.asarray([[0.1, 0.4, -0.2]])
    next_state = np.full((1, 3), -1, dtype=np.int64)
    reference = np.asarray([[.2, .3, .5]])
    original = solve(next_state, rewards, reference, study.GAMMA, study.TAU)
    target = study.TAU * (original.log_policy - original.log_policy.mean(1, keepdims=True))
    logits = np.log(reference) - 2 * target / study.TAU
    logits -= logits.max(1, keepdims=True)
    changed_reference = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    changed = solve(next_state, rewards, changed_reference, study.GAMMA, study.TAU)
    changed_target = study.TAU * (changed.log_policy - changed.log_policy.mean(1, keepdims=True))
    positive_error = float(np.max(np.abs(changed_target + target)))
    if positive_error > 1e-12:
        raise AssertionError("Known target-only control did not reverse the one-step optimum")

    if max(maximum.values(), default=0) > 1e-12:
        raise AssertionError("Independent encoded solves failed semantic equivalence")
    report = {
        "status": "passed",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "representative_tasks": [model.task.key for model in representatives],
        "rendering_task_checks": checks,
        "native_execution_checks": execution_checks,
        "maximum_errors": dict(maximum),
        "wrong_alignment_error": wrong_error,
        "target_positive_control_error": positive_error,
        "criteria": {"equivalence": 1e-10, "implemented_acceptance": 1e-12},
    }
    output = experiment / "acceptance"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "cpu.json", report)
    print(json.dumps(report), flush=True)


def defaultdict_float():
    from collections import defaultdict
    return defaultdict(float)


def gpu_preflight(experiment, device):
    experiment = Path(experiment).resolve()
    study.validate_model_specs()
    models = study.load_models(experiment, "calibration")
    representatives = [next(model for model in models if model.task.family == family) for family in study.FAMILIES]
    report = {
        "status": "running",
        "device": device,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "models": {},
    }
    output = experiment / "acceptance"
    output.mkdir(parents=True, exist_ok=True)
    for model_id in study.MODEL_SPECS:
        started = time.monotonic()
        policy = study.load_policy(model_id, device)
        max_normalization = 0.0
        max_repeat = 0.0
        states = 0
        try:
            for model in representatives:
                bank = reachable_states(model)
                selected = bank[np.linspace(0, len(bank) - 1, min(16, len(bank)), dtype=int)]
                for rendering in RENDERINGS:
                    renderer = ReadoutRenderer(rendering)
                    rendered = [renderer.render(model.state(int(index))) for index in selected]
                    prompts = [item.messages for item in rendered]
                    codes = [item.semantic_codes for item in rendered]
                    with torch.no_grad():
                        first = policy.batch_log_probabilities(prompts, codes).detach().cpu().numpy()
                    max_normalization = max(max_normalization, float(np.max(np.abs(np.exp(first).sum(1) - 1))))
                    states += len(selected)
                rendered = [ReadoutRenderer("R0").render(model.state(int(index))) for index in selected]
                with torch.no_grad():
                    first = policy.batch_log_probabilities([x.messages for x in rendered], [x.semantic_codes for x in rendered]).detach().cpu().numpy()
                    second = policy.batch_log_probabilities([x.messages for x in rendered], [x.semantic_codes for x in rendered]).detach().cpu().numpy()
                max_repeat = max(max_repeat, float(np.max(np.abs(first - second))))
            if max_normalization > 1e-12 or max_repeat > 1e-12:
                raise AssertionError("GPU preflight failed normalization or deterministic repeatability")
            report["models"][model_id] = {
                "status": "passed",
                "states_across_renderings": states,
                "max_probability_normalization_error": max_normalization,
                "max_repeated_log_probability_error": max_repeat,
                "forward_state_rows": policy.forward_observations,
                "prompt_tokens": policy.forward_prompt_tokens,
                "forward_microbatches": policy.forward_microbatches,
                "wall_seconds": time.monotonic() - started,
                "metadata": policy.metadata,
            }
        finally:
            del policy
            gc.collect()
            torch.cuda.empty_cache()
    report["status"] = "passed"
    report["completed_at_utc"] = datetime.now(UTC).isoformat()
    report["historical_provenance"] = {
        "h0_source_calibration": str(study.H0_ADAPTER.parent.parent / "calibration/metrics.json"),
        "figure3_results": str(study.ROOT / "environment/artifacts/v2/figure3_h0_h3_exactpg_2026-09-14/figure3_data.json"),
        "model_files_sha256": study.model_hashes(),
        "check": "loaded adapters match recorded paths and immutable file hashes; new tasks intentionally have no historical probability rows",
    }
    write_json(output / "gpu_preflight.json", report)
    print(json.dumps({"status": report["status"], "models": {k: v["wall_seconds"] for k, v in report["models"].items()}}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("cpu", "gpu"))
    parser.add_argument("--experiment", type=Path, default=study.EXPERIMENT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.mode == "cpu":
        cpu_acceptance(args.experiment)
    else:
        gpu_preflight(args.experiment, args.device)


if __name__ == "__main__":
    main()
