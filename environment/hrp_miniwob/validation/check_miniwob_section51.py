"""Acceptance checks for the Section 5.1 diagnostic experiment."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from hrp_miniwob.adapters.miniwob_readout import ReadoutRenderer
from hrp_miniwob.adapters.miniwob_section51 import (
    NEUTRAL_NEW,
    NEUTRAL_OLD,
    NeutralReadoutRenderer,
    controlled_kernel,
    controlled_metrics,
    controlled_reachable_states,
    evaluate_controlled,
    solve_controlled,
)
from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_section51 as study
from hrp_miniwob.benchmarks import miniwob_section51_analysis as analysis
from hrp_miniwob.oracle import evaluate, solve
from hrp_miniwob.v2.browser import Browser


def cpu_acceptance(experiment):
    experiment = Path(experiment).resolve()
    models = study.load_models(experiment, "calibration")
    representatives = [next(model for model in models if model.task.family == family) for family in study.FAMILIES]
    maximum = defaultdict_float()
    branch_checks = mapping_checks = browser_checks = 0
    for model in representatives:
        state = model.state(model.initial_state)
        for rendering in study.C_RENDERINGS:
            original = ReadoutRenderer(rendering).render(state)
            neutral = NeutralReadoutRenderer(rendering).render(state)
            if original.semantic_codes != neutral.semantic_codes or original.mapping != neutral.mapping:
                raise AssertionError("Neutral prompt changed the reversible action mapping")
            if NEUTRAL_OLD in neutral.messages[0]["content"] or NEUTRAL_NEW not in neutral.messages[0]["content"]:
                raise AssertionError("Neutral prompt retains a zero-cost claim or lacks its replacement")
            if original.messages[1] != neutral.messages[1]:
                raise AssertionError("Neutral prompt changed the public observation")
            mapping_checks += 1

        rng = np.random.default_rng(5000 + model.task.seed)
        reference = rng.dirichlet(np.ones(len(model.task.actions)), size=len(model.next_state))
        policy = rng.dirichlet(np.ones(len(model.task.actions)), size=len(model.next_state))
        deterministic_optimum = solve(model.next_state, model.task_rewards, reference, study.GAMMA, study.TAU)
        deterministic_eval = evaluate(
            model.next_state, model.task_rewards, reference, policy, model.initial_state,
            study.GAMMA, study.TAU,
        )
        kernel = controlled_kernel(model, 1.0, 0.0)
        stochastic_optimum = solve_controlled(kernel, reference, study.GAMMA, study.TAU)
        stochastic_eval = evaluate_controlled(kernel, reference, policy, study.GAMMA, study.TAU)
        maximum["p1_optimal_value"] = max(
            maximum["p1_optimal_value"],
            float(np.max(np.abs(deterministic_optimum.value - stochastic_optimum.value))),
        )
        maximum["p1_optimal_policy"] = max(
            maximum["p1_optimal_policy"],
            float(np.max(np.abs(deterministic_optimum.policy - stochastic_optimum.policy))),
        )
        maximum["p1_policy_value"] = max(
            maximum["p1_policy_value"],
            float(np.max(np.abs(deterministic_eval.value - stochastic_eval.value))),
        )
        if abs(stochastic_eval.occupancy.sum() + stochastic_eval.terminal_occupancy_mass - 1) > 1e-12:
            raise AssertionError("Controlled occupancy lost absorbing mass")
        for reliability, cost in ((0.0, .05), (.5, .01), (.8, 0), (1.0, .05)):
            candidate = controlled_kernel(model, reliability, cost)
            if not np.allclose(candidate.probability.sum(2), 1, rtol=0, atol=1e-14):
                raise AssertionError("Stochastic execution probabilities do not normalize")
            for index in controlled_reachable_states(candidate):
                public = model.state(int(index))
                for action, is_click in enumerate(candidate.click_mask):
                    if not is_click or public.remaining == 1:
                        continue
                    expected = (public.remaining - 2) * model.task.physical_states + public.physical
                    if candidate.next_state[index, action, 1] != expected:
                        raise AssertionError("Failed click did not consume exactly one decision")
                    branch_checks += 1
            if not np.allclose(candidate.utility_cost[:, candidate.click_mask], cost):
                raise AssertionError("Click cost was not charged on both execution branches")
            if np.any(candidate.utility_cost[:, ~candidate.click_mask]):
                raise AssertionError("A non-click action was charged click cost")
            if np.any(candidate.task_reward[:, :, 1]):
                raise AssertionError("Failed clicks changed native task success")
            metrics = controlled_metrics(candidate, reference, policy)
            maximum["gap_identity"] = max(maximum["gap_identity"], metrics["gap_identity_error"])
            maximum["return_decomposition"] = max(
                maximum["return_decomposition"], metrics["return_decomposition_error"]
            )

        graph0 = study.harness_graph(model, "H0")
        graph3 = study.harness_graph(model, "H3")
        if tuple(graph0.task.actions) != tuple(model.task.actions):
            raise AssertionError("Current H0 graph changed native semantic actions")
        if not any(action.startswith("batch:") for action in graph3.task.actions):
            raise AssertionError("Current H3 graph lacks batched click actions")

    # Recheck the compiled native branch against live browser transitions on a
    # representative initial state/action basis. The stochastic failure branch
    # is a deployment wrapper and is checked algebraically above.
    with Browser(horizon=study.HORIZON) as browser:
        for model in representatives:
            for action_index, action in enumerate(model.task.actions):
                initial = browser.reset(model.task.seed, model.task.family)
                if initial.task != model.task:
                    raise AssertionError("Seeded public task changed during native branch audit")
                result = browser.step(action)
                expected = int(model.next_state[model.initial_state, action_index])
                actual = -1 if result.state.terminated else (
                    (result.state.remaining - 1) * model.task.physical_states + result.state.physical
                )
                if actual != expected or result.reward != model.task_rewards[model.initial_state, action_index]:
                    raise AssertionError("Controlled normal-execution branch disagrees with native browser")
                browser_checks += 1

    # Deliberate faulty occupancy normalization must alter the mismatch whenever
    # terminal mass is nonzero.  This ensures acceptance catches that bug.
    model = representatives[0]
    reference = np.full(model.next_state.shape, 1 / model.next_state.shape[1])
    policy = np.full_like(reference, 1 / reference.shape[1])
    optimum = solve(model.next_state, model.task_rewards, reference, study.GAMMA, study.TAU)
    evaluated = evaluate(
        model.next_state, model.task_rewards, reference, policy, model.initial_state,
        study.GAMMA, study.TAU,
    )
    state_kl = np.sum(policy * (np.log(policy) - optimum.log_policy), axis=1)
    correct = study.TAU / (1 - study.GAMMA) * float(evaluated.occupancy @ state_kl)
    wrong = study.TAU / (1 - study.GAMMA) * float(
        (evaluated.occupancy / evaluated.occupancy.sum()) @ state_kl
    )
    if abs(correct - wrong) < 1e-6:
        raise AssertionError("Fault-injected occupancy renormalization was not detected")

    # The point estimator itself has no reward, return, success, or exact metric
    # access. Direct-rollout baselines are separate functions and are labeled so.
    source = inspect.getsource(analysis.single_probe)
    forbidden = ("task_reward", "controlled_metrics", "regularized_return", "success_probability")
    if any(token in source for token in forbidden):
        raise AssertionError("Finite mismatch estimator reads a forbidden truth field")
    environment_fields = set(analysis.SamplingEnvironment.__dataclass_fields__)
    if environment_fields != {"policy", "initial_state", "next_state", "controlled"}:
        raise AssertionError("Sampling service contract changed unexpectedly")

    streams = {
        analysis.stable_seed(5102, "pilot"),
        analysis.stable_seed(5103, "score"),
        analysis.stable_seed(5104, "adaptation-pilot"),
        analysis.stable_seed(5105, "adaptation-score"),
    }
    if len(streams) != 4:
        raise AssertionError("Pilot and scoring random streams collide")
    graph_control = analysis.graph_reconstruction(np.asarray([13, 7, 5, 3]))
    if any(graph_control[name]["max_centered_error"] > 1e-12 for name in ("star", "chain", "complete")):
        raise AssertionError("A connected graph failed to reconstruct centered log frequencies")
    if graph_control["disconnected"]["identifies_full_centered_coordinate"]:
        raise AssertionError("Disconnected graph incorrectly claimed full identification")
    if max(maximum.values(), default=0) > 1e-11:
        raise AssertionError(f"CPU acceptance numerical tolerance failed: {dict(maximum)}")

    report = {
        "status": "passed",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "representative_tasks": [model.task.key for model in representatives],
        "neutral_mapping_checks": mapping_checks,
        "failed_click_budget_checks": branch_checks,
        "live_native_branch_checks": browser_checks,
        "maximum_errors": dict(maximum),
        "fault_injected_occupancy_difference": abs(correct - wrong),
        "predictor_forbidden_truth_fields": list(forbidden),
        "random_streams_distinct": True,
        "graph_reconstruction": graph_control,
        "readout_source_audit": study.audit_readout_source(),
    }
    output = experiment / "acceptance"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "cpu.json", report)
    print(json.dumps(report), flush=True)
    return report


def gpu_acceptance(experiment, device):
    experiment = Path(experiment).resolve()
    models = study.load_models(experiment, "calibration")
    representatives = [next(model for model in models if model.task.family == family) for family in study.FAMILIES]
    specs = {
        # Use the superset H0/H3 decoder for models whose smoke includes H3;
        # it contains the same uppercase tokens plus the required lowercase set.
        "base": study.cell_specs("calibration")["D_base_H3"],
        "h0_final": study.cell_specs("calibration")["D_h3_update_0000_H0"],
        "h3_adapt_final": study.cell_specs("calibration")["D_h3_update_1024_H3"],
    }
    report = {
        "status": "running",
        "device": device,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "models": {},
    }
    output = experiment / "acceptance"
    output.mkdir(parents=True, exist_ok=True)
    for model_id, spec in specs.items():
        started = time.monotonic()
        policy = study.load_policy(spec, device)
        maximum_normalization = maximum_repeat = 0.0
        rows = 0
        try:
            for model in representatives:
                variants = []
                if model_id == "base":
                    variants.append((study.cell_specs("calibration")["C_base_R0"], "C neutral R0"))
                    variants.append((study.cell_specs("calibration")["D_base_H3"], "D native H3"))
                elif model_id == "h0_final":
                    variants.append((study.cell_specs("calibration")["B_h0_final_R1"], "B R1"))
                    variants.append((study.cell_specs("calibration")["D_h3_update_0000_H0"], "D node0 H0"))
                else:
                    variants.append((study.cell_specs("calibration")["C_h3_adapt_final_R3"], "C neutral R3"))
                    variants.append((study.cell_specs("calibration")["D_h3_update_1024_H3"], "D node1024 H3"))
                for variant, _ in variants:
                    _, bank, rendered = study.graph_and_prompts(model, variant)
                    take = np.linspace(0, len(bank) - 1, min(8, len(bank)), dtype=int)
                    prompts = [rendered[index].messages for index in take]
                    codes = [rendered[index].semantic_codes for index in take]
                    with torch.no_grad():
                        first = policy.batch_log_probabilities(prompts, codes).detach().cpu().numpy()
                        second = policy.batch_log_probabilities(prompts, codes).detach().cpu().numpy()
                    maximum_normalization = max(
                        maximum_normalization, float(np.max(np.abs(np.exp(first).sum(1) - 1)))
                    )
                    maximum_repeat = max(maximum_repeat, float(np.max(np.abs(first - second))))
                    rows += len(take)
            if maximum_normalization > 1e-12 or maximum_repeat > 1e-12:
                raise AssertionError("GPU normalization or deterministic repeatability failed")
            report["models"][model_id] = {
                "status": "passed",
                "state_rows": rows,
                "max_probability_normalization_error": maximum_normalization,
                "max_repeated_log_probability_error": maximum_repeat,
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
    report["all_checkpoint_hashes"] = study.model_hashes()
    write_json(output / "gpu.json", report)
    print(json.dumps({"status": "passed", "models": {k: v["wall_seconds"] for k, v in report["models"].items()}}), flush=True)
    return report


def defaultdict_float():
    from collections import defaultdict
    return defaultdict(float)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "gpu"))
    parser.add_argument("--experiment", type=Path, default=study.EXPERIMENT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.mode == "cpu":
        cpu_acceptance(args.experiment)
    else:
        gpu_acceptance(args.experiment, args.device)


if __name__ == "__main__":
    main()
