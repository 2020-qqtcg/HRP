"""Finite-harness CPU/native-browser acceptance, not a language-model experiment."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from hrp_miniwob.validation.check_miniwob_meta_harness import ProbePolicy
from hrp_miniwob.validation.check_miniwob_v2 import exact_update_check, gradient_check
from hrp_miniwob.suite import reachable_states
from hrp_miniwob.v2 import FAMILIES
from hrp_miniwob.v2.browser import Browser
from hrp_miniwob.v2.model import compile_task
from transformers import AutoTokenizer

from hrp_miniwob.harnesses.miniwob_harnesses import CODES, PROFILES
from hrp_miniwob.adapters import miniwob_harnesses as adapter
from hrp_miniwob.adapters import verl_miniwob
from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_harnesses as entry




def directed_checks(browser, directory):
    task = browser.reset(42, "ascending-numbers").task
    ordered = sorted(task.actions, key=lambda a: task.data["tiles"][int(a.removeprefix("click:tile"))]["number"])
    cases = {}
    for profile in ("H1", "H2", "H3", "H4"):
        # This reviewer deliberately approves incorrect moves, unlike the program.
        bound = adapter.Adapter(ProbePolicy(.01), adapter.Candidate(entry.candidate_path(profile)),
                                ProbePolicy(-.01), ProbePolicy(-.1), directory / profile)
        harness = adapter.Harness(browser, bound)
        try:
            state = harness.reset(task)
            if profile in ("H1", "H4"):
                result = harness.step("verify:" + ordered[-1])
                assert result.state.physical == state.physical and harness.counts["browser_actions"] == 0
                assert harness.last_review["approved"] == (profile == "H4")
                assert result.state.public()["verify_result"]["action"] == ordered[-1]
                result = harness.step(ordered[-1])
                assert result.state.terminated and result.reward == 0 and result.state.memory == 0
            elif profile == "H2":
                result = harness.step(ordered[-1])
                assert not result.state.terminated and result.state.memory == 1
                assert harness.counts["browser_actions"] == 0
                result = harness.step(ordered[-1])
                assert result.state.terminated and harness.counts["program_checks"] == 1
            elif profile == "H3":
                result = harness.step("batch:" + ",".join(a.removeprefix("click:") for a in ordered[-2:]))
                assert result.state.terminated and harness.counts["browser_actions"] == 1
                harness.reset(task)
                # Reach progress 3 with exactly one actor decision left. The
                # final batch must execute BOTH clicks and receive success.
                for action in [ordered[0], ordered[0], ordered[0], ordered[1], ordered[2]]:
                    state = harness.step(action).state
                assert state.remaining == 1
                result = harness.step("batch:" + ",".join(a.removeprefix("click:") for a in ordered[-2:]))
                assert result.reward == 1 and result.state.terminated and len(harness.last_execution) == 2
            cases[profile] = harness.counts
        finally:
            bound.close()
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("models/Qwen3.5-2B"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    report = {"status": "running", "scope": __doc__, "graphs": [],
              "source_sha256": entry.source_hashes()}
    errors = []
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

        def tokenizer_only(policy, *unused_args, **unused_kwargs):
            policy.tokenizer, policy.code_token_ids = tokenizer, {}
            policy.metadata = {"decoder": {}}

        with patch.object(adapter.base.LanguagePolicy, "__init__", tokenizer_only):
            decoder = adapter.LanguagePolicy(activation_checkpointing=False)
        assert len(set(decoder.code_token_ids.values())) == len(CODES)
        report["decoder_tokens"] = len(decoder.code_token_ids)
        report["maximum_audited_prompt_tokens"] = 0
        with entry.bindings(), entry.base.bindings(training=True):
            assert entry.base.runner.Candidate is adapter.Candidate
            assert verl_miniwob.LanguagePolicy is adapter.LanguagePolicy
        models = []
        with Browser(horizon=6) as browser:
            browser.page.on("pageerror", lambda error: errors.append(str(error)))
            for family in FAMILIES:
                model = compile_task(browser.reset(42, family).task, horizon=6)
                models.append(model)
                for profile in PROFILES:
                    directory = args.output / family / profile
                    bound = adapter.Adapter(ProbePolicy(.01), adapter.Candidate(entry.candidate_path(profile)),
                                            ProbePolicy(-.01), ProbePolicy(.1), directory)
                    try:
                        graph = bound.graph(model)
                        reference = graph.reference().copy()
                        mu, nu, paths = adapter.start_bank(graph)
                        bank = reachable_states(graph)
                        assert np.isclose(mu.sum(), 1) and np.isclose(nu.sum(), 1) and mu[graph.initial_state] == .5
                        assert np.array_equal(np.flatnonzero(mu), bank)
                        assert all(target < source for source in bank for target in graph.next_state[source] if target >= 0)
                        harness = adapter.Harness(browser, bound)
                        selected = sorted({graph.initial_state, int(bank[0]), int(bank[len(bank) // 2])})
                        checks = 0
                        for index in selected:
                            chat, codes = bound.messages(graph.state(index))
                            text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                            tokens = len(tokenizer.encode(text, add_special_tokens=False))
                            assert tokens <= 2048
                            assert len(codes) == len(set(codes)) == len(graph.task.actions)
                            report["maximum_audited_prompt_tokens"] = max(report["maximum_audited_prompt_tokens"], tokens)
                            for a, action in enumerate(graph.task.actions):
                                state = adapter.replay_start(harness, graph, index, paths)
                                result = harness.step(action)
                                nxt = -1 if result.state.terminated else graph.index(result.state.remaining, result.state.memory, result.state.physical)
                                assert nxt == graph.next_state[index, a] and result.reward == graph.task_rewards[index, a], (family, profile, index, action)
                                assert result.state.remaining == state.remaining - 1
                                if result.state.terminated:
                                    try:
                                        harness.step(action)
                                    except RuntimeError:
                                        pass
                                    else:
                                        raise AssertionError("Terminal task accepted another action")
                                checks += 1
                        error = gradient_check(graph)
                        updated = exact_update_check(model, bound) if family == FAMILIES[0] else None
                        metrics = adapter.evaluate_models([model], bound, directory / "evaluation")
                        assert np.array_equal(reference, graph.reference())
                        with np.load(directory / "evaluation/policies" / f"{model.task.key}.npz") as arrays:
                            assert tuple(arrays["actions"]) == graph.task.actions
                            assert arrays["log_policy"].shape == graph.next_state.shape
                        row = {"family": family, "harness": profile, "actions": len(graph.task.actions),
                               "controller_states": graph.task.memories, "reachable_states": len(bank),
                               "live_edges": checks, "gradient_error": error, "exact_update": updated,
                               "gap_identity_error": metrics["metrics"]["gap_identity_error"],
                               "mixed_gap_identity_error": metrics["metrics"]["mixed_gap_identity_error"]}
                        report["graphs"].append(row)
                        write_json(args.output / "status.json", report)
                        print(json.dumps(row), flush=True)
                    finally:
                        bound.close()
            report["directed_execution"] = directed_checks(browser, args.output / "directed")
        assert not errors, errors
        assert entry.source_hashes() == report["source_sha256"], "Source changed during acceptance"
        report.update(status="passed", browser_page_errors=errors)
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        write_json(args.output / "status.json", report)
    print(json.dumps({"status": report["status"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
