"""Reproducible browser validation and oracle rollouts."""

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .assets import MANIFEST, ROOT, fetch_assets
from .browser import CheckboxBrowser
from .diagnostics import validate_oracle
from .model import compile_task
from .oracle import solve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch", help="Download and verify pinned MiniWoB frontend assets")
    for name in ("validate", "demo"):
        command = commands.add_parser(name)
        command.add_argument("--horizon", type=int, default=10)
        command.add_argument("--gamma", type=float, default=0.95)
        command.add_argument("--tau", type=float, default=0.1)
        command.add_argument("--click-cost", type=float, default=0.02)
        command.add_argument("--submit-cost", type=float, default=0.0)
        command.add_argument("--submit-reference", type=float)
        command.add_argument("--output", type=Path, default=ROOT / "artifacts" / name)
        command.add_argument("--headed", action="store_true")
        if name == "validate":
            command.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42])
            command.add_argument("--policies", type=int, default=32)
            command.add_argument("--policy-seed", type=int, default=2026)
        else:
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--policy-seed", type=int, default=2026)
            command.add_argument("--style", choices=("json", "text"), default="json")
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_assets()
        print(json.dumps({"revision": MANIFEST["revision"], "verified_files": len(MANIFEST["files"])}))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "validate":
        # A failed rerun must not leave an old successful run's marker in place.
        (args.output / "summary.json").write_text(json.dumps({
            "status": "incomplete", "requested_seeds": args.seeds,
            "upstream_revision": MANIFEST["revision"],
        }, indent=2) + "\n")
    started = time.perf_counter()
    with CheckboxBrowser(args.horizon, args.click_cost, args.submit_cost, args.headed) as browser:
        if args.command == "demo":
            model = compile_task(browser, args.seed)
            costs = model.costs(args.click_cost, args.submit_cost)
            reference = model.reference(args.submit_reference)
            oracle = solve(model.next_state, model.task_rewards - costs, reference, args.gamma, args.tau)
            state = browser.reset(args.seed)
            browser.screenshot(str(args.output / "initial.png"))
            rng = np.random.default_rng(args.policy_seed)
            trace = []
            while not state.terminated:
                index = (state.remaining - 1) * 2 ** len(state.task.labels) + state.mask
                action_index = int(rng.choice(len(state.task.actions), p=oracle.policy[index]))
                action = state.task.actions[action_index]
                observation = state.observation(args.style)
                result = browser.step(action)
                trace.append({"observation": observation, "action": action,
                              "probability": float(oracle.policy[index, action_index]),
                              "reward": result.reward, "cost": result.cost,
                              "raw_reward": result.raw_reward, "reason": result.reason,
                              "next_state": asdict(result.state)})
                print(f"{len(trace)}. {action}: mask={result.state.mask}, reward={result.reward}", flush=True)
                state = result.state
            browser.screenshot(str(args.output / "final.png"))
            report = {"policy_source": "exact_soft_dp", "task": asdict(model.task),
                      "horizon": args.horizon, "gamma": args.gamma, "tau": args.tau,
                      "costs": costs.tolist(), "reference": reference[0].tolist(),
                      "policy_seed": args.policy_seed, "trace": trace,
                      "upstream_revision": MANIFEST["revision"]}
            (args.output / "demo.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"Saved rollout: {args.output / 'demo.json'}")
            return

        reports = []
        for task_seed in args.seeds:
            print(f"Enumerating browser transitions for seed {task_seed}...", flush=True)
            clock_before = browser.page.evaluate("Date.now()")
            model = compile_task(browser, task_seed)
            repeated = compile_task(browser, task_seed)
            np.testing.assert_array_equal(model.next_state, repeated.next_state)
            np.testing.assert_array_equal(model.raw_submit_rewards, repeated.raw_submit_rewards)
            assert model.task == repeated.task
            assert browser.page.evaluate("Date.now()") == clock_before
            costs = model.costs(args.click_cost, args.submit_cost)
            reference = model.reference(args.submit_reference)
            report, arrays = validate_oracle(
                model.next_state, model.task_rewards, costs, reference,
                model.initial_state, args.gamma, args.tau, args.policy_seed, args.policies,
            )
            report["task"] = asdict(model.task)
            report["environment"] = {
                "horizon": args.horizon,
                "semantic_actions": model.task.actions,
                "nonterminal_states": len(model.next_state),
                "physical_states": len(model.physical_next),
                "initial_state": model.initial_state,
                "browser_checks": model.browser_checks + repeated.browser_checks,
                "independent_enumerations": 2,
                "page_clock_frozen": True,
                "native_grading_matches_public_instruction": True,
                "observation_contract": "canonical_stateless_full_observation",
                "terminal_reward": "1 for fully correct submit, 0 otherwise",
                "regularized_objective": "task_reward - action_cost - tau*log(policy/reference)",
            }
            arrays.update({"next_state": model.next_state, "task_rewards": model.task_rewards,
                           "costs": costs, "reference": reference,
                           "physical_next": model.physical_next,
                           "raw_submit_rewards": model.raw_submit_rewards})
            np.savez_compressed(args.output / f"seed_{task_seed}.npz", **arrays)
            (args.output / f"seed_{task_seed}.json").write_text(json.dumps(report, indent=2) + "\n")
            reports.append(report)
            print(f"Seed {task_seed}: {len(model.next_state)} states; "
                  f"{report['environment']['browser_checks']} browser checks passed.", flush=True)

        summary = {
            "status": "passed",
            "scope": "browser_grounded_controlled_environment_and_numerical_theory_validation",
            "policy_source": "seeded_tabular_policies_not_trained_language_models",
            "upstream": MANIFEST,
            "runtime": {"python": platform.python_version(), "numpy": version("numpy"),
                        "playwright": version("playwright"), "chromium": browser.browser.version},
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(Path(__file__).parent.glob("*.py"))},
            "elapsed_seconds": time.perf_counter() - started,
            "tasks": reports,
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Saved validation: {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
