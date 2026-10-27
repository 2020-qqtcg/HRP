"""Focused acceptance checks for Section 5.1 measurement V2."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_section51 as study
from hrp_miniwob.benchmarks import miniwob_section51_measurement_v2 as v2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=v2.EXPERIMENT)
    args = parser.parse_args(argv)
    experiment = args.experiment.resolve()

    # Total-beta rules must add exactly beta mass regardless of menu size.
    for actions in (3, 5, 7, 13, 21, 30):
        for beta in (.5, 1.0):
            rule = f"total_beta_{beta:.1f}"
            alpha = v2.alpha_for(rule, actions)
            assert abs(alpha * actions - beta) < 1e-15

    # The Binomial marginal implementation is checked against direct enumeration
    # on a one-state categorical policy.
    policy = np.asarray([[.1, .3, .6]], dtype=np.float64)
    target = np.log(np.asarray([[.2, .2, .6]], dtype=np.float64))
    occupancy = np.asarray([1 - study.GAMMA])
    exact = v2.exact_expected_mismatch(policy, target, occupancy, 16, "total_beta_0.5")
    rng = np.random.default_rng(6106001)
    estimates = []
    alpha = v2.alpha_for("total_beta_0.5", 3)
    for counts in rng.multinomial(16, policy[0], size=200_000):
        estimated = (counts + alpha) / (16 + .5)
        estimates.append(study.TAU * np.sum(estimated * (np.log(estimated) - target[0])))
    monte_carlo = float(np.mean(estimates))
    assert abs(exact - monte_carlo) < 5e-4

    d = v2.read(experiment / "calibration/D/acceptance.json")
    c = v2.read(experiment / "calibration/C/acceptance.json")
    frozen = v2.read(experiment / "calibration/frozen_choices.json")
    selected = d["selected"]
    assert d["status"] == "passed" and c["status"] == "passed"
    assert selected["max_condition_macro_absolute_expected_M_bias"] <= .005
    assert selected["max_family_condition_absolute_expected_M_bias"] <= .01
    assert selected["max_node_return_difference_RMSE"] <= .01
    assert c["summary"]["selected"]["configuration_aggregate_RMSE"] <= .03
    assert frozen["choices"]["measurement"]["id"] == selected["id"]

    manifest = v2.read(experiment / "suite/manifest.json")
    exclusions = v2.read(experiment / "exclusions.json")
    entries = manifest["entries"]
    assert len(entries) == 576
    assert len({entry["seed"] for entry in entries}) == 576
    assert len({entry["content_sha256"] for entry in entries}) == 576
    assert not ({entry["seed"] for entry in entries} & set(exclusions["seeds"]))
    assert not ({entry["content_sha256"] for entry in entries} & set(exclusions["content_sha256"]))
    for family in study.FAMILIES:
        assert sum(entry["family"] == family for entry in entries) == 192

    result = {
        "status": "passed", "validated_at_utc": datetime.now(UTC).isoformat(),
        "checks": {
            "total_beta_invariant": True,
            "binomial_expectation_vs_monte_carlo_absolute_error": abs(exact - monte_carlo),
            "D_gate": True, "C_gate": True, "frozen_choice_matches_selection": True,
            "fresh_test_tasks": 576, "family_tasks": 192,
            "seed_overlap": 0, "content_overlap": 0,
        },
    }
    output = experiment / "acceptance"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "cpu.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
