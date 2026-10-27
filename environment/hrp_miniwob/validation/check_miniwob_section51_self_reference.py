"""Acceptance tests for finite-horizon self-reference and the sealed H0--H5 inputs."""

import argparse
import json
from pathlib import Path

import numpy as np

from hrp_miniwob.finite_horizon import (
    independent_success_rollouts,
    self_reference_metrics,
)


def check_numerics():
    # A shared successor checks accumulated visitation from distinct action paths.
    nxt = np.asarray([[-1, -1], [0, -1], [0, 1], [1, 2]])
    rewards = np.asarray([[1., 0.], [0., 1.], [0., 0.], [0., 0.]])
    p = np.asarray([[.3, .7], [.6, .4], [.8, .2], [.25, .75]])
    tau = .1
    metrics, arrays = self_reference_metrics(nxt, rewards, np.log(p), 3, tau)
    visits = np.zeros(4)
    success = 0.

    def enumerate_paths(state, mass):
        nonlocal success
        visits[state] += mass
        for action in range(2):
            child_mass = mass * p[state, action]
            success += child_mass * rewards[state, action]
            if nxt[state, action] >= 0:
                enumerate_paths(nxt[state, action], child_mass)

    enumerate_paths(3, 1.)
    np.testing.assert_allclose(metrics["success_exact"], success, atol=1e-14)
    np.testing.assert_allclose(arrays["visits"], visits, atol=1e-14)
    # Independent path-integral solution for deterministic terminal-binary graphs.
    ceiling = tau * np.logaddexp(np.log1p(-success), np.log(success) + 1/tau)
    np.testing.assert_allclose(metrics["ceiling"], ceiling, atol=1e-14)
    assert metrics["mismatch"] > 0
    np.testing.assert_allclose(metrics["C_minus_M"], success, atol=1e-14)
    outcomes, _ = independent_success_rollouts(nxt, rewards, np.log(p), 3, 4, 100_000, 51942)
    assert abs(outcomes.mean() - success) < .008
    for probability in (1e-8, .02, .2, .9, 1-1e-8):
        m, _ = self_reference_metrics(np.asarray([[-1, -1]]), np.asarray([[1., 0.]]),
                                      np.log([[probability, 1-probability]]), 0, tau)
        np.testing.assert_allclose(m["J"], probability, atol=1e-14)
    try:
        self_reference_metrics(np.asarray([[0, -1]]), np.asarray([[0., 1.]]),
                               np.log([[.5, .5]]), 0, tau)
    except ValueError:
        pass
    else:
        raise AssertionError("A cyclic graph must not enter the finite-horizon solver")
    return {"exhaustive_path_success": success, "sampled_success": float(outcomes.mean()),
            "path_integral_ceiling_error": abs(metrics["ceiling"]-ceiling),
            "identity_error": metrics["gap_identity_error"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path)
    args = parser.parse_args(argv)
    result = {"status": "passed", "numerics": check_numerics()}
    if args.experiment:
        from hrp_miniwob.benchmarks import miniwob_section51_self_reference as study
        from hrp_miniwob.suite import reachable_states

        manifest = study.verify_sources(args.experiment)
        checked, states = 0, 0
        for entry in manifest["entries"]:
            for harness in study.HARNESSES:
                graph, bank, codes = study.load_graph(args.experiment, entry, harness)
                np.testing.assert_array_equal(bank, reachable_states(graph))
                nxt, rewards, initial = study.compact_graph(graph, bank)
                if np.any(nxt >= np.arange(len(nxt))[:, None]):
                    raise AssertionError("Archived graph is not topologically ordered")
                chat, rendered_codes = study.render(graph, graph.initial_state)
                assert tuple(rendered_codes) == tuple(codes)
                assert "gamma=0.95" not in chat[0]["content"]
                assert "reference" not in chat[0]["content"]
                if entry["split"] == "calibration":
                    lp = np.full(nxt.shape, -np.log(nxt.shape[1]))
                    metrics, _ = self_reference_metrics(nxt, rewards, lp, initial, study.TAU)
                    success = metrics["success_exact"]
                    expected = study.TAU * np.logaddexp(np.log1p(-success), np.log(success)+1/study.TAU)
                    np.testing.assert_allclose(metrics["ceiling"], expected, atol=1e-12)
                checked += 1
                states += len(bank)
        result.update(graphs=checked, reachable_states=states, test_tasks=manifest["test_tasks"],
                      harnesses=list(study.HARNESSES), manifest_sha256=study.digest(args.experiment / "manifest.json"),
                      tests=["exhaustive trajectory enumeration", "independent path-integral optimum",
                             "finite visitation and terminal mass", "independent success rollouts",
                             "source/model fingerprints", "all task/graph checksums",
                             "all native action mappings", "all reachable state banks",
                             "all harness/family finite-horizon identities"])
        study.write(args.experiment / "acceptance/cpu.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
