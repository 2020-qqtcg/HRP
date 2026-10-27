"""Isolated H0-H5 training/evaluation on the v2 three-task suite."""

import argparse
import json
import sys
from contextlib import contextmanager

from hrp_miniwob.harnesses.miniwob_harnesses import PROFILES
from hrp_miniwob.adapters import miniwob_harnesses as adapter
from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_v2 as base

ROOT = base.ROOT
DEFAULT_EXPERIMENT = ROOT / "environment/artifacts/v2/h0_h5"
BASE_SOURCES, BASE_PREPARE, BASE_READ_CONFIG = base.source_hashes, base.prepare, base.read_config


def candidate_path(profile):
    return ROOT / "environment/hrp_miniwob/harnesses" / f"miniwob_{profile.lower()}.py"


def source_hashes():
    hashes = BASE_SOURCES()
    paths = [ROOT / "environment/hrp_miniwob/harnesses/miniwob_harnesses.py", ROOT / "environment/hrp_miniwob/adapters/miniwob_harnesses.py",
             ROOT / "environment/hrp_miniwob/benchmarks/miniwob_harnesses.py", ROOT / "environment/hrp_miniwob/benchmarks/miniwob_harnesses.py",
             ROOT / "environment/hrp_miniwob/orchestration/run_miniwob_harnesses.py", *map(candidate_path, PROFILES)]
    hashes.update({str(path.relative_to(ROOT)): base.runner.digest(path) for path in paths})
    return hashes


def prepare(args):
    BASE_PREPARE(args)
    path = args.experiment.resolve() / "config.json"
    config = json.loads(path.read_text())
    config.update(harness_protocol="h0_h5_v2", harnesses=list(PROFILES),
                  action_decoder="complete parameterized menu; one reversible A-Z/a-z token per action",
                  controller="explicit verification result or automatic gate",
                  budget_unit="one actor decision; a batch may execute two native clicks; gamma applies per actor decision",
                  claim_scope="H0-H5 controlled interfaces on three native task families; no independent memory tool")
    write_json(path, config)


def read_config(experiment):
    config = BASE_READ_CONFIG(experiment)
    if config.get("harness_protocol") != "h0_h5_v2":
        raise ValueError("Prepare a new finite-harness experiment; legacy candidates and caches are incompatible")
    return config


@contextmanager
def bindings():
    """Reuse v2's worker bindings without editing any sealed source file."""
    replacements = {"adapter": adapter, "source_hashes": source_hashes, "prepare": prepare,
                    "read_config": read_config, "BASELINE": candidate_path("H0"),
                    "DEFAULT_EXPERIMENT": DEFAULT_EXPERIMENT}
    original = {key: getattr(base, key) for key in replacements}
    try:
        for key, value in replacements.items():
            setattr(base, key, value)
        yield
    finally:
        for key, value in original.items():
            setattr(base, key, value)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["list"]:
        print(json.dumps({key: {"candidate": str(candidate_path(key)), "behavior": value}
                          for key, value in PROFILES.items()}, indent=2))
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--harness", choices=PROFILES)
    parser.add_argument("--harnesses", choices=PROFILES, nargs="+")
    selection, rest = parser.parse_known_args(argv)
    if selection.harness:
        if not rest or rest[0] not in ("train", "evaluate") or "--candidate" in rest:
            parser.error("--harness applies to train/evaluate and replaces --candidate")
        rest += ["--candidate", str(candidate_path(selection.harness))]
    if selection.harnesses:
        if not rest or rest[0] != "cross-eval" or "--candidates" in rest:
            parser.error("--harnesses applies to cross-eval and replaces --candidates")
        rest += ["--candidates", *(str(candidate_path(h)) for h in selection.harnesses)]
    with bindings():
        base.main(rest)


if __name__ == "__main__":
    main()
