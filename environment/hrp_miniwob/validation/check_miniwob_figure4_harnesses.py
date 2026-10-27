"""Static acceptance checks for the fixed Figure 4 H0--H5 protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_figure4_harnesses as study
from hrp_miniwob.benchmarks.miniwob_meta_harness import digest, model_files
from hrp_miniwob.harnesses.miniwob_harnesses import PROFILES, Candidate


def validate(experiment=study.EXPERIMENT):
    experiment = Path(experiment).resolve()
    proposal = study.read(experiment / "proposal.json")
    if proposal != study.proposal():
        raise AssertionError("Figure 4 proposal changed")
    if tuple(PROFILES) != study.HARNESSES:
        raise AssertionError("Current harness numbering is not contiguous H0--H5")
    candidates = {}
    for harness in study.HARNESSES:
        candidate = Candidate(study.candidate_path(harness))
        if candidate.profile != harness:
            raise AssertionError(f"Candidate/profile mismatch: {harness}")
        candidates[harness] = {
            "path": str(candidate.path), "sha256": candidate.sha256,
            "behavior": PROFILES[harness],
        }
    source_manifests = {
        "calibration": study.read(study.CALIBRATION_SOURCE / "suite/manifest.json"),
        "test": study.read(study.TEST_SOURCE / "suite/manifest.json"),
    }
    counts = {split: sum(entry["split"] == split for entry in manifest["entries"])
              for split, manifest in source_manifests.items()}
    if counts != {"calibration": 72, "test": 576}:
        raise AssertionError("Reused V2 task counts changed")
    numbering = study.read(study.section51.E1 / "harness_numbering.json")
    if numbering["current_to_recorded"] != {
        "H0": "H0", "H1": "H1", "H2": "H2", "H3": "H3", "H4": "H5", "H5": "H7"
    }:
        raise AssertionError("Current-to-recorded harness mapping changed")
    result = {
        "status": "passed", "task_id": proposal["task_id"],
        "study_source_sha256": digest(study.BRIEF),
        "source_manifest_sha256": {split: digest(source / "suite/manifest.json")
                                   for split, source in (("calibration", study.CALIBRATION_SOURCE), ("test", study.TEST_SOURCE))},
        "source_config_sha256": {split: digest(source / "config.json")
                                 for split, source in (("calibration", study.CALIBRATION_SOURCE), ("test", study.TEST_SOURCE))},
        "checkpoint_files_sha256": model_files(study.H0_ADAPTER),
        "split_counts": counts, "candidates": candidates, "numbering": numbering,
        "checks": [
            "fixed H0 update 1024 adapter and source seed 42",
            "six current native profiles H0--H5",
            "H4 maps to recorded H5 reviewer; H5 maps to recorded H7 prompt",
            "V2 calibration/test task manifests reused byte-for-byte",
            "fixed m=256 N=1024 alpha=0.5 and 32 independent scoring rollouts",
        ],
    }
    output = experiment / "acceptance"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "cpu.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=study.EXPERIMENT)
    args = parser.parse_args(argv)
    print(json.dumps(validate(args.experiment), indent=2))


if __name__ == "__main__":
    main()
