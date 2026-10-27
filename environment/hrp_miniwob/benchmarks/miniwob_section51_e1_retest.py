"""Reuse verified E1 IID adaptation probabilities for finite-measurement retesting."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_section51 as study
from hrp_miniwob.benchmarks.miniwob_meta_harness import digest, model_files

EXPERIMENT = study.ROOT / "environment/artifacts/v2/section51_measurement_old_e1_iid_2026-09-23"
SOURCE = study.FIGURE3 / "old_e1_iid"


def reuse_adaptation(experiment=EXPERIMENT):
    experiment = Path(experiment).resolve()
    models = study.load_models(experiment, "test")
    config = study.read(SOURCE / "config.json")
    assert [m.task.seed for m in models] == config["task_seeds"]
    assert [m.task.fingerprint for m in models] == config["task_content_sha256"]
    assert digest(study.E1 / "suite/manifest.json") == config["suite_manifest_sha256"]

    archived_path = study.FIGURE3 / "source_snapshot/environment/hrp_miniwob/harnesses/miniwob_harnesses.py"
    assert digest(archived_path) == config["measurement_sources_sha256"][
        "environment/hrp_miniwob/harnesses/miniwob_harnesses.py"
    ]
    module_spec = importlib.util.spec_from_file_location("archived_e1_harness", archived_path)
    archived = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(archived)

    specs = study.cell_specs("test")
    all_hashes = study.model_hashes()
    outputs = {}
    for cell, spec in specs.items():
        if spec["study"] != "D":
            continue
        if study.cell_complete(experiment, "test", cell):
            outputs[cell] = study.read(study.cell_directory(experiment, "test", cell) / "completion.json")
            continue
        harness = spec["harness"]
        update = spec.get("update", 0)
        relative = f"seed42_H3/checkpoint_{update:06d}/{harness}" if update else f"initial/{harness}"
        source = SOURCE / "evaluations" / relative
        identity = study.read(source / "evaluation.json")
        completion = study.read(source / "completion.json")
        assert completion["metrics_sha256"] == digest(source / "metrics.json")
        assert completion["replay_audit_sha256"] == digest(source / "replay_audit.json")
        assert identity["cohort"] == "old_e1_iid"
        assert identity["supplement_config_sha256"] == digest(SOURCE / "config.json")
        assert identity["source_suite_sha256"] == config["suite_manifest_sha256"]
        adapter_key = f"h3_update_{update:04d}"
        assert identity["adapter_sha256"] == all_hashes[adapter_key]
        rows = {row["task_seed"]: row for row in study.read(source / "metrics.json")["rows"]}
        output = study.cell_directory(experiment, "test", cell)
        (output / "tasks").mkdir(parents=True, exist_ok=True)
        measured_path = study.BASE_MODEL if spec["model_id"] == "base" else spec["adapter"]
        write_json(output / "identity.json", {
            "proposal_sha256": digest(experiment / "proposal.json"),
            "suite_manifest_sha256": digest(experiment / "suite/manifest.json"),
            "split": "test", "cell_id": cell, "study": "D", "model_id": spec["model_id"],
            "model_files_sha256": all_hashes[spec["model_id"]],
            "adapter": str(spec["adapter"]) if spec["adapter"] else None,
            "rendering": spec.get("rendering"), "harness": harness,
            "update": spec.get("update"), "device": "cuda:0",
        })
        assert model_files(measured_path) == all_hashes[spec["model_id"]]
        old_candidate = archived.Candidate(study.candidate_path(harness))
        started = time.monotonic()
        table_hashes, source_hashes, states = {}, {}, 0
        for model in models:
            assert rows[model.task.seed]["content_sha256"] == model.task.fingerprint
            path = source / f"tasks/{model.task.key}/policies/{model.task.key}.npz"
            graph, bank, rendered = study.graph_and_prompts(model, spec)
            # Validate the actual historical prompts, action order and effects.
            for index, prompt in zip(bank, rendered):
                messages, codes = old_candidate.messages(graph.state(int(index)))
                assert messages == prompt.messages and tuple(codes) == tuple(prompt.semantic_codes)
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved["next_state"], graph.next_state)
                np.testing.assert_array_equal(saved["rewards"], graph.task_rewards)
                np.testing.assert_array_equal(saved["fixed_bank"], bank)
                assert tuple(saved["actions"]) == tuple(graph.task.actions)
                log_policy = np.log(saved["reference"][bank]) if spec["model_id"] == "base" else saved["log_policy"][bank]
                np.testing.assert_allclose(np.exp(log_policy).sum(1), 1., rtol=0, atol=1e-12)
                if spec["model_id"] == "base":
                    baseline = config["baseline_records"][harness]
                    base_path = Path(baseline["directory"]) / f"policies/{model.task.key}.npz"
                    assert digest(base_path) == baseline["policies_sha256"][model.task.key]
                    with np.load(base_path) as e1:
                        np.testing.assert_array_equal(saved["reference"], e1["reference"])
            target = study.task_table_path(output, model)
            np.savez_compressed(
                target, bank=bank, log_policy=log_policy,
                prompt_sha256=np.asarray([p.prompt_sha256 for p in rendered]),
                semantic_actions=np.asarray(graph.task.actions),
                semantic_codes=np.asarray(rendered[0].semantic_codes),
                next_state=graph.next_state, task_rewards=graph.task_rewards,
                initial_state=np.asarray(graph.initial_state),
            )
            table_hashes[model.task.seed] = digest(target)
            source_hashes[model.task.key] = digest(path)
            states += len(bank)
            write_json(target.with_suffix(".json"), {
                "task_seed": model.task.seed, "family": model.task.family,
                "content_sha256": model.task.fingerprint, "study": "D", "harness": harness,
                "states": len(bank), "graph_states": len(graph.next_state),
                "actions": len(graph.task.actions), "table_sha256": table_hashes[model.task.seed],
                "source_path": str(path), "source_sha256": source_hashes[model.task.key],
                "cache_reused": True,
            })
        write_json(output / "accounting.json", {
            "status": "complete", "split": "test", "cell_id": cell, "study": "D",
            "tasks": len(models), "states": states, "cache_reused": True,
            "forward_state_rows": 0, "prompt_tokens": 0, "forward_microbatches": 0,
            "wall_seconds": 0., "physical_gpu": None,
            "import_wall_seconds": time.monotonic() - started,
            "source_directory": str(source), "source_table_sha256": source_hashes,
            "source_identity_sha256": digest(source / "evaluation.json"),
        })
        result = {
            "status": "complete", "cache_reused": True,
            "identity_sha256": digest(output / "identity.json"),
            "accounting_sha256": digest(output / "accounting.json"),
            "task_table_hashes_sha256": study.sha256_bytes(json.dumps(table_hashes, sort_keys=True).encode()),
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        write_json(output / "completion.json", result)
        outputs[cell] = result
        print(f"Verified and imported {cell}: {len(models)} E1 IID tasks", flush=True)
    (experiment / "acceptance").mkdir(parents=True, exist_ok=True)
    write_json(experiment / "acceptance/reused_adaptation.json", {
        "status": "passed", "cohort": "old_e1_iid", "source": str(SOURCE),
        "tasks": len(models), "cells": outputs,
        "checks": ["task contents", "checkpoint hashes", "historical prompts", "action order",
                   "transition and reward graphs", "reachable states", "reference probabilities"],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT)
    reuse_adaptation(parser.parse_args().experiment)
