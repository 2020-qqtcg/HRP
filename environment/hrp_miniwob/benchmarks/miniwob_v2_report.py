"""Paired task-bootstrap helpers used by the paper's E1 analysis."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        # The training writer may be replacing a report while this monitor reads.
        return None


def paired_interval(after, before, experiment, key="fixed_bank_energy_mse"):
    old = {(r["family"], r["content_sha256"]): r for r in before["rows"]}
    assert old.keys() == {(r["family"], r["content_sha256"]) for r in after["rows"]}
    groups = defaultdict(lambda: defaultdict(list))
    for row in after["rows"]:
        family, content = row["family"], row["content_sha256"]
        groups[family][content].append(row[key] - old[family, content][key])
    rng = np.random.default_rng(42)
    samples, means = [], []
    for clusters in groups.values():
        totals = np.array([sum(values) for values in clusters.values()])
        counts = np.array([len(values) for values in clusters.values()])
        draw = rng.integers(len(totals), size=(2000, len(totals)))
        samples.append(totals[draw].sum(1) / counts[draw].sum(1))
        means.append(totals.sum() / counts.sum())
    lo, hi = np.quantile(np.mean(samples, axis=0), [.025, .975])
    return float(np.mean(means)), float(lo), float(hi)
