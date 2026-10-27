"""Command-line entry point for an experiment factory."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Sequence

from src.hrp.core import run_experiment
from src.hrp.models import Experiment


def main(argv: Sequence[str] | None = None) -> int:
    """Load an experiment factory, run HRP, and print JSON diagnostics."""
    parser = argparse.ArgumentParser(prog="hrp")
    parser.add_argument(
        "experiment",
        help="Python experiment factory in module:callable form",
    )
    arguments = parser.parse_args(argv)
    module_name, factory_name = arguments.experiment.split(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    experiment = factory()
    if not isinstance(experiment, Experiment):
        raise TypeError("Experiment factory must return src.hrp.Experiment")
    result = run_experiment(
        experiment.adapters,
        experiment.target_models,
        experiment.teacher,
        experiment.cases,
        experiment.config,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
