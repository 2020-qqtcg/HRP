# Controlled MiniWoB++

Browser environments, finite semantic actions, exact dynamic-programming
oracles, harness comparisons, and Exact-PG training.
See the root [README](../README.md) for installation and commands.

- `hrp_miniwob/`: environment and experiment code.
- `training/`: locked GPU training dependencies, including verl's worker backend.
- `vendor/`: third-party source and pinned frontend assets.

Experiment commands create task suites, checkpoints, and reports under
`artifacts/`. Model checkpoints are supplied separately.

Paper experiments live in `hrp_miniwob/benchmarks/`:

| Experiment | Module |
| --- | --- |
| H0–H5 training and cross-evaluation | `miniwob_harnesses` |
| Presentation changes | `miniwob_readout` |
| Interface adaptation and harness swaps | `miniwob_adaptation`, `miniwob_adaptation_pairs` |
| Finite mismatch calibration | `miniwob_section51_measurement_v2` |
| Deployment predictions | `miniwob_figure4_harnesses` |
| Frozen-model harness scores | `miniwob_section51_self_reference` |

Run a module with:

```bash
uv run --project environment/training python -m hrp_miniwob.benchmarks.MODULE --help
```

Adaptation and analysis consume the task suites and checkpoints from earlier
stages. Code labels H1 and H3 correspond to paper labels H3 and H1, respectively.
