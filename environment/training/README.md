# MiniWoB++ training environment

Python 3.12, verl 0.9.0, PyTorch 2.13.0 with CUDA 12.9,
Transformers 5.10, and PEFT 0.20. Dependencies are pinned in `uv.lock`.

From the repository root:

```bash
bash scripts/setup_miniwob_training.sh
bash scripts/miniwob_harnesses.sh train --help
```

Supply local actor, reference and reviewer checkpoints to the `prepare` command.
Training uses Exact-PG with verl's worker and AdamW optimizer backend.
