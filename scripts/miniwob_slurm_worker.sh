#!/usr/bin/env bash
set -euo pipefail
# This partition contains NVIDIA GPUs. Slurm also exports AMD visibility
# variables, which verl rejects when CUDA_VISIBLE_DEVICES is present.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
# Slurm's global GPU IDs are stable across jobs on this node. CUDA_VISIBLE_DEVICES
# can be remapped to zero inside a single-GPU allocation.
export TRITON_CACHE_DIR="/tmp/hrp-${UID}/gpu-${SLURM_JOB_GPUS:?}/lane-${HRP_GPU_LANE:-0}/triton"
mkdir -p "$TRITON_CACHE_DIR"
exec 9>"$TRITON_CACHE_DIR/.worker.lock"
flock 9
exec "$@"
