#!/bin/bash
#SBATCH --job-name=pinn-train
#SBATCH --output=runs/slurm-%j.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# GPU is optional for this model -- uncomment if a partition is available to you.
# Check your access with `sh_part` before relying on it.
##SBATCH --partition=gpu
##SBATCH --gpus=1

set -euo pipefail

# --- environment -------------------------------------------------------------
# No venv: torch, numpy, scipy and matplotlib come from modules. Verify names with
# `ml spider <name>`. Do not add gdal (not used, and it breaks the load after `ml purge`).
ml purge
ml math devel viz
ml python/3.12.1 physics py-scipy/1.12.0_py312 py-numpy/1.26.3_py312 \
   py-matplotlib/3.8.3_py312 py-pytorch/2.4.1_py312
export PYTHONPATH="$PWD/src:$PYTHONPATH"   # package is not pip-installed; submit from the repo root

# --- paths -------------------------------------------------------------------
# Read data in place; write everything heavy to scratch.
export PINN_DATA_ROOT="${OAK:-$HOME}/methane-plume/data"
export PINN_RUN_ROOT="${SCRATCH:-$HOME}/methane-plume/runs"
export PINN_DEVICE=auto

mkdir -p "$PINN_RUN_ROOT"

# --- run ---------------------------------------------------------------------
CONFIG="${1:-configs/default.yaml}"
RUN_NAME="${2:-job-$SLURM_JOB_ID}"

python3 scripts/train.py --config "$CONFIG" --run-name "$RUN_NAME"
