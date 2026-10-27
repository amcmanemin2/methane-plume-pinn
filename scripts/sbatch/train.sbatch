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
# Verify the available module names on the cluster with `ml spider python`.
ml purge
ml python/3.12.1

source "${GROUP_HOME:-$HOME}/envs/pinn/bin/activate"

# --- paths -------------------------------------------------------------------
# Read data in place; write everything heavy to scratch.
export PINN_DATA_ROOT="${OAK:-$HOME}/methane-plume/data"
export PINN_RUN_ROOT="${SCRATCH:-$HOME}/methane-plume/runs"
export PINN_DEVICE=auto

mkdir -p "$PINN_RUN_ROOT"

# --- run ---------------------------------------------------------------------
CONFIG="${1:-configs/default.yaml}"
RUN_NAME="${2:-job-$SLURM_JOB_ID}"

python scripts/train.py --config "$CONFIG" --run-name "$RUN_NAME"
