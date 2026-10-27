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
# torch, numpy and scipy come from these modules (not from the venv), so they must be
# loaded before activating it. Verify names with `ml spider <name>`.
ml purge
ml math devel viz
ml python/3.12.1 uv physics gdal/3.10.2 py-h5py/3.10.0_py312 py-mpi4py/4.0.3_py312 \
   py-scipy/1.12.0_py312 py-numpy/1.26.3_py312 py-scikit-image/0.24.0_py312 \
   claude-code py-pytorch/2.4.1_py312

source "$GROUP_HOME/envs/cee292_venv/bin/activate"

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
