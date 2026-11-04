#!/bin/bash
#SBATCH --job-name=pinn-train
#SBATCH --output=runs/slurm-train-%j.out
#SBATCH --partition=gpu,serc
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# Train the pit-2 PINN on a GPU, then evaluate it on the 12 test runs.
#   sbatch scripts/sbatch/train_gpu.sbatch <run-name> [--set key=value ...]
# Holds about 26 GB of training data in RAM (12 runs at 40 m stride).

set -euo pipefail
ml purge
ml math devel viz
ml python/3.12.1 physics py-scipy/1.12.0_py312 py-numpy/1.26.3_py312 py-h5py/3.10.0_py312 \
   py-matplotlib/3.8.3_py312 py-pytorch/2.4.1_py312
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export PINN_DATA_ROOT="$OAK/anm49/methane-plume-pinn/data"
export PINN_RUN_ROOT="$SCRATCH/methane-plume-pinn/runs"
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NAME="${1:?usage: sbatch train_gpu.sbatch <run-name> [--set k=v ...]}"; shift
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 scripts/train.py --run-name "$NAME" "$@"
if [[ "${SKIP_EVAL:-0}" != 1 ]]; then
  python3 scripts/evaluate.py
fi
