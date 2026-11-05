#!/bin/bash
#SBATCH --job-name=pinn-sanity
#SBATCH --output=runs/slurm-sanity-%j.out
#SBATCH --partition=gpu,serc
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#
# One sanity experiment (arguments go to scripts/sanity.py). Submit from the repo root, e.g.
#   sbatch scripts/sbatch/sanity.sbatch --only overfit --overfit-parts a --set training.clip_grad=null --note "..."
# Raise --time with sbatch's own --time if the experiment is longer.

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
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 scripts/sanity.py "$@"
