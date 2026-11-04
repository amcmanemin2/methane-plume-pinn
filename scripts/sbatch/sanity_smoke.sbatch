#!/bin/bash
#SBATCH --job-name=pinn-sanity
#SBATCH --output=runs/slurm-sanity-%j.out
#SBATCH --partition=gpu,serc
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#
# Sanity checks + smoke test (no full training run). Submit from the repo root:
#   sbatch scripts/sbatch/sanity_smoke.sbatch
# 1. unit tests (incl. file-based)   2. scripts/sanity.py (norm, LES FD residual, overfit, full-size timing)
# 3. smoke training (configs/smoke.yaml, < 5 min of training)   4. smoke evaluation (val + 1 test run)

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
python3 -c "import torch, h5py, scipy, matplotlib, yaml; print('imports ok', torch.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "=== 1. unit tests"; date
python3 -m unittest discover -s tests -v
echo "=== 2. sanity checks"; date
python3 scripts/sanity.py
echo "=== 3. smoke training"; date
python3 scripts/train.py --config configs/smoke.yaml --run-name smoke \
    --note "smoke: first run with auto output bias + frozen grad-norm loss weights (2026-11-04 plan)"
echo "=== 4. smoke evaluation"; date
python3 scripts/evaluate.py --test-runs daytime_NE
date
