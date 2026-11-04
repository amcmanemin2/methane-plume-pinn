#!/bin/bash
#SBATCH --job-name=pinn-extract
#SBATCH --output=runs/slurm-extract-%j.out
#SBATCH --partition=serc,normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#
# Extract all 24 LES runs (pit-2 crop) to Oak; about 4.5 min per run, about 2 h total.
# Existing output files are skipped, so a resubmission only does what is missing.
# Submit from the repo root:   sbatch scripts/sbatch/extract.sbatch

set -euo pipefail

ml purge
ml math devel viz
ml python/3.12.1 physics py-scipy/1.12.0_py312 py-numpy/1.26.3_py312 py-h5py/3.10.0_py312 \
   py-matplotlib/3.8.3_py312 py-pytorch/2.4.1_py312
export PYTHONPATH="$PWD/src:$PYTHONPATH"

export PINN_DATA_ROOT="$OAK/anm49/methane-plume-pinn/data"
export HDF5_USE_FILE_LOCKING=FALSE

python3 scripts/extract.py --all "$@"
