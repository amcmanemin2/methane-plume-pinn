# Methane Plume PINN

A physics-informed neural network for methane plume dispersion, following Chapter 4 of
Meng Jia's 2025 Colorado School of Mines thesis, *Dispersion Modeling and Sensor Placement
for Characterizing Methane Emissions on Oil and Gas Sites*. Course project for CEE 292.

The model solves the atmospheric advection–diffusion PDE directly, so it avoids the
Gaussian plume/puff assumptions and empirical dispersion parameters, and it estimates the
emission source term (location and rate) jointly with the concentration field.

## Status

Scaffold only. The network, PDE residual and loss terms are not implemented — those
follow the chapter's formulation and the `physics`, `domain` and `loss_weights` blocks in
[configs/default.yaml](configs/default.yaml) are deliberately empty until then.

`scripts/train.py` currently runs an environment check: it resolves a device, reports the
configured paths, and verifies that higher-order autograd is accurate, which is the one
capability a PINN cannot work without.

## Layout

```
configs/          run configuration (YAML)
src/methane_pinn/ library code
scripts/          entry points
scripts/sbatch/   Slurm submission scripts
data/             input data (gitignored; on Sherlock this lives on Oak)
runs/             checkpoints, logs, figures (gitignored)
```

Reference PDFs live in `files/` and are gitignored — they are not redistributed here.

## Configuration

Nothing is hardcoded to one machine. Three environment variables control placement:

| Variable          | Meaning                              | Default         |
| ----------------- | ------------------------------------ | --------------- |
| `PINN_DATA_ROOT`  | where input data is read from        | `./data`        |
| `PINN_RUN_ROOT`   | where outputs are written            | `./runs`        |
| `PINN_DEVICE`     | `auto`, `cpu`, `cuda` or `mps`       | `auto`          |

## Running on Sherlock

Module and partition names change over time — confirm them with `ml spider python` and
`sh_part` rather than trusting the values below.

```bash
git clone <repo-url> && cd methane-plume-pinn

# No venv: torch, numpy, scipy and matplotlib come from modules.
ml purge
ml math devel viz
ml python/3.12.1 physics py-scipy/1.12.0_py312 py-numpy/1.26.3_py312 \
   py-matplotlib/3.8.3_py312 py-pytorch/2.4.1_py312
export PYTHONPATH="$PWD/src:$PYTHONPATH"   # the package is not pip-installed

# Verify the environment on a dev node before queueing a real job.
sh_dev
python3 scripts/train.py --check-only
```

Then submit with:

```bash
sbatch scripts/sbatch/train.sbatch configs/default.yaml my-run-name
```

Keep checkpoints on `$SCRATCH` and read data from `$OAK`; `$HOME`
has a small quota.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/train.py --check-only
```
