# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

The design is in `PLAN.md` (approved 2026-11-03; it records the decisions and the facts for the report). Implemented so far:
- `src/methane_pinn/runs.py`: the 24 runs and the train/test split.
- `extract.py`: raw LES to a cropped HDF5 file per run in `$OAK/anm49/methane-plume-pinn/data/les/`.
- `data.py`: superposition, combination tables, and the virtual sensors and lidar.

PINN: `physics.py` (terrain B-spline, LES wind interpolation, Gaussian sources, PDE/ground residuals), `model.py` (MLP with tanh and exp output), `sampling.py` (training points, R3 pool), `train.py`, `evaluate.py`, `plots.py`. The loss scales (`c_ref`, `tau`, `U_ref`, `loss_weights`) in `configs/default.yaml` are starting values. Tune them on the validation combinations only.

Commands (after `export PYTHONPATH="$PWD/src:$PYTHONPATH"`):
- Tests: `python3 -m unittest discover -s tests -v`. Set `PINN_DATA_ROOT` to a directory containing `les/*.h5` to run the file-based tests.
- Extract all runs: `sbatch scripts/sbatch/extract.sbatch`. Existing files are skipped.
- Quick extraction test: `python3 scripts/extract.py --run stable_N --frames 3 --out-dir $SCRATCH/methane-plume-pinn/test/les`.
- pytest is not installed. Use `unittest`.
- Sanity checks + smoke test (unit tests, `scripts/sanity.py`, `configs/smoke.yaml`, eval): `sbatch scripts/sbatch/sanity_smoke.sbatch`.
- Train and evaluate on a GPU: `sbatch scripts/sbatch/train_gpu.sbatch <run-name> [--note "what changed"] [--set key=value ...]`. Outputs go to `$SCRATCH/methane-plume-pinn/runs/<YYYYmmdd-HHMMSS>_<run-name>/` (config.yaml with seeds/git, git.patch, losses.csv every epoch, best.pt, last.pt, eval/). Each run appends to `runs/runs_log.csv` there; `LATEST` names the newest run dir. Evaluation only: `python3 scripts/evaluate.py [--run-dir <dir>] [--splits val test]`.
- Configs can start with `base: default.yaml` and override entries (see `configs/smoke.yaml`).
- Change one thing between runs and say what with `--note`.
- The `dev` partition rejects `sbatch` (interactive only). Use `-p gpu,serc`.

## Python environment (Sherlock)

No venv for now: everything comes from Sherlock modules. torch, numpy, scipy and matplotlib are provided by modules through `PYTHONPATH`; pyyaml is bundled inside the pytorch module; pandas currently comes from the user install in `~/.local` (no module provides it).

The default login shell already has the right modules loaded. In Slurm scripts, load them explicitly:

```bash
ml purge
ml math devel viz
ml python/3.12.1 physics py-scipy/1.12.0_py312 py-numpy/1.26.3_py312 \
   py-matplotlib/3.8.3_py312 py-pytorch/2.4.1_py312
export PYTHONPATH="$PWD/src:$PYTHONPATH"   # the package is not pip-installed
```

- Use `python3`, not `python` (`/bin/python` is 2.7 on Sherlock). Check `python3 --version` prints 3.12.x.
- Do not load `gdal/3.10.2` (not used here, and it fails to load after `ml purge`). When one module in an `ml` line fails, the whole line fails silently, so none of the later modules load. Check imports after loading.
- Run the environment check with `PYTHONPATH="$PWD/src:$PYTHONPATH" python3 scripts/train.py --check-only`. Plain `PYTHONPATH=src` overwrites the module-provided PYTHONPATH, and h5py/torch then fail to import.
- The `py-h5py/3.10.0_py312` module is needed for HDF5 I/O. Add it to `ml` lines in Slurm scripts.
- Do not `pip install` / `uv pip install` anything without asking: uv ignores the module packages, would download torch with several GB of CUDA wheels, and picks numpy 2.x, which tries to compile from source on this old OS (glibc 2.17).

## Project intent

The project is a physics-informed neural network (PINN) for methane plume dispersion. It is based on Chapter 4 of Meng Jia's 2025 Colorado School of Mines PhD thesis, "Dispersion Modeling and Sensor Placement for Characterizing Methane Emissions on Oil and Gas Sites". Per the thesis abstract, the PINN approach:

- Solves the atmospheric advection-diffusion PDE, so it does not need the Gaussian plume/puff assumptions or empirical dispersion parameters.
- Is meant to stay accurate under calm wind, where Gaussian models do poorly.
- Estimates the emission source term (location and rate) at the same time as the concentration field, which gives localization and quantification directly from sensor data.
- Is meant to be much cheaper than CFD, so it can support near real-time continuous monitoring.

The thesis also covers a fast Gaussian puff model and a genetic-algorithm sensor placement method (chapters other than 4). Those are the likely baselines for comparison.

When implementing, read the chapter PDF first. It defines the governing PDE, the boundary and initial conditions, the loss terms, and the data setup, and the code should follow it.
