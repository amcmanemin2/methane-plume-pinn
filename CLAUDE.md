# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

The repository is a fresh start with no source code, build system, tests, or git history. Its only file is `Jia_dissertation_ch4.pdf`, the reference material for the project (course: CEE 292). There are no build, lint, or test commands yet. Add them here once a toolchain exists.

## Python environment (Sherlock)

The venv is `$GROUP_HOME/envs/cee292_venv`, built with uv. torch, numpy and scipy are not installed in it. They come from Sherlock modules through `PYTHONPATH`, so the venv only works with those modules loaded.

Every Slurm script (and every interactive session) must load the modules before activating the venv:

```bash
ml purge
ml math devel viz
ml python/3.12.1 uv physics gdal/3.10.2 py-h5py/3.10.0_py312 py-mpi4py/4.0.3_py312 \
   py-scipy/1.12.0_py312 py-numpy/1.26.3_py312 py-scikit-image/0.24.0_py312 \
   claude-code py-pytorch/2.4.1_py312
source "$GROUP_HOME/envs/cee292_venv/bin/activate"
```

- Use `python3`, not `python` (`/bin/python` is 2.7 on Sherlock). Check `python3 --version` prints 3.12.x after loading modules.
- Install with `uv pip install --no-deps -e .`. Never run a plain `uv pip install -e .` or `pip install -e .`: uv ignores the module packages and would download torch with several GB of CUDA wheels, and would pick numpy 2.x, which tries to compile from source on this old OS (glibc 2.17).
- Other new packages go in with `uv pip install <pkg>`, always together with the pins `"numpy==1.26.3" "scipy==1.12.0"` (the module versions). Check with `--dry-run` first. If a package has no wheel for this OS, pin it to an older version rather than letting uv compile it.
- uv's cache is on `$SCRATCH` (`UV_CACHE_DIR`), so it does not use `$HOME` quota; no need for `--no-cache`.

## Project intent

The project is a physics-informed neural network (PINN) for methane plume dispersion. It is based on Chapter 4 of Meng Jia's 2025 Colorado School of Mines PhD thesis, "Dispersion Modeling and Sensor Placement for Characterizing Methane Emissions on Oil and Gas Sites". Per the thesis abstract, the PINN approach:

- Solves the atmospheric advection-diffusion PDE, so it does not need the Gaussian plume/puff assumptions or empirical dispersion parameters.
- Is meant to stay accurate under calm wind, where Gaussian models do poorly.
- Estimates the emission source term (location and rate) at the same time as the concentration field, which gives localization and quantification directly from sensor data.
- Is meant to be much cheaper than CFD, so it can support near real-time continuous monitoring.

The thesis also covers a fast Gaussian puff model and a genetic-algorithm sensor placement method (chapters other than 4). Those are the likely baselines for comparison.

When implementing, read the chapter PDF first. It defines the governing PDE, the boundary and initial conditions, the loss terms, and the data setup, and the code should follow it.
