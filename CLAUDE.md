# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

The repository is a fresh start with no source code, build system, tests, or git history. Its only file is `Jia_dissertation_ch4.pdf`, the reference material for the project (course: CEE 292). There are no build, lint, or test commands yet. Add them here once a toolchain exists.

## Project intent

The project is a physics-informed neural network (PINN) for methane plume dispersion. It is based on Chapter 4 of Meng Jia's 2025 Colorado School of Mines PhD thesis, "Dispersion Modeling and Sensor Placement for Characterizing Methane Emissions on Oil and Gas Sites". Per the thesis abstract, the PINN approach:

- Solves the atmospheric advection-diffusion PDE, so it does not need the Gaussian plume/puff assumptions or empirical dispersion parameters.
- Is meant to stay accurate under calm wind, where Gaussian models do poorly.
- Estimates the emission source term (location and rate) at the same time as the concentration field, which gives localization and quantification directly from sensor data.
- Is meant to be much cheaper than CFD, so it can support near real-time continuous monitoring.

The thesis also covers a fast Gaussian puff model and a genetic-algorithm sensor placement method (chapters other than 4). Those are the likely baselines for comparison.

When implementing, read the chapter PDF first. It defines the governing PDE, the boundary and initial conditions, the loss terms, and the data setup, and the code should follow it.
