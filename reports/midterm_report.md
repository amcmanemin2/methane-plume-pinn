# Physics-Informed Neural Networks for Methane Dispersion over an Open-Cut Coal Mine: Midterm Progress Report

CEE 292, November 2026

Author: [name]

## 1. Introduction

Continuous monitoring of methane emissions requires forward models that relate emission rates at known sources to concentrations at sensor locations. Gaussian plume and puff models are fast but rely on empirical dispersion parameters and perform poorly in calm winds and complex terrain. Large-eddy simulation (LES) resolves the turbulent transport but is too expensive for near real-time use. Physics-informed neural networks (PINNs) represent the concentration field with a neural network trained to satisfy the governing advection-diffusion equation, and have been proposed as a middle ground (Raissi et al., 2019).

This project builds on Chapter 4 of Jia (2025), who trained a PINN to solve the advection-diffusion equation for methane releases at an oil and gas test site. Her cases used a small flat domain (120 m by 75 m by 6 m), a constant wind, a known diffusion coefficient, and a 45 s time window. The goal of this project is to test whether the same approach can predict methane concentration fields over an open-cut coal mine, using turbulent LES fields as the reference. This report describes the data, the implementation, the verification checks, and the training experiments to date. The main finding so far is that the method does not transfer directly to the LES setting, and the experiments have identified specific reasons why.

## 2. Data

The reference data are FastEddy LES simulations of methane releases at the Muswellbrook coal mine in New South Wales. The simulation grid has 512 by 506 by 186 cells with a horizontal spacing of 20 m and terrain-following vertical levels. The terrain ranges from 109 m to 688 m above sea level, and the two mine pits are 190 m to 300 m deep. Each simulation contains 30 passive tracers, one per ground-level source, each released at 33.3 kg/h for one hour. This project uses the 15 sources in the northeastern pit. The 24 simulations cover three stability classes (daytime, convective, and stable) and eight wind directions. Output is saved every 120 s, which gives 30 snapshots during the release.

A common crop of 4 km by 4 km by 1 km above ground, centered on the pit, was extracted for every simulation. Because the tracers are passive and their transport in the LES is linear, the field for any combination of emission rates is an exact weighted sum of the single-source fields. This allows many source combinations to be generated without storing new fields. The runs were split by simulation, with 12 runs for training and 12 for testing. Validation combinations are drawn from the training runs, and no test data are used in training.

## 3. Methods

The PINN maps space and time, and in the original design also the 15 emission rates and 113 meteorological sensor readings, to a concentration. The network is a fully connected multilayer perceptron with six hidden layers of 256 units and hyperbolic tangent activations. Following Jia, the output passes through an exponential so that concentrations are positive, and all inputs are scaled to the interval from -0.5 to 0.5. To handle the deep pits, the network uses height above the local ground as its vertical input. The terrain is represented by a smooth cubic B-spline, so automatic differentiation applies the chain rule through the terrain and the PDE derivatives are taken in true physical coordinates.

The governing equation is the advection-diffusion equation with the LES wind field interpolated in space and time. Diffusion is described by a horizontal and a vertical eddy diffusivity for each stability class. Each source is represented by a Gaussian. The loss combines the squared PDE residual at collocation points, a zero-flux condition along the terrain normal at the ground, a zero initial condition, and in some experiments a data term on LES grid points. Collocation points are refreshed with the residual-based resampling method (R3) of Daw et al. (2023), in which points with above-average residual are kept and the rest are redrawn. The optimizer is Adam with an exponentially decaying learning rate and gradient clipping.

## 4. Verification

The implementation was first checked against cases with known answers. The PDE residual is zero for an analytic Gaussian puff solution, it equals the negative source term for a constant field, and derivatives through the full model with curved terrain agree with finite differences. Unit tests also cover the wind interpolation, source superposition, and the data split, and all of them pass.

A second check asked whether the LES itself satisfies the PDE that the PINN is asked to solve. The terms of the equation were computed from the LES fields by finite differences on the terrain-following grid. The best-fitting constant diffusivities were then found by non-negative least squares. The operator was validated on an analytic field on the real grid, with errors of 1.5 percent for first derivatives and 3 to 9 percent for second derivatives. Table 1 shows that even with the fitted diffusivities, the constant-coefficient equation leaves most of the LES transport balance unexplained. Repeating the check on 30-minute mean fields with the mean wind did not improve the agreement. This indicates that the 2-minute wind snapshots do not capture the turbulent fluxes that control the dispersion.

Table 1. Fitted eddy diffusivities and relative residual of the advection-diffusion equation on LES snapshots.

| Stability class | K_h (m²/s) | K_z (m²/s) | Relative residual |
|---|---|---|---|
| Daytime | 19.0 | 6.6 | 0.86 |
| Convective | 1.1 | 1.8 | 0.98 |
| Stable | 2.4 | 0.25 | 0.88 |

## 5. Initial Training Results

The first test was an overfitting test on data alone, using one simulation, ten source combinations, and a fixed set of 200,000 LES points. The best run reached a relative L2 error of 0.36 (R² of 0.87) after 5,000 epochs, against a target of 0.05. Removing gradient clipping or using a constant learning rate made the result worse. Analysis of the errors showed that 81 to 97 percent of the squared error came from the 1 percent of cells with the highest concentration, near the sources. On grid cells not seen during training the relative error was 0.96, which means the network memorized the near-source peaks rather than representing them. Excluding cells within 100 m of a source still left far-field errors of 0.39 on the training points and 0.71 on unseen cells.

Adding the PDE term made training diverge. Balancing the loss weights by gradient norm at the first epoch gave the PDE term a weight of about 10,000, and the predictions exceeded the LES concentrations by a factor of about 400. These results led to a change of scope. Instead of one network for all simulations and source combinations, the next step was to apply Jia's method to a single scenario with a fixed source and no data term, and to compare the result with the LES.

## 6. Single-Scenario Experiments

These experiments used the daytime simulation with a northerly wind and one source at 500 kg/h, with space and time as the only inputs and the diffusivities fixed at the fitted daytime values. Each experiment changed one setting relative to the previous one, as summarized in Table 2.

Table 2. Single-scenario experiments, each run for 2,000 epochs.

| Run | Change from previous run | Outcome |
|---|---|---|
| S1a-1 | Jia's setup on the 4 km crop over 1 h | Diverged; concentrations about 10^5 times too large |
| S1a-2 | Learning rate reduced tenfold | Same divergence, slower |
| S1a-3 | Domain reduced to 1 km by 1 km by 300 m over 10 min | Same divergence |
| S1a-4 | R3 resampling turned off | Stable, but PDE residual not reduced |
| S1a-5 | Source width increased from 10 m to 50 m | Stable, no plume formed |
| S1a-6 | Learning rate restored | Stable, no plume formed |
| S1a-7 | PDE time scale changed from 1 s to 600 s | PDE residual reduced tenfold, but a uniform field formed |

In the first three runs the network learned a field that was uniform in space and grew in time. A 10 m source is too small for the network to resolve in a kilometer-scale domain, so the network spreads it as a small, uniform rate of increase everywhere. R3 made this worse, because points at the unresolvable source peak always had the largest residual. These points filled the resampled pool until training effectively fitted only the source peak. Turning R3 off removed the divergence but did not let the network learn. With a wider source and the original learning rate, the PDE loss stayed equal to that of a zero field. This showed that the PDE term was too weakly weighted compared with the initial condition.

Scaling the residual by the natural time scale of the domain (S1a-7) allowed the network to reduce the PDE residual for the first time. However, the network then produced a nearly uniform concentration of about 100 times the reference scale. Without boundary conditions on the lateral and top faces, any constant field satisfies the PDE away from the source, and only the initial condition rules this out. Its weight had become small relative to the PDE term.

## 7. Discussion and Next Steps

Jia's setting differs from the LES problem in ways that matter for training. Her domain is small relative to the source, each of her ten time segments covers only a few seconds of transport, her wind and diffusivity are constant, her ground is flat, and her reference is the PDE solution itself. Her chapter also does not report the source width, network size, optimizer, loss weights, or the normalization of the concentration, and several of these proved critical here.

The next step is therefore to replicate Jia's three cases as closely as possible before returning to the LES. For constant wind and diffusivity, the exact solution for a continuous Gaussian source over flat ground can be computed as an integral of Gaussian puffs. This provides a reference that Jia did not report. If our implementation reproduces her cases, the remaining difficulties can be attributed to the LES setting, and the choices that make her cases train can be carried over to it. The single-scenario work will then continue with consistent weights for the initial and boundary terms and an inflow boundary condition, followed by a network with one output per source that uses the linearity of the equation to represent any source combination exactly.

## References

Daw, A., Bu, J., Wang, S., Perdikaris, P., and Karpatne, A. (2023). Mitigating propagation failures in physics-informed neural networks using retain-resample-release (R3) sampling. Proceedings of the 40th International Conference on Machine Learning, 7264-7302.

Jia, M. (2025). Dispersion modeling and sensor placement for characterizing methane emissions on oil and gas sites. PhD thesis, Colorado School of Mines, Chapter 4.

Raissi, M., Perdikaris, P., and Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics 378, 686-707.
