# PLAN — PINN for methane plumes over an open-cut coal mine (CEE 292)

Phase 1 output: data inventory and design. No PINN code yet. On approval this file is saved as `PLAN.md` in the repo.
Decisions confirmed with the user on 2026-11-03 are marked [decided].

## Context

**Goal (in one sentence):** test whether a PINN can predict methane concentration fields from meteorological data at a
few random sensor points, given known sources, and evaluate its output against the full LES fields.

Jia (2025, Ch. 4) solves the advection–diffusion PDE with a PINN for idealized cases with known sources. We apply her
method to turbulent FastEddy LES of methane releases at the Muswellbrook coal mine. The extension is a single PINN
conditioned on (a) which sources are emitting and at what rate and (b) meteorology measured at 20 virtual sensors. It is
trained on half the LES runs and tested on the other half with RMSE and R². Jia reports no RMSE or R²; she shows figures
only. So we compute these metrics ourselves, for her method applied to our data.

---

## 1. The data (read-only; nothing in the simulation trees is modified)

- **Raw LES:** `$SCRATCH/fasteddy/aus/runs/muswellbrook/<tree>/<run>/output/FE.<step>`, netCDF4 (xarray `h5netcdf`). The
  `fire_safety_odor_analysis` folder holds only reduced caches (column maxima of 4 fixed pit combinations), which cannot be split
  back into individual sources, so we read the raw files.
- **Grid:** 512 × 506 × 186, Δx = Δy = 20 m, about 10 × 10 km, periodic laterally. Terrain-following levels (about 10 m apart
  near the ground), top 2782 m. Terrain 109–688 m ASL; the pits are 190–300 m deep.
- **Time:** output every 120 s. Release t = 3600–7200 s, giving 30 snapshots per run during the release (t = 3720–7200 s).
- **Variables:** u, v, w [m/s], θ [K], ρ [kg/m³], perturbation pressure, SGS TKE, `AuxScalar_0..29`, plus topography and surface fields.
- **Tracers:** `AuxScalar_i` is a **mass concentration in kg m⁻³**. There are 30 ground-level sources: 0–14 in pit 1 (SW) and 15–29
  in pit 2 (NE). Each releases 33.33 kg/h for 1 h. All 30 are present in every run, each as its own tracer.
- **Scope [decided]: pit 2 only.** We use the 15 pit-2 sources (scalars 15–29, x 5370–5730 m, y 5630–5850 m) and ignore the pit-1 tracers.
  Because each source is a separate passive tracer, the pit-2 fields are unaffected by pit 1.
- **Size:** 7.8 GB per snapshot, 11.7 TB in total.
- **Runs used [decided]:** 24 = 3 stability classes × 8 wind bearings (N, NE, E, SE, S, SW, W, NW).

| Class | Tree | 10 m wind | Surface heat flux |
|---|---|---|---|
| Daytime | `idealized_auxscbc` | Ug 6 m/s | +0.15 K m/s |
| Convective | `idealized_convective_auxscbc` | ≈ 3.1 m/s | +0.15 K m/s |
| Stable (Pasquill F) | `idealized_stable_auxscbc` | ≈ 1.2 m/s | −0.008 K m/s |

- **Storage [decided]:** extracted training data goes on Oak, in `$OAK/anm49/methane-plume-pinn/data` (about 16 TB free there).
  Jobs stage what they need to `$SCRATCH` or `$L_SCRATCH` for fast reads. ⚠ The raw LES on `$SCRATCH` becomes eligible for purge
  90 days after its last modification (from about 2026-12-21 for the daytime runs). Copy anything you want to keep to Oak before then.

## 2. Jia's method (what we follow)

- **PDE:** ∂c/∂t + u·∇c − ∇·(K∇c) = s, with constant u and constant diagonal K in Jia's case.
- **Network:** MLP (x, y, z, t) → c, with tanh hidden layers and an exp output so that c > 0. Inputs are scaled to [−0.5, 0.5]
  and the PDE is rescaled to match. Depth, width and optimizer are not reported.
- **Source term:** each source is a sharp Gaussian of strength q_i.
- **Loss:** PDE residual + initial condition c(t=0) = 0 + zero vertical flux at the ground. Jia uses no data term.
- **Training:** 100k collocation points with R3 resampling: every epoch, points with residual above the mean are kept and the
  rest are redrawn at random. Seq2Seq training over 10 time segments, followed by a fine-tuning pass. Early stopping at a loss of 1e-3.
- **Test cases:** a 120 × 75 × 6 m site over 45 s, at Péclet numbers 3.6 and 960, with 1 and 2 sources.

## 3. Mapping Jia to our data

| Jia | Our setup | Why |
|---|---|---|
| Flat 120 m box | **One common crop for all runs**: 4 × 4 km centered on pit 2, at about (5560, 5740) m (200 × 200 columns), 0–1000 m above ground [decided] | Identical inputs and grid for every run, so one model covers all of them. 1 km takes in most of the plume, including the convective cases. |
| Flat ground | Height-above-ground coordinate (explained below) | The pits are up to 300 m deep. |
| Constant wind | 3D LES wind u, v, w, interpolated in space and linearly in time between the 2-min snapshots, used **only inside the PDE residual during training** [decided] | At test time the model sees only the sensors. |
| Constant K | 2 trainable constants (K_h, K_z) per stability class (6 numbers) | K stands in for the mixing that the 2-min wind snapshots miss. |
| No lateral BC | Lateral and top crop faces are pinned to the LES concentration (Dirichlet), by oversampling face cells in the data term | User choice; the crop faces are open boundaries. |
| No data | Data term: random LES grid points from every snapshot of the training runs [decided] | |
| Input (x, y, z, t) | (x, y, z_agl, t) + 15 source rates + u, v, w, θ at 20 sensors + an in-pit wind profile (u, v, w at 10–110 m) | The extension. |

**Height above ground (z_agl).** In the LES the ground is the terrain surface, which can sit about 300 m lower inside the pit than on the
rim. Feeding the network absolute height would put "the ground" at a different z in every column. We use the height above
the local ground instead: z_agl = z − h(x, y), where h is the terrain elevation. The ground is then always z_agl = 0, which makes the
ground boundary condition, the source heights and the sampling simple.

There is one subtlety. The PDE derivatives must be taken in true physical coordinates. Moving 20 m east at a fixed altitude on a
slope changes z_agl. To handle this, we write the model as c(x, y, z, t) = NN(x, y, z − h(x, y), t) and let PyTorch autograd apply the chain
rule. That requires h to be differentiable, so we represent the 20 m topography as a smooth bicubic spline evaluated in torch rather than
as a lookup table. The zero-flux condition at the ground is applied along the terrain normal, n ∝ (−∂h/∂x, −∂h/∂y, 1).

**Sensors.** There are 20 fixed virtual sensors at 10 m above ground [decided], all at the same locations in every run: 10 drawn at random within 500 m
of the pit-2 sources (inside and around the pit) and 10 at random elsewhere in the crop (fixed seed). From each snapshot we record u, v, w and θ,
giving 80 numbers.

**Wind-profile sensor (sodar/lidar-like):** one more sensor sits on a pit floor. It records u, v and w every 10 m from 10 to 110 m above ground,
i.e. 11 heights × 3 = 33 numbers. Location [decided]: the floor of pit 2, on the lowest terrain cell near the source centroid (5560, 5740 m),
moved off any source cell.

Together these give m(t), 113 numbers per snapshot, interpolated linearly in time. They are network inputs, normalized with training-set means and standard deviations.
Sensor methane is used only for evaluation and is **not** an input [decided].

## 4. Data augmentation (superposition)

**Assumption (stated explicitly).** Methane is treated as a passive scalar: it does not change the flow, and transport in the LES is linear in
concentration (upwind advection plus eddy diffusion, zero initial field, zero-flux ground). So within one run, the field for any set of
constant emission rates is the weighted sum of the single-source fields:

  c(x, t; q) = Σᵢ (qᵢ / 33.33 kg/h) · cᵢ(x, t)

This is exact for the LES. It does **not** hold across different runs (different turbulence) or for time-varying rates. Real methane
at high rates is also buoyant. That effect is absent here, and the report should say so.

**Sampling (simplified per your request):**
1. Choose the number of active sources K uniformly from 1 to 15, then pick K of the 15 pit-2 sources at random.
2. Draw the total rate Q uniformly from (0, 5000] kg/h.
3. Split Q randomly among the active sources (Dirichlet(1) weights). The other sources are set to 0.

**How the augmented samples are produced (no new files).** Per run we store only the 15 single-source fields, as one array
B[t, cell, i] (30 snapshots × cells × 15 sources). An augmented sample is just a list of 15 rates q. Its field at any snapshot is a weighted
sum, B[t] · (q / 33.33).

Example: scalar 18 at 500 kg/h plus scalar 22 at 1000 kg/h gives c = 15·c₁₈ + 30·c₂₂. For a training batch we draw a run, a snapshot,
some grid cells and a q, and compute only those values on the fly. Storing the combinations themselves would take about 8 GB each
per run. Instead we save the list of combinations with their seeds (a few kB), so every sample can be reproduced exactly.

## 5. Split: by run, alternating bearings between classes [decided]

Every combination within a run is built from the same 15 fields, so whole runs are held out. Half the runs are used for training, and the
held-out bearings differ between the three classes:

| Class | Train | Test |
|---|---|---|
| Daytime | N, E, S, W | NE, SE, SW, NW |
| Convective | NE, SE, SW, NW | N, E, S, W |
| Stable | N, SE, S, NW | NE, E, SW, W |

**Sample counts [decided]** (each sample is one source combination on one flow field: 30 snapshots × about 2.2M cells; only the rate lists are stored, with fixed seeds):

| Split | Flow fields (runs) | Combinations per run | Total samples |
|---|---|---|---|
| Train | 12 training runs | 2,000 | 24,000 |
| Validation | the same 12 training runs | 200 (disjoint from training combinations) | 2,400 |
| Test | 12 test runs | 100 | 1,200 |
| **Total** | **24** | | **27,600** |

- The space is effectively unlimited: 2¹⁵ − 1 = 32,767 on/off subsets per run, times a continuous total rate.
- Validation shares the training flows. It is used for early stopping and tuning only, and is expected to look better than test.
- The test set has 12 independent flow realizations. No data from the test runs is ever used in training.

## 6. Code structure (style of `files/pinn-examples/hw01`: an `nn.Sequential` PINN class, `physics_loss` and `boundary_loss` functions using `torch.autograd.grad`, a `train_*` loop with tqdm and a `losses` dict, matplotlib plots)

```
src/methane_pinn/
  runs.py       registry of the 24 runs, train/test bearings, pit-2 constants
  extract.py    raw LES → per-run file on $OAK: the 15 pit-2 tracer fields, u v w on the crop (30 snapshots), the 20 sensor series + the pit-2 wind profile, topography
  data.py       load the extracted runs, superpose(q), combination sampler (§4), run split (§5), sensor normalization
  model.py      PINN: MLP over [x, y, z_agl, t, q/Q_max, m(t)] → exp(c*); input scaling as in Jia eq. 4.3; smooth terrain h(x, y)
  physics.py    Gaussian sources, PDE residual (LES wind + trainable K), ground zero-flux loss, initial-condition loss
  train.py      training loop: PDE + data (incl. faces) + ground + IC losses, R3 resampling, checkpoints on $SCRATCH
  evaluate.py   RMSE, R², relative L2: per snapshot, on the time mean, and at the sensors
  plots.py      Jia-style xy/yz slices (LES | PINN | error), metric-vs-time curves, parity plots, loss curves
scripts/  extract.py, train.py (extends the existing check), evaluate.py + sbatch files
configs/  default.yaml (fills the empty domain/physics/loss_weights blocks)
tests/    superposition exactness; PDE residual ≈ 0 for an analytic constant-wind Gaussian puff; terrain derivative vs finite differences
```

Builds on the existing `src/methane_pinn/{paths,device}.py` and `scripts/train.py --check-only`. Seq2Seq time segmentation (Jia) is added
only if training over the full 1 h window does not converge. There is no separate per-scenario reproduction stage [decided].

## 7. Metrics [decided]

Metrics are computed on the 12 test runs × 100 combinations, over the full crop:
- **Per snapshot:** RMSE, R² and relative L2 at each of the 30 snapshots, reported as curves over time and as means.
- **Time mean:** the same metrics on the release-window mean field.
- **At the sensors:** the methane time series at the 20 sensors.

All results are broken down by stability class and bearing.

## 8. Compute and memory

| Step | Estimate |
|---|---|
| Extraction (measured) | Crop = 200 × 200 × 111 levels (level 111 is the first fully above 1000 m AGL) = 4.4M cells; 31 frames (t = 3600–7200 s; the first frame is tracer-free). About 8.5 s per frame, about 4.5 min per run, so all 24 runs go in **one** job (`scripts/sbatch/extract.sbatch`, 1 CPU, 8 GB, about 2 h). lzf-compressed output about 4.4 GB per run, about 105 GB in total on Oak. |
| Subsampling | Required. The full LES grid (48M cells) cannot be used directly. We use the crop, plus random sampling of grid points in each batch. The PINN itself is mesh-free. Training reads a 40 m-coarsened copy of the 12 training runs (every 2nd column: about 2 GB per run, about 25 GB in RAM on a GPU node). Evaluation uses the full 20 m crop. |
| Training | MLP of about 6 × 256 with 132 inputs (4 coordinates + 15 rates + 113 met values). Second derivatives are needed only with respect to x, y, z, t. Batch of about 20k collocation + 20k data + 5k ground/IC points: about 50–100 ms per step on an A100/V100. About 150k steps ≈ 3–5 GPU-h per training run, under 10 GB of GPU memory. Including tuning (about 10 runs): about 50 GPU-h on `gpu` or `serc` (`-G 1`, explicit `--time`). |
| Evaluation | 12 runs × 100 combinations × 30 snapshots × 2.2M cells, in batched forward passes: about 1–2 GPU-h. |

## 9. Facts to keep for the report

- **Extraction checks (measured):** extracted fields are bit-identical to the raw LES. Crop mass / emitted = 0.97–0.99 early in the release. The terrain-following levels satisfy z_agl = η_k·(2782 m − h) to within 5e-5, so wind interpolation can use regular (x, y, η) index space. Lidar on the pit-2 floor at (5630, 5770) m, 108.9 m ASL (the domain minimum).

- **FastEddy setup:** FastEddy v5.0.1 + aux-scalar vertical-BC patch; RK3 time stepping, dt = 0.02 s; hybrid 5th/6th-order momentum
  advection; first-order upwind tracer advection; Lilly/Deardorff TKE SGS model (c_k = 0.10).
- **Surface and site:** z0 = 0.12 m; latitude −32.24°; 400 m damping layer; periodic lateral boundaries.
- **Mass closure:** individual tracers 0.93–1.02 of the emitted mass.
- **Methane properties:** source gas is 91.2 % methane; ρ_gas = 0.739 kg/m³, ρ_air = 1.204 kg/m³.

Caveats to state in the report:
- The upwind numerical diffusion is about 10 m²/s, which exceeds the physical mixing in stable runs.
- The stable layer is only a few grid cells deep.
- 2-min wind snapshots cannot resolve the eddies, so K absorbs that missing mixing.
- Buoyancy is neglected.

Each training run saves its config, seed, git hash and job ID next to its outputs.

## Verification (Phase 2)

1. `PYTHONPATH=src python3 scripts/train.py --check-only` on a GPU node.
2. Run `tests/` (superposition exactness, analytic puff residual, terrain chain rule).
3. Extract one run. Check that the crop's tracer mass is consistent with 33.33 kg/h per source, then run the 24-task array.
4. Overfit test: train on 1 run with 1 source. It should reach R² > 0.9 on its own data. Then train on the full 12-run setup.
5. Evaluate on the 12 test runs and produce the tables and figures.

## Open items
None. All design choices are confirmed.

## Amendment (2026-11-04): change of scope
The sanity checks showed that the full surrogate of §3–§6 cannot be trained as set up: the MLP cannot overfit one run, and the PDE does not describe the LES (FD residual 0.86–0.98). Training now proceeds in stages that keep the PINN code:
1. Jia's method per scenario (one training run + one source combination), first PDE-only, then with the data term.
2. One PINN per run with 15 unit-source outputs (superposition).

The met-sensor conditioning is a later extension. Evidence, decisions and every run are in `TRAINING_LOG.md`.
