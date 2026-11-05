# Training log

The record of the training process for the report: what was run, what changed between runs and why, and what happened. Each entry changes **one thing** relative to the entry it builds on. Run directories are under `$SCRATCH/methane-plume-pinn/runs/`. `runs_log.csv` there lists every run with its job ID, git hash and note. Each run dir also has `config.yaml` (full config, seeds, git hash) and `git.patch` (uncommitted code at run time).

## Baseline setup (decided 2026-11-04)

| Item | Setting |
|---|---|
| Network | MLP 6 × 256, tanh, exp output; inputs (x, y, z_agl, t) scaled to [−0.5, 0.5], 15 rates / Q_max, 113 normalized met inputs |
| Init | Xavier-normal weights, zero biases; output bias = log(mean LES c on plume cells / c_ref) from training data |
| Optimizer | Adam (default betas), lr 1e-3, exponential decay to 1e-5 over the step budget, grad-clip 1.0 |
| Loss | data (MSE / c_ref² + 0.1 × log-MSE) + PDE + ground zero-flux + IC; weights balanced by gradient norm at epoch 1, then frozen |
| Points per epoch (Jia) | 100k data, 100k collocation (R3 every epoch), 100k ground, 100k IC; up to 50k epochs; early stop at 1e-3 |
| Time window | full release, 3600–7200 s (no Seq2Seq) |
| Precision, seeds | float32; run seed 0; combination seeds 1/2/3; val points 12345; PDE monitor 54321; output bias 777; sensors 2026 |

Overfit test setup (sanity check 4): daytime_N, first 10 training combinations, fixed set of 200k LES points over all 30 snapshots (50% plume cells, 10% crop faces). Full batch every epoch. Pass criterion: relative L2 < 0.05 (R² > 0.997) on that set.

## Entries

### E0. First GPU smoke run with the earlier code (job 50990517, 2026-11-03)
- Setup: 3 daytime runs, 300 steps, fixed weights (data 1, pde 0.01, bc 1, ic 1), output bias −5.
- Result: the output collapsed to a constant (val R² −0.006, unchanged between checks). The loss terms differed by about 8 orders of magnitude (PDE ≈ 3e3 and rising, data ≈ 1.5, ground ≈ 1e-5, IC ≈ 1e-9).
- Diagnosis:
  - the exp output starting at e⁻⁵ gives vanishing gradients;
  - clipping lets the PDE term dominate the step;
  - the LR decayed fully within 300 steps.
- Led to the baseline setup above (data-based output bias, balanced weights, logging every epoch).

### E1. Sanity checks + smoke test with the baseline (job 51102103, 2026-11-04)
Run dirs: `20261104-121323_sanity/`, `20261104-123129_smoke/`.
- Unit tests 26/26 pass. Leakage and normalization checks pass. Data points reach a scaled z_agl of 0.735, i.e. the top grid level is at 1002–1250 m AGL while the PDE domain ends at 1000 m. That is left for a later entry.
- PDE fit on the LES field by finite differences (constant K, 2-min wind snapshots). The relative residual is 0.86–0.98 even with the best-fit K:

  | Class | K_h (m²/s) | K_z (m²/s) | Relative residual |
  |---|---|---|---|
  | Daytime | 19.0 | 6.6 | 0.86 |
  | Convective | 1.1 | 1.8 | 0.98 |
  | Stable | 2.4 | 0.25 | 0.88 |

- **4a, data-only overfit: fail.**
  - Relative L2 0.355 (R² 0.873) after 5000 epochs.
  - The gradient norm was 20–80 throughout against clip 1.0, so clipping was active at every step.
- **4b, all terms: diverged.**
  - Balanced weights at epoch 1: pde 1.3e4, bc 5.6e5. The PDE and ground gradients are tiny at init, because the network cannot change the source term.
  - The data loss went from 4.9 to 4.1e5 (c ≈ 400× LES).
- Smoke test: the pipeline worked end to end; metrics meaningless (inherits 4b).
- Full-size timing: 0.60 s/epoch, 17.4 GB peak, about 8.3 h for 50k epochs.

### E2. Overfit 4a without gradient clipping (job 51113784, 2026-11-04)
Run dir: `20261104-133542_E2_overfit_a_noclip/`.
- Change from E1-4a: `training.clip_grad = null` (no clipping). Everything else is identical.
- Hypothesis: clipping at 1.0, with gradient norms of 20–80, shrinks every step by 20–80×, so 5000 epochs is too short.
- **Result: rejected, much worse.** Relative L2 1.035 (R² −0.08), against 0.355 (R² 0.873) with clipping.
  - The loss plateaued at about 2.86 for the first 750 epochs (E1 was already at 2.42 by epoch 750).
  - At epoch 837 a gradient spike (norm 5.4e6, loss 4.9e4) threw the network into a state it never recovered from.
- E1 had the same kind of spike (epoch 964: gradient norm 4.3e4, loss 117), but clipping limited the step and the run recovered within a few epochs.
- Conclusion: the exp output makes the loss explode whenever log c overshoots, so clipping is needed as a guard. The large gradient norms in E1 come mostly from these spikes, not from steady clipping of useful steps. **Clipping stays at 1.0.**

### E3. Overfit 4a with a constant learning rate (job 51114847, 2026-11-04)
Run dir: `20261104-134739_E3_overfit_a_constlr/`.
- Change from E1-4a: `training.final_lr = 1e-3`, i.e. no LR decay; constant Adam 1e-3, as in the class example (hw01). Clipping is back at 1.0. Everything else is identical to E1.
- Hypothesis: E1 was still improving at epoch 5000, but the LR had decayed to 1e-5 (100× smaller), so the run stalled from lack of LR rather than capacity.
- **Result: rejected.**
  - Final relative L2 0.556 (R² 0.69). The best checkpoint (epoch 3000) reached relative L2 0.469 (R² 0.78), against E1's 0.355.
  - The loss oscillates: 1620 epochs were more than 2× the running minimum, and the median gradient norm was 6–33.
  - It briefly reached loss 0.089, lower than E1's final 0.438, so the network can go lower, but at 1e-3 it does not settle.
- Conclusion: LR decay is needed. **The E1 schedule (1e-3 → 1e-5) stays.**

### D1. Where the overfit error is (diagnostic only, no training; 2026-11-04)
The E1 and E3 `best.pt` checkpoints were evaluated on (i) their own 200k training points and (ii) 200k fresh grid cells from the same run, the same 10 combinations and the same snapshots.

| Checkpoint | Point set | Relative L2 | SSE share of the top 1% of cells by LES c | SSE share of points < 20 m AGL (4% of points) | Median abs. log₁₀ error (c > 1e-8) |
|---|---|---|---|---|---|
| E1 | training points | 0.355 | 81% | 43% | 0.47 |
| E1 | fresh cells | 0.958 | 97% | 91% | 0.48 |
| E3 | training points | 0.469 | 92% | 72% | 0.56 |
| E3 | fresh cells | 0.938 | 98% | 93% | 0.57 |

- The linear-space error, and so RMSE, R² and relative L2, is almost all in the near-source peak cells close to the ground. The top 0.1% of cells hold 93% of Σc².
- The network fits those peaks only at the specific training points it saw (relative L2 0.36). At other cells of the same fields it misses them (0.96), so it is memorizing the peaks, not representing them.
- The rest of the plume is fitted consistently, in-sample and on fresh cells: a factor of about 3 (0.47 dex) median error.
- Interpretation: the near-source peak is about 20–60 m wide in a 4 km crop, i.e. about 0.005–0.015 in scaled coordinates. A plain tanh MLP has a spectral bias toward smooth functions, so it cannot represent features that sharp. Changing the optimizer will not get linear relative L2 below 0.05.
- This also matters for the report: PLAN.md §7 metrics in linear space are dominated by a handful of cells next to the sources.

### Revised overfit criterion (decided 2026-11-04, after D1)
The near-source peaks dominate the linear metrics, so the overfit criterion and all evaluations now also report:
- **Far field:** cells more than 100 m horizontally from every pit-2 source. **Pass if far-field relative L2 < 0.05.**
- **Log space:** on cells with LES c > 1e-8 kg m⁻³, the median |log₁₀(PINN/LES)| and the fraction of cells within a factor of 2.

The PLAN.md linear metrics on all cells are kept alongside. Constants: `NEAR_SOURCE_M`, `LOG_FLOOR` and `OVERFIT_PASS_REL_L2` in `src/methane_pinn/evaluate.py` (function `point_metrics`).

### D2. Rescore of E1–E3 with the revised criterion (diagnostic only, no training; 2026-11-04)
Results saved in `D2_rescore_E1-E3.json`. Point sets as in D1. The far field is about 98% of the points.

| Checkpoint | Point set | Far-field relative L2 | Far-field R² | All-cells relative L2 | Median abs. log₁₀ error | Within 2× |
|---|---|---|---|---|---|---|
| E1 best = last (epoch 5000) | training points | 0.636 | 0.58 | 0.355 | 0.47 | 34% |
| E1 | fresh cells | 1.004 | −0.04 | 0.958 | 0.48 | 34% |
| E2 last (epoch 5000) | training points | 1.263 | −0.65 | 1.035 | 0.60 | 26% |
| E3 best (epoch 3000) | training points | 0.614 | 0.61 | 0.469 | 0.56 | 29% |
| E3 last (epoch 5000) | training points | **0.387** | 0.85 | 0.556 | **0.27** | **54%** |
| E3 last | fresh cells | 0.707 | 0.48 | 0.942 | 0.30 | 51% |

- **No run passes the revised criterion.** The far field is not fitted either, even at the training points (best 0.39). So the sharp near-source peaks are not the only problem.
- Far-field error is much larger on fresh cells than on training points (E1: 1.00 vs 0.64; E3: 0.71 vs 0.39). Even away from the sources, the network partly memorizes points instead of representing the field.
- E3 (constant LR) has the worse all-cells score but the better far-field and log-space fit. Under the decaying schedule, E1 spent its last epochs fitting the peak cells that dominate the linear MSE, at the expense of the rest of the plume.
- **Checkpoint selection:** `best.pt` is chosen by the all-cells RMSE, which tracks the peaks. Under the revised criterion E3's `last.pt` beats its `best.pt`.
- Conclusion: the plain 6 × 256 tanh MLP, trained on the linear MSE, does not reach the revised criterion either. The candidates left are an architecture change (Fourier-feature inputs), a loss change (weight the log term more, or down-weight the near-source peaks), or more capacity and epochs. Each needs approval.

## Change of scope (decided 2026-11-04)
**Why.** E1–E3, D1 and D2 show that the full surrogate (one network for 12 turbulent flows × any 15-source mix × 113 met inputs, trained against LES snapshots) cannot be trained as set up:
- A plain MLP cannot overfit even one run with 10 combinations.
- The PDE does not describe the LES. The FD relative residual is 0.86–0.98 on 2-min snapshots. It is 0.99 / 1.00 / 0.70 (daytime / convective / stable) on the 30-min mean field with the mean wind (last 30 min of the release, sources 0, 7, 14); there the NNLS fit pushes K to 0 for daytime and convective. The 30-min mean also is not steady: it changes by about 40% between its two halves.
- The FD operator was checked on an analytic Gaussian (300 m × 150 m) placed on the real terrain-following grid. Relative errors: c_x 1.5%, c_z 0.3%, ∇²_h c 8.7%, c_zz 2.9%. So the mismatch is a property of the LES, not of the check.

Jia's setting is much simpler: one network per scenario, constant wind, known K, PDE-only loss, and the PDE solution as the truth.

**Decision.** Keep the PINN code (MLP, residuals, training loop) and shrink the problem:
- **Stage 1:** Jia's method per scenario (one training run + one fixed source combination; inputs x, y, z_agl, t).
  - S1a: PDE + ground + IC only, as in Jia.
  - S1b: add the data term.
- **Stage 2:** one PINN per run with 15 unit-source outputs (superposition).
- The met-sensor conditioning of PLAN.md becomes a possible later extension.

Test runs are not used in stage 1.

**Code added for stage 1:**
- `network.inputs: coords`;
- `physics.K_trainable` and `physics.K_by_class`;
- `physics.c_ref: auto` (the scenario's mean LES plume concentration; a scale only);
- the `scenario:` config block;
- the data-free mode (`loss_weights.data: 0`);
- `training.best_metric`;
- the `scenario` evaluation split;
- far-field and log metrics in every evaluation.

Config: `configs/scenario_jia.yaml`.

### S1a-1. Jia's method, daytime_N, source 7 at 500 kg/h, 2000-epoch stability test (job 51141533, 2026-11-04)
- Setup: `configs/scenario_jia.yaml`.
  - Inputs (x, y, z_agl, t); 6 × 256 tanh MLP with exp output.
  - Loss PDE + ground + IC, weights 1, no data term.
  - K fixed at (19.0, 6.6) m²/s, the daytime FD fit.
  - LES wind.
  - 100k collocation / ground / IC points; R3 every epoch; Adam 1e-3 → 1e-5 over the 2000 epochs; clip 1.0.
  - c_ref = scenario mean plume c (4.5e-8 kg/m³ in the CPU dry run at stride 4).
- Purpose: check stability and the loss balance before the full 50k-epoch run. The LES is used only for monitoring and evaluation.
- Run dir: `20261104-175138_S1a-1_daytime_N_src7/`. c_ref = 4.68e-8 kg/m³. 0.50 s/epoch, 16.3 GB peak.
- **Result: diverged.** The network does not even satisfy its own PDE.

  | Epoch | PDE (R3 pool) | Ground | IC | PDE on fixed points | Gradient norm |
  |---|---|---|---|---|---|
  | 1 | 2.0e3 | 1.9e-4 | 1.0 | n/a (first check at epoch 200) | 8.8 |
  | 200 | 7.3e4 | 7.2e3 | 9e-11 | 8.2e3 | 3.4e6 |
  | 1000 | 5.6e4 | 9.6e2 | 3.3e2 | 4.0e4 | 1.4e6 |
  | 2000 | 5.3e4 | 5.6e2 | 2.5e2 | 3.9e4 | 5.4e5 |

  - The predicted c grew to about 2e-2 kg/m³ everywhere, about 10⁵ × the LES. Median |log₁₀ error| is 5.4, and 0% of cells are within 2×.
  - Mass check: 500 kg/h for 1 h is 500 kg, which spread over the whole 1.6e10 m³ crop is 3e-8 kg/m³. So the prediction is about 6e5 × too large: a non-physical state, not a turbulence or K mismatch.
  - All terms rose from the first epochs, including the ground and IC terms.
- **Diagnosis (not acted on):** this is a PINN optimization failure on the PDE alone, separate from the PDE–LES mismatch. Likely causes:
  - (i) Adam at 1e-3 is too aggressive for this stiff loss. Adam normalizes the step, so clipping hardly limits it.
  - (ii) The scale range: the Gaussian source (σ 10 m / 5 m) is about 0.0025 of the 4 km crop. Jia's 120 m box with a 1 m-scale source has a ratio about 3–4× smaller.
  - (iii) The time horizon: 3600 s against an advective time of about 800 s across the crop. This propagation-failure regime is why Jia uses Seq2Seq. PLAN.md §6 already names Seq2Seq as the fallback if the full window does not converge.

### S1a-2. As S1a-1 with a 10× lower learning rate (job 51148543, 2026-11-04)
- Change from S1a-1: Adam lr 1e-4 → 1e-6 (`training.learning_rate=1e-4`, `training.final_lr=1e-6`). Everything else is identical.
- Hypothesis: every term rising from the first epochs points to optimizer instability. Adam's step size is about lr regardless of the gradient size, so 1e-3 overshoots on this stiff loss.
- Run dir: `*_S1a-2_daytime_N_src7_lr1e-4/`. 0.29 s/epoch.
- **Result: rejected, same divergence, only slower.**
  - The PDE loss on fixed points rose steadily: 3.5e3 (epoch 200) → 1.6e4 (epoch 2000).
  - The predicted c grew to about 9e-3 kg/m³; median |log₁₀ error| 4.9 against the LES.
  - The IC held (about 1e-9) and the ground term fell.
  - The direction of training is wrong, not its step size.

### D3. What the S1a PINN learns (diagnostic on S1a-2 last.pt; no training; 2026-11-04)
At 20k uniform collocation points:
- **The field is uniform in space and grows in time.** Median c is about 5.5–6.0e-3 kg/m³ at every height band (0–1000 m) and every distance from the lateral faces.
- In time, the median c is 1.4e-11 (3600–4200 s), 2.8e-4 (4200–5400 s), 1.1e-2 (5400–6600 s) and 1.5e-2 (6600–7200 s).
- RMS of the PDE terms: c_t 5.9e-6, u·∇c 5.3e-7, diffusion 3.5e-8 kg m⁻³ s⁻¹; residual 5.9e-6. The source is 0 at these uniform points. corr(c_t, −u·∇c) = 0.06, so there is no advective structure.
- **Mechanism.**
  - The Gaussian source (σ 10 m / 5 m, about 0.25% of the crop) is far below what the smooth tanh network can resolve on [−0.5, 0.5] inputs (spectral bias, as in D1).
  - About 20% of the collocation points are placed next to the sources (`p_source = 0.2`), and R3 keeps the high-residual ones, which are the same points.
  - With a spatially smooth c, the L2-optimal way to meet "LHS = s > 0" at those points is a small positive c_t everywhere: about the near-source share of the points × the source strength. For a share of 0.2 that is about 0.2 × s̄ ≈ 2e-6, the same order as the measured 5.9e-6.
  - The IC pins c = 0 at 3600 s, so c then grows uniformly in time.
- Conclusion: the source term cannot be represented at this domain scale, and the PDE loss turns that into uniform mass creation. Remedies must make the source resolvable relative to the network, or give the network resolution at the source scale. Candidates: a Jia-sized sub-problem (smaller domain and time window), a wider source, or Fourier-feature inputs.

### S1a-3. Jia-sized sub-problem (job 51153165, 2026-11-04)
- Change from S1a-2: the scenario is restricted to a box around the active source (`scenario.subdomain`):
  - 1 × 1 km (x 5030–6030 m, y 5250–6250 m), centred on source 7;
  - 0–300 m AGL;
  - the first 10 min of the release, 3600–4200 s (frames 0–5).

  The model's input scaling, the collocation / ground / IC sampling, the monitoring data cells and the evaluation all follow the box. Everything else is as in S1a-2, including Adam 1e-4 → 1e-6.
- Scale check against Jia: source width / domain width is about 1% (S1a-1/2: 0.25%). The horizon is about 3 advective transits of the box (600 s × about 5.6 m/s / 1 km), against about 3 for Jia (45 s × 8 m/s / 120 m) and about 5 for S1a-1/2 (3600 s over 4 km).
- Hypothesis (from D3): with the source resolvable relative to the domain, the network can represent the near-source forcing instead of spreading it as a uniform c_t.
- Note: the lateral and top faces of the box have no boundary condition, as in Jia. Within 10 min the plume leaves through the downwind face.
- Run dir: `*_S1a-3_daytime_N_src7_box1km/`. c_ref = 1.52e-7 kg/m³, 21,502 data cells (stride 2), 0.33 s/epoch.
- **Result: rejected, same divergence.**
  - The PDE loss on fixed points rose 5.2e2 (epoch 200) → 3.3e3 (epoch 2000); the R3 pool loss rose 1.9e2 → 6.8e3.
  - The predicted c grew to about 3e-3 kg/m³ (median |log₁₀ error| 4.3, scenario evaluation 1.8 per snapshot).

### D4. What the S1a-3 PINN learns, and the role of R3 (diagnostic on S1a-3 last.pt; no training; 2026-11-04)

| Point set (20k) | Median c (kg/m³) | RMS c_t | RMS u·∇c | RMS diffusion | RMS source | RMS residual |
|---|---|---|---|---|---|---|
| Uniform in the box | 2.4e-3 | 9.1e-6 | 7.2e-7 | 7.2e-8 | 2.3e-8 | 8.9e-6 |
| Near the source (`p_source = 1`) | 2.5e-3 | 9.2e-6 | 2.2e-8 | 2.0e-8 | 4.7e-6 | 8.3e-6 |

PDE terms in kg m⁻³ s⁻¹.

- As in D3, the field is uniform in space (2.4–2.5e-3 at 0–30, 30–100 and 100–300 m AGL). It grows in time: 4.9e-4 (3600–3750 s) → 4.6e-3 (4050–4200 s).
- c_t is the same at uniform and near-source points. The network has no local response to the source even in the 1 km box.
- Mass check: by 4200 s, 83 kg have been emitted, so a uniform spread over the 3e8 m³ box would be at most 2.8e-7 kg/m³. The prediction implies about 10⁴ × more mass.
- **This is not the L2 compromise suggested in D3.** For a field the network cannot localize, the L2 optimum on the training distribution (80% uniform points, 20% near-source) would be c_t ≈ 0.2 × the mean source ≈ 6e-7. Even c = 0 would give a lower loss than the learned state. The measured 9e-6 is about 15× the compromise, and the loss on the fixed points (that distribution) rose throughout training.
- **Mechanism: R3 collapses onto the unresolvable source peak.**
  - R3 keeps every collocation point whose residual is above the pool mean.
  - The network cannot represent the 10 m source, so the points at the source peak always have the largest residual. They are always retained, while uniform points are redrawn.
  - The pool thus fills with source-peak points, and training effectively fits only those. The best smooth fit there is c_t ≈ the peak source (about 1.8e-5 for 500 kg/h), applied everywhere; 9e-6 is that order.
  - The pool loss (6.8e3 ⇒ scaled residual about 82 ⇒ 1.25e-5 kg m⁻³ s⁻¹) is consistent with a pool at the peak.
- The same applies to S1a-1/2 (D3). So "spectral bias" alone (D3) was an incomplete explanation: R3 amplifies it into uniform mass creation.
- Note for the report: Jia's R3 assumes the residual can be reduced wherever it is high. With a source the network cannot resolve, that assumption fails.

### S1a-4. As S1a-3 with R3 off (job 51160436, 2026-11-04)
- Change from S1a-3: `training.r3 = false`. Each epoch draws a fresh set of 100k collocation points, 20% of them near the source (`p_source = 0.2`). Everything else is identical: the 1 km box, the first 10 min, Adam 1e-4 → 1e-6, PDE + ground + IC only.
- Hypothesis (D4): R3 collapses the pool onto the unresolvable source peak, which causes the uniform mass creation.
- Expected: no blow-up. A smooth field that still does not resolve the source peak (c_t about 0.2 × the mean source at most), with a falling PDE loss on fixed points.
- Run dir: `*_S1a-4_daytime_N_src7_box1km_noR3/`. 0.37 s/epoch.
- **Result: the R3 part of the hypothesis is confirmed; the field is still wrong.**
  - The blow-up is gone. The PDE loss on fixed points stays flat at about 2.0e2 (epoch 1: 1.9e2; epoch 2000: 1.97e2), against 5.2e2 → 3.3e3 with R3 (S1a-3). The predicted c is about 40× smaller than in S1a-3.
  - But the PDE loss does not fall either: the network cannot reduce the residual at all. The initial residual is the source term at the near-source points, which a smooth field cannot match.
  - The predicted c still drifts upward slowly (validation RMSE 2.1e-5 → 7.7e-5 kg/m³; median |log₁₀ error| 1.3 → 2.3). That is a weak version of the same uniform-growth compromise.
  - Scenario evaluation: per-snapshot relative L2 1.07 (all cells), 4.3 (far field), median |log₁₀ error| 1.3, 4% of cells within 2×.
- Conclusion:
  - R3 turned the unresolvable source into a runaway (D4 confirmed).
  - Without R3, the PINN is stable but learns nothing useful: with 10 m / 5 m Gaussian sources it cannot represent the forcing even at Jia's domain scale.
  - **The remaining blocker is source resolution**, not the optimizer and not R3. Candidates: a wider source (one config change) or Fourier-feature inputs (a small model change).

### Comparison with Jia's setup (for the report; 2026-11-04)

| | Jia (Ch. 4) | Ours (S1a-3/4) |
|---|---|---|
| Domain | 120 × 75 × 6 m, flat | 1 × 1 km × 300 m, real terrain |
| Time | 45 s in 10 Seq2Seq segments of 4.5 s + full-window fine-tune | 600 s, one window |
| Wind | constant, horizontal (8 or 0.3 m/s) | 3D LES wind, varying in space and time |
| K | known, 10 or 1 m²/s | fixed at the FD fit (19, 6.6) |
| Source | sharp Gaussian, 1 kg/h, z = 3 m (mid-height); σ not reported, about 5–10 m wide in her figures (our estimate) | σ 10 m / 5 m at about 4 m AGL, 500 kg/h |
| Points | 100k per segment, R3 | 100k over the whole window |
| Epochs | up to 50k per segment, early stop 1e-3 | 2000 (tests) |
| Validation | none against data (normalized figures only) | turbulent LES |

Not reported by Jia: σ, network depth and width, optimizer, learning rate, final loss values.

Main differences that explain S1a-1 to S1a-4:
- source width / domain is about 5% for Jia (from her figures), against 1% (box) and 0.25% (crop) for us;
- each Seq2Seq segment covers only about 36 m of transport;
- her space-time point density is about 10⁵–10⁶ × ours.

### S1a-5. As S1a-4 with a wider source (job 51164065, 2026-11-04)
- Change from S1a-4: Gaussian source widths `physics.sigma_h = 50 m`, `physics.sigma_z = 15 m` (were 10 m / 5 m). The source is now about 5% of the 1 km box, matching Jia's ratio as read from her figures. The near-source collocation points (`p_source = 0.2`, spread 2σ) widen with it. Everything else is identical: box, first 10 min, R3 off, Adam 1e-4 → 1e-6, PDE + ground + IC only, K fixed.
- Hypothesis (S1a-4): source resolution is the remaining blocker. With a resolvable source the PDE loss should fall, and a plume should form downwind of the source.
- Trade-off: the near field is smoother than the LES's 20 m source cell, so near-source errors are expected. The far-field and log metrics are the fair comparison.
- Run dir: `*_S1a-5_daytime_N_src7_box1km_noR3_sigma50/`. 0.33 s/epoch.
- **Result: stable and well scaled, but the network barely trains. No plume forms.**
  - With the wider source the scaled source term is far smaller, so the loss starts at 1.07 (PDE 0.049, IC 1.02) instead of about 190.
  - The IC term is fitted (1.0 → 1e-4) and the ground term falls (2.8e-3 → 3.6e-5).
  - The PDE loss barely moves: 0.0486 → 0.0476 on the training points, and 0.0519 → 0.0519 on the fixed points.
  - The gradient norm is tiny (about 2.5e-3 from epoch 400 on).
  - Scenario evaluation: per-snapshot relative L2 1.00 (all cells), 1.08 (far field); median |log₁₀ error| 0.69; 25% of cells within 2×.
  - These numbers are better than S1a-1 to S1a-4 only because the field stays near the c_ref scale; it does not resemble a plume.
- Field check on `last.pt` (20k uniform points below 60 m AGL after 3900 s; c / c_ref medians):

  | Region | Median c / c_ref |
  |---|---|
  | Within 75 m of the source | 1.81 |
  | Downwind 100–400 m | 1.30 |
  | Upwind 100–400 m | 1.46 |
  | Crosswind beyond 300 m | 1.37 |

  - The field has a faint bump at the source but no downwind plume: upwind is about the same as downwind.
  - RMS c_t 3.7e-9 against u·∇c 1.3e-10 and source 8.0e-9 kg m⁻³ s⁻¹. The network still answers the source with time growth, not with advection.
- Conclusion: source resolution plus no R3 removes the runaway. But with Adam 1e-4 → 1e-6 (lowered in S1a-2 to fight a blow-up that D4 later traced to R3), 2000 epochs move the network very little. It is under-trained rather than failing.

### S1a-6. As S1a-5 with the original learning rate (job 51244424, 2026-11-05)
- Change from S1a-5: Adam lr back to 1e-3 → 1e-5 (the E1 schedule and `configs/default.yaml`). It was lowered to 1e-4 → 1e-6 in S1a-2 to fight a blow-up that D4 traced to R3; R3 is still off. Everything else is identical: σ 50 m / 15 m, 1 km box, first 10 min, PDE + ground + IC only, K fixed, 2000 epochs.
- Hypothesis (S1a-5): the network is under-trained, not failing. With a 10× larger step the PDE loss should fall and a downwind plume should appear.
- Run dir: `*_S1a-6_daytime_N_src7_box1km_noR3_sigma50_lr1e-3/`. 0.48 s/epoch.
- **Result: rejected. Same as S1a-5: stable, no learning of the PDE.**
  - The PDE loss is 0.0486 → 0.0477 on the training points and 0.0524 → 0.0520 on the fixed points.
  - The IC and ground terms are about 1e-5 to 1e-6 from epoch 200 on.
  - Gradient norm 2e-4 to 3e-3.
  - Field (`last.pt`, below 60 m AGL after 3900 s; median c / c_ref): 1.17 near the source, 0.79 downwind, 0.91 upwind, 0.84 crosswind. Overall percentiles: p1 0.005, p50 0.11, p99 5.7. No downwind plume.
  - Scenario evaluation: per-snapshot relative L2 1.00 (all cells), 1.00 (far field), median |log₁₀ error| 1.45, 0% of cells within 2× per snapshot.

### D5. Why the PDE term does not train (S1a-5/6; no training; 2026-11-05)
- **The trained PDE loss equals that of a zero field.** With c ≡ 0 the residual is just −s, and the PDE loss on the fixed collocation points (100k, `p_source = 0.2`, seed 54321) is **0.0499**. S1a-5 and S1a-6 end at 0.048–0.052. The network has removed essentially none of the PDE residual.
- **Mechanism: loss scaling (conditioning).**
  - With τ = 1 s, the scaled PDE term starts at about 0.05, while the IC term starts at about 1 (c = c_ref at t = 3600 s).
  - The fastest way to lower the total loss is to make c small. The network moves to c ≪ c_ref (median 0.1 c_ref), where all derivatives are tiny, and the PDE term stays at its zero-field value.
  - A plume requires c / c_ref of about s_peak·σ / (U·c_ref) ≈ **28** near the source. The physical-unit operator with τ = 1 s rewards that only weakly: its time and advection derivatives are about c/(600 s) and c·U/L.
- The natural time scales of the box are L/U ≈ 190 s and T = 600 s, so τ = 1 s under-weights the PDE by a factor of τ² ≈ 4e4 to 4e5.
- Writing the PDE in Jia's scaled time t* = t/T (eq. 4.4 multiplied by T) is equivalent to τ = T = 600 s.

### S1a-7. As S1a-6 with the PDE time scale τ = 600 s (job 51248268, 2026-11-05)
- Change from S1a-6: `physics.tau = 600` s (was 1 s), the length of the box's time window. This is equivalent to writing the PDE in Jia's scaled time t* = t/T (eq. 4.4 × T), and to multiplying the PDE loss weight by τ² = 3.6e5. Everything else is identical: σ 50 m / 15 m, 1 km box, first 10 min, R3 off, Adam 1e-3 → 1e-5, ground and IC weights 1, K fixed, 2000 epochs.
- Hypothesis (D5): with τ = 1 s the PDE term is about 20× smaller than the IC term at the start and gives almost no gradient. With τ = T it dominates, so the network should reduce the PDE residual below the zero-field value and form a downwind plume.
- Expected first-epoch PDE loss: about 0.05 × 3.6e5 ≈ 1.8e4.
- Run dir: `*_S1a-7_daytime_N_src7_box1km_noR3_sigma50_tau600/`. 0.28 s/epoch.
- **Result: partly confirmed. The PDE is now being solved, but a new imbalance appears.**
  - First PDE loss 1.75e4, as expected. **The PDE loss on the fixed points falls about 10×** (1.4e4 at epoch 200 → 1.9e3 at epoch 2000, against the zero-field value of about 1.8e4). This is the first S1a run that reduces the PDE residual.
  - But the IC and ground terms grow: IC 1.0 → 40, ground 2.8e-3 → 690. Their weight (1) is now negligible next to the PDE term.
  - The field (`last.pt`, below 60 m AGL after 3900 s; nearest-cell LES at the same points):

    | Region | PINN c / c_ref | LES c / c_ref |
    |---|---|---|
    | Within 75 m of the source | 106 | 11.3 |
    | Downwind 100–400 m | 82 | 6.6 |
    | Upwind 100–400 m | 115 | 0.0 |
    | Crosswind beyond 300 m | 98 | 0.0 |

    At t < 3650 s the PINN has c / c_ref ≈ 5.4, where the IC requires 0.
  - Scenario evaluation: per-snapshot relative L2 8.9 (all cells), 65 (far field); median |log₁₀ error| 1.8.
- **Diagnosis: the PDE has a null space that only the IC pins down.**
  - The box has no lateral or top boundary condition (as in Jia), so any constant field satisfies the PDE everywhere except at the source.
  - With the PDE weighted about 3.6e5× more than the IC, the cheapest state is a nearly uniform field of about 100 c_ref, which violates the IC (about 5 c_ref at t = 3600 s). It reduces the PDE residual near the source by matching the source part-way.
  - The LES shows the expected plume: 11 near the source, 6.6 downwind, 0 upwind and crosswind.
- Options for S1a-8 (each one change):
  - (a) rescale the IC and ground weights consistently with τ (w_ic = w_bc = τ² = 3.6e5);
  - (b) add a zero-concentration inflow boundary condition on the box faces where the wind enters (clean air; the LES is 0 upwind). This removes the constant null space at all times.
