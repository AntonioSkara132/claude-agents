# Result: Episode 14 von Mises calibration (package `episode14_dynamics_density1200_gpu.zip`, 2026-10-12)

**Summary.** The package installs and validates cleanly on this GPU, but the fit cannot run as prescribed: 9 of the 11 chunks
cannot be trained on. Chunks 04, 05, 08, 09, 10 fail the initial-referenced depth term (the dough leaves its initial
footprint), and when that term is replaced by a frame-referenced one, chunks 02, 03, 04, 05, 07, 08, 09, 10 all fail with
non-finite adjoints at checkpoint-segment boundaries (global steps 7040, 5312, 3136, 4096, 3840, 320, 3776, 4032: every
one a multiple of the 64-step segment length; the same solver failure as on Episode 11). Only chunks 06, 11 (the package's
validation chunks) and 12 survive. A fit with chunk 12 as the only training chunk ran all 20 updates and overfits that
chunk: training objective 0.328 -> 0.238, mean validation objective 1.48 -> 1.56, held-out chunk 06 chamfer worse. The
README's f32 setting also fails the solver's checkpoint-recompute consistency check (chunk 09, step 7671, max difference
2.4e-4), so f64 with atomic P2G was used. **No Episode 14 parameters should be adopted from this run.** The blocker is the
segment-replay adjoint in TaichiDough.

## Setup actually run

- Package: 180 checksums OK; 177 data files installed into the existing checkout (c0fcb0d + uncommitted, default-off
  additions: `loss.depth_reference`, dataset schema v4) without overwriting anything. README validation step (no runtime)
  passed for all 11 chunks (`runs/episode14_transfer_check`).
- Because the spatula covers the dough during the motions, each chunk is scored only at its first and last frame
  (`scored_window.stride = end_frame - 1`), with `loss.depth_reference: "frame"` (current observed vs current rendered
  depth) instead of the initial-referenced depth term; depth 1.0 / coverage 0.5 / distance 0.5 otherwise as shipped.
- Dataset `dataset_episode14_dynamics_temporal_von_mises_1000pa_endframes_framedepth_train12only_v1.json`: training chunk 12;
  validation 02-11 (forward only). von Mises 1000 Pa, E0 20 kPa, batch 1, 20 updates, f64, atomic P2G, reference policy frozen.
- Run: `experiments/differentiable_mpm/runs/episode14_von_mises_fit`, 20/20 updates, no errors, 9.4 MB.

## Chunk characterization (recorded clouds, scene frame)

| chunk | frames | travel | height p95 first -> last | footprint cm2 | role |
|---|---|---|---|---|---|
| 02 | 65 | 71 mm | 31 -> 42 | 96 -> 70 | adjoint NaN @7040 |
| 03 | 65 | 60 | 41 -> 43 | 70 -> 69 | adjoint NaN @5312 |
| 04 | 57 | 143 | 43 -> 49 | 68 -> 58 | depth fail; adjoint NaN @3136 |
| 05 | 48 | 106 | 47 -> 49 | 56 -> 60 | depth fail; adjoint NaN @4096 |
| 06 | 47 | 43 | 49 -> 53 | 62 -> 61 | validation (ok) |
| 07 | 39 | 14 | 53 -> 60 | 58 -> 43 | adjoint NaN @3840 |
| 08 | 32 | 26 | 60 -> 41 | 41 -> 68 | depth fail; adjoint NaN @320 |
| 09 | 47 | 64 | 39 -> 47 | 68 -> 54 | depth fail; adjoint NaN @3776 |
| 10 | 108 | 9 | 46 -> 22 | 54 -> 92 | depth fail; adjoint NaN @4032 |
| 11 | 99 | 16 | 25 -> 12 | 110 -> 130 | validation (ok) |
| 12 | 39 | 3 | 14 -> 22 | 134 -> 106 | training (ok) |

Chunks 08, 10, 11 are the clean compressions (dough pressed from 60/46/25 mm to 41/22/12 mm), i.e. the information for E
and the yield stress that Episode 11 lacked; 08 and 10 cannot be trained on until the adjoint is fixed.

## 1. Final parameters (update 20) and 2. loss

| parameter | start | update 20 | bound |
|---|---|---|---|
| youngs_modulus | 20 000 | 52 545 Pa (still rising ~5 %/update) | 2 000 - 60 000 |
| poisson_ratio | 0.47 | 0.490 | upper |
| viscosity | 1.0 | 0.0 | lower |
| floor_retention | 0.80 | 0.200 | **lower** (Episode 11 drove it to the upper bound) |
| tool_friction_coefficient | 0.90 | 1.000 | upper |
| tool_stickiness | 0.05 | 0.300 | upper |

Training objective (chunk 12): 0.328 (update 1) -> 0.238 (update 20), flat after update 6. Validation objective per chunk,
initial -> final (strict `evaluate`): 02 1.71 -> 1.78, 03 1.75 -> 2.20, 04 1.70 -> 1.19, 05 1.50 -> 2.04, 06 1.60 -> 1.60,
07 0.82 -> 1.53, 08 2.17 -> 1.39, 09 0.84 -> 1.59, 10 0.80 -> 0.81, 11 1.86 -> 1.44; mean 1.48 -> 1.56 (worse).

## 3. Validation chamfer (recorded tool motion, full forward simulation, mm: mean / last frame / max)

| chunk | update 0 | update 10 | update 20 |
|---|---|---|---|
| 11 (squish 25 -> 12 mm) | 35.3 / 63.5 / 65.7 | 32.4 / 41.7 / 45.2 | 32.6 / 43.5 / 46.7 |
| 06 | 23.9 / 36.9 / 52.9 | 28.9 / 41.4 / 70.6 | 28.9 / 41.4 / 71.1 |

At the initial parameters the simulated chunk 11 dough is shoved ~100 mm sideways instead of being pressed flat; the fit
reduces the shove (floor retention down, stickiness up) but never produces the flattening, and chunk 06 gets worse.
Strict evaluator (top-view pixel MAE, post-initial mean): chunk 11 7.9 -> 9.6 mm, chunk 06 11.2 -> 9.9 mm, chunk 12 5.7 -> 4.6 mm;
the strict status is "invalid, 1 of 2 results" on most chunks because with two scored frames one failing the
`min_common_pixels` overlap condition is enough.

## 4. Completion / errors

20/20 updates, no errors in the fit. Earlier attempts: f32 rejected by the checkpoint-recompute check (chunk 09); the
adjoint failures above abort a fit at the first invalid episode, so they were classified beforehand
(`download/classification_episode14_vm.json` all-frames, `download/classification_endframes.json` end-frame setup).

## 5. Output and download

Run 9.4 MB; download folder on the GPU host `/workspace/runs/episode14/download/` (2.3G): the fit run, the strict
evaluations at the initial and final parameters, per-10-update forward snapshots of chunks 11 and 06 (GIFs + chamfer),
both classifications, the dataset used. Live page used during the run: https://claude.ai/artifact/NfXQgLFfQWWdngQnHkMv3P

## What would make Episode 14 usable

1. Fix the segment-replay adjoint (failures at multiples of segment_length, plasticity-independent, also f64); that alone
   returns chunks 02-05, 07-10 to the fit, including the three clean compressions.
2. Keep the first+last-frame scoring with the frame-referenced depth term (occluded intermediate frames), or score the
   first unoccluded frame after each motion from the temporal annotations.
3. Use f64 + atomic P2G (f32 fails the consistency check on long chunks).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
