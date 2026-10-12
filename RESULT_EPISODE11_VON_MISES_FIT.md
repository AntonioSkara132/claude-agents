# Result: Episode 11 von Mises fit (interim, 2026-10-11)

Request: `REQUEST_EPISODE11_VON_MISES_FIT.md`. Host: CUDA GPU (24 GB), Taichi 1.7.4, Python 3.11, f64.
TaichiDough at `c0fcb0d` (pulled; includes the batch-shuffle options, `sdf_cache.py`, `partial_view_icp.py`,
`continue_minibatch_fit.py`, `make_von_mises_dataset.py`).

## Setup (done, all hashes verified)

- Data in place under `experiments/differentiable_mpm/data/`: the Episode 11 temporal set, the table-aligned
  calibration, and `episode18_dynamics_icp_final_v1/` (the four tool files). Every `expected_sha256` in the
  Episode 11 configs verifies.
- Von Mises dataset created with `scripts/make_von_mises_dataset.py`:
  `dataset_episode11_dynamics_temporal_von_mises_1000pa_v1.json` + 19 configs (plasticity von-mises,
  yield 1000 Pa, E0 20 000 Pa, mass 0.3 kg from the configs, `plastic_min/max` removed from fit and bounds;
  fit = youngs_modulus, poisson_ratio, viscosity, floor_retention, tool_friction_coefficient, tool_stickiness).
- Preflight validation of all 19 episodes passes.
- Two environment notes, no code change: worker subprocesses must run through the host's glibc loader shim
  (`sys.executable` redirected; plain py311 cannot import Taichi here), and `--output-dir` must be under
  `experiments/differentiable_mpm/runs/` (the request's `runs/...` is rejected by `RunStore`).

## The requested fit cannot run: 14 of 16 training chunks invalidate the candidate at the initial parameters

Classification at E0 = 20 kPa with gradients (`/workspace/runs/episode11/classification.json`):

| chunks | status |
|---|---|
| 04, 05 | ok (loss 0.32, 0.33) |
| 02, 08, 09, 10, 14, 16, 17, 18, 21, 23, 26 | `Fixed initial-referenced depth support 0 is below 1` (at frame 4 or 11 of the window) |
| 06, 25, 27 | forward completes (final loss 0.39-0.79) but the **backward produces non-finite adjoints** in every state field (`Invalid trajectory adjoint at global step ...`) |
| 07, 15, 22 (validation) | ok (forward) |

The fit aborts on the first invalid episode (nothing to backtrack to), which is what happened in batch 1.

### Cause of the depth-support failures: the dough leaves its initial image footprint

The loss's depth term compares depth at pixels valid both now and in the initial frame (and in the initial
render). For Episode 11 that overlap decays to zero within the window, because the dough is moved across the
image, unlike the Episode 18 kugla recordings where it stays in place. Observed-valid pixels and overlap with
the initial mask, per scored frame (`prepare_experiment`, training camera raster):

- chunk18 (initial 795 px): overlap 738, 656, 547, 411, 255, 98, **0 from frame 7** (valid pixels stay ~800).
- chunk10 (initial 624 px): 542, 438, 352, 258, 163, 66, 2, **0 from frame 8**.
- chunk04 (initial 975 px, one of the two that pass): 917 ... 24, 19, 12, 23: survives with ~20 px.

So this is not an empty or bad observation frame (every frame has 3-4k finite points and ~500-900 valid
pixels); the initial-referenced depth term is not evaluable for recordings with this much dough translation.
Options on your side: `loss.depth_weight: 0` (coverage + distance terms only; the check is then skipped),
shorter scored windows, or a depth term referenced to the previous frame.

### The adjoint failures are not von Mises-specific, and there are more of them than the depth term showed

With the depth term off (so every chunk reaches the backward pass), the same classification was run under
von Mises AND under the original stretch-clamp plasticity, one episode at a time:

| chunk | von Mises, depth off | stretch-clamp, depth off |
|---|---|---|
| 02, 04, 05, 08, 21, 23, 26 | finite gradient (loss 0.28-1.09) | finite gradient (loss 0.37-1.15) |
| 09, 10, 14, 16, 17, 18 | non-finite adjoint | non-finite adjoint |
| 06, 25, 27 (with the depth term) | non-finite adjoint | (not re-run) |

Identical failure sets under both plasticity models, so the backward-pass blow-up is a property of these
recordings' dynamics (large motion / contact) in the differentiable solver, not of the von Mises model.
In total 9 of the 16 training chunks have unstable adjoints; 7 are usable (02, 04, 05, 08, 21, 23, 26).
Correction to an earlier interim note here: a pass that I read as "09 and 16 finite when run alone" was
reading the preflight validations, not the gradient results.

### The adjoint failures (06, 25, 27), first seen with the depth term on

Forward passes finish with the highest final losses of the set (0.39-0.79); the backward replay then yields
non-finite adjoints for x, v, C, F, Jp from a late step backwards. Same three fail with and without the depth
term. Looks like the von Mises return-mapping gradient under large plastic flow; not investigated further.

## Diagnostic fit 1: requested settings on the two chunks that pass as-is (04, 05) -- stopped at update 19

Loss 0.326 -> 0.257 over 19 updates (~3 min each). Parameters after 19: E 23 412 Pa, nu 0.490 (bound),
viscosity 2.7, floor_retention 0.90 (bound), tool_friction 1.00 (bound), tool_stickiness 0.224. Three of six
parameters sit at their bounds: two chunks do not constrain them. Stopped to free the GPU; run directory
`experiments/differentiable_mpm/runs/episode11_vm1000_e20k_fit_validsubset` kept.

## Diagnostic fit 2 (the main result): requested settings, depth term off, 7 finite-adjoint chunks -- FINISHED

`experiments/differentiable_mpm/runs/episode11_vm1000_e20k_fit_depthoff7`: von Mises 1000 Pa, E0 20 kPa, f64, CUDA,
batch size 4, lr 0.01, **100/100 updates, no errors**, backward replay consistent on all 100 objectives, 348 MB.
Training chunks 02, 04, 05, 08, 21, 23, 26; validation 07, 15, 22; `loss.depth_weight: 0` (coverage + distance terms).

1. Final `parameters_after` (update 100):

| parameter | start | final | bound |
|---|---|---|---|
| youngs_modulus | 20 000 | **14 979 Pa** | 2 000 - 60 000 |
| poisson_ratio | 0.47 | **0.490** | upper bound 0.49 |
| viscosity | 1.0 | **59.5 Pa s** | upper bound 60 |
| floor_retention | 0.80 | **0.900** | upper bound 0.9 |
| tool_friction_coefficient | 0.90 | **1.000** | upper bound 1.0 |
| tool_stickiness | 0.05 | **0.300** | upper bound 0.3 |

Five of six parameters end on their upper bounds (nu, viscosity, floor retention, friction, stickiness); only
E is interior. The bounds, not the data, determine this result.

2. Loss. Minibatch objective 0.390 (update 1) -> 0.299 (update 100); per batch composition the hard batch
went 0.68 -> ~0.39-0.46 and the easy batch 0.39 -> ~0.29-0.33, flat after update ~40. Per-chunk differentiable
objective at the final parameters: training 02 1.07, 04 0.19, 05 0.40, 08 0.31, 21 0.22, 23 0.51, 26 0.27
(start, from the update-0 classification: 1.09, 0.28, 0.47, 0.42, 0.31, 0.70, 0.36); validation 07 0.57,
15 0.48, 22 0.37.

2b. Differentiable objective per chunk, update 0 -> final (strict `evaluate` at both parameter sets):
training 02 1.092 -> 1.068, 04 0.283 -> 0.186, 05 0.468 -> 0.403, 08 0.422 -> 0.306, 21 0.307 -> 0.222,
23 0.695 -> 0.509, 26 0.361 -> 0.272 (mean 0.518 -> 0.424); **validation 07 0.754 -> 0.572, 15 0.582 -> 0.481,
22 0.473 -> 0.367 (mean 0.603 -> 0.473)**. The validation chunks improve by the same ~20% as training.

3. Validation rec-to-sim. The strict evaluator (`dynamic-topview-proxy-replay/v1`, recorded tool motion,
top-view comparison of the visible surface) reports pixel metrics rather than a single metre distance; both
parameter sets evaluated with the same code (`..._evaluate_initial`, `..._evaluate`). Post-initial means:

| validation chunk | pixel MAE, update 0 -> final | pixel IoU | depth-change MAE | strict status |
|---|---|---|---|---|
| 07 | 6.43 -> 5.97 mm | 0.283 -> 0.334 | 5.69 -> 5.95 mm | completed, 13/13 frames |
| 15 | 5.41 -> 5.72 mm | 0.308 -> 0.339 | 4.34 -> 4.53 mm | 7 of 14 frames flagged invalid |
| 22 | 3.19 -> 2.99 mm | 0.373 -> 0.412 | 2.92 -> 2.78 mm | 7 of 17 frames flagged invalid |

(Training chunks: pixel MAE 3.4-5.3 mm, IoU 0.30-0.45 at the final parameters, 1-7 flagged frames each. The
flagged frames are the `min_common_pixels: 50` footprint-overlap condition, the same transport effect as the
depth term; the dataset-level strict status is therefore "evaluation_failed" at both parameter sets.) The
fit's own in-run strict evaluation had failed for an environment reason (the evaluator is spawned with the
plain interpreter inside the worker, where Taichi cannot import); the two `evaluate` runs above use a
`sitecustomize` hook that redirects `sys.executable` to the glibc shim.

My own measure (full forward simulation with the recorded tool motion, chamfer between simulated particles
and the recorded cloud, both in the scene frame):

| validation chunk | update 0: mean / last / max | final: mean / last / max |
|---|---|---|
| 07 | 14.7 / 21.8 / 24.0 mm | 11.9 / 15.2 / 18.3 mm |
| 15 | 12.8 / 21.8 / 22.1 | 11.8 / 18.8 / 19.7 |
| 22 | 11.1 / 19.9 / 21.4 | 9.5 / 15.5 / 16.2 |

4. Completed all 100 updates; no errors in the fit log. 5. Output 348 MB.

### What the simulations show (GIFs in `download/gifs/`)

In Episode 11 the dough is transported across the table during a chunk. The simulated dough follows but lags:
the chamfer starts at the reconstruction error (5-8 mm) and grows to ~20 mm at the initial parameters, ~15-19 mm
at the final ones. The contact parameters at their upper bounds buy about a third of the gap and no more; E and
viscosity hardly matter for transport (the update-62 and update-100 simulations of chunk07, E 16.7 vs 15.0 kPa and
viscosity 31 vs 59, differ by 0.5 mm mean particle displacement and give identical chamfer). The simulated tool
also smears a corner of the dough onto the floor (two detached blobs, ~190 particles, 1.9 mm high), at both
parameter sets. Conclusion for the requester: with this contact model the simulator cannot drag the dough as
far as the recordings do; widening the bounds or changing the tool-contact model is the lever, not more updates.

Files: `chunk{07,15,22}_{initial,final}.gif` (recorded | simulated | overlay, top-down),
`chunk07_perspective_preview.gif` (renderer view), `chunk07_perspective_initial_vs_final_lastframe.png`.

## Re-fit with widened contact bounds (2026-10-12) -- the main result now

Why: in the first fit five of six parameters sat on their upper bounds, and a direct test on validation chunk 07
showed floor retention was the binding one (chamfer last-frame 15.2 mm at the bound 0.9, 11.8 mm at 0.99, 11.1 mm
at 0.999; stickiness beyond its bound changed nothing). In the solver the tangential velocity is multiplied by
`floor_retention` on every floor-contact substep (`solver.py:547`, 0.2 ms substeps), so 0.9 is a 10 % loss per
substep: the dough is glued to the table unless a tool is pushing it. Every Episode 11 chunk transports the dough
57-219 mm (footprint overlap first/last frame 0.00-0.16 in 17 of 18 chunks; only chunk 25 kneads in place), so
this bound limits the whole episode.

Setup: identical to the 7-chunk depth-off fit (von Mises 1000 Pa, E0 20 kPa, batch 4, f64, lr 0.01, 100 updates,
training 02 04 05 08 21 23 26, validation 07 15 22) except `floor_retention` bounds 0.2-0.999 (was 0.9) and
`tool_stickiness` 0-1 (was 0.3). Datasets `*_depthoff_7chunks_widebounds_v1.json` and `*_widebounds_cont69_v1.json`.

Execution note: the run was deterministic and identical to the first fit until floor retention reached 0.9 at
update 10. At update 69 (10:43) it aborted with "Experiment sources changed during episode execution": another
session edited `loss.py` and `dataset_config.py` in the shared tree (an opt-in `depth_reference: frame` option;
the default path is byte-identical and our dataset has the depth term off, so the objective is unchanged). The
remaining 31 updates were run as a second run starting from the update-69 parameters with a fresh Adam state
(`runs/episode11_vm1000_e20k_fit_depthoff7_widebounds` 69 updates + `..._widebounds_cont69` 31 updates).
The fresh optimizer state moved the parameters noticeably in those 31 updates (see below); a true continuation
needs `scripts/continue_minibatch_fit.py`, which requires a cleanly finished original run.

1. Final `parameters_after` (update 100 = continuation update 31):

| parameter | first fit (old bounds) | update 69 | **final** | bounds |
|---|---|---|---|---|
| youngs_modulus | 14 979 | 25 798 | **30 881 Pa** | 2 000 - 60 000 |
| poisson_ratio | 0.490 (bound) | 0.490 | **0.490** (bound) | 0.45 - 0.49 |
| viscosity | 59.5 (bound) | 24.8 | **53.9 Pa s** | 0 - 60 |
| floor_retention | 0.900 (bound) | 0.9914 | **0.9699** | 0.2 - 0.999 |
| tool_friction_coefficient | 1.000 (bound) | 0.907 | **0.871** | 0.5 - 1.0 |
| tool_stickiness | 0.300 (bound) | 0.437 | **0.694** | 0 - 1 |

Only nu remains on a bound. Floor retention settles at 0.97-0.99 (interior), friction comes off its bound.

2. Loss. Minibatch objective: the two alternating batches went 0.68 / 0.39 (update 1-2) -> 0.54 / 0.10 (update
68-69) -> ~0.45 / 0.06 at the end; the hard batch is dominated by chunk 02, whose recording has a 207 mm
centroid jump between two consecutive frames (reconstruction glitch; its objective stays at 1.07 in every fit).
Per-chunk differentiable objective at the final parameters (strict `evaluate`): training 02 1.068, 04 0.065,
05 0.308, 08 0.133, 21 0.080, 23 0.302, 26 0.133 (mean 0.298; first fit 0.424; update 0 0.518);
**validation 07 0.382, 15 0.217, 22 0.197 (mean 0.265; first fit 0.473; update 0 0.603)**. The widened bounds
cut the validation objective by a further 44 %, the same relative gain as on training.

3. Validation rec-to-sim at the final parameters:

| chunk | chamfer mean / last / max (mm), update 0 -> first fit -> **wide bounds** | pixel MAE (strict) first fit -> wide | pixel IoU |
|---|---|---|---|
| 07 | 14.7/21.8/24.0 -> 11.9/15.2/18.3 -> **9.5/12.4/12.7** | 5.97 -> 4.73 mm | 0.334 -> 0.416 |
| 15 | 12.8/21.8/22.1 -> 11.8/18.8/19.7 -> **8.9/11.5/12.3** | 5.72 -> 5.36 | 0.339 -> 0.451 |
| 22 | 11.1/19.9/21.4 -> 9.5/15.5/16.2 -> **7.4/9.1/9.9** | 2.99 -> 2.90 | 0.412 -> 0.510 |

Reconstruction error alone is 5-8 mm, so the last-frame gap is now 2-5 mm above the floor set by the recording.
The strict evaluator still reports "evaluation_failed" at the dataset level for the same `min_common_pixels`
footprint-overlap reason as before (1-8 flagged frames per chunk); chunk 07 completes with all frames valid.

Progress along the fit (chunk 07 chamfer mean/last/max at the parameters of update N, recorded tool motion):
u0 14.7/21.8/24.0, u10 12.6/15.9/19.9, u20 8.9/11.6/11.6, u30 9.7/12.1/13.3, u40 8.8/11.7/11.7, u50 8.9/11.8/11.8,
u60 8.7/11.6/11.6, u79 9.8/12.5/13.5, u89 9.8/12.6/13.6, u99 9.5/12.4/12.8. The chamfer was essentially converged
by update 20 (floor retention ~0.99); the later drift of E, viscosity and stickiness changes it by <1 mm, i.e.
those parameters are weakly identified by this objective, floor retention is the one that matters.

4. Completed all 100 updates (69 + 31, one abort for the external reason above, no numerical errors; backward
replay consistent in both runs). 5. Output: 69 MB + 63 MB for the two runs, plus 480 MB for the strict evaluation.

Recommendation for the TaichiDough side: keep the widened floor-retention bound (or better, fit
`log(1 - floor_retention)`: the per-substep form makes the parameter extremely non-linear near 1), drop or mask
chunk 02's glitch frame, and note the other session's finding that the non-finite adjoints on chunks 06/09/10/
14/16/17/18/25/27 are checkpoint-segment-boundary replay failures (they move with `segment_length`), not physics.

GIFs: `download/gifs/chunk{07,15,22}_wbfinal.gif` (final wide-bounds parameters),
`download/gifs/widebounds_progress/` (chunk 07 every 10 updates, both runs).

## Download

`/workspace/runs/episode11/download/` on the GPU host (1.3 GB; wide-bounds runs `episode11_vm1000_e20k_fit_depthoff7_widebounds{,_cont69}`, their strict evaluation `..._cont69_evaluate`, `selected_parameters_u69.json`): `episode11_vm1000_e20k_fit_depthoff7/` (the fit
run, with `optimizer_state.json`, `selected_parameters.json`, `result.json`, per-episode workers),
`episode11_vm1000_e20k_fit_depthoff7_evaluate{,_initial}/` (strict evaluations at the final and the initial
parameters; `selected_parameters_initial.json`), `episode11_vm1000_e20k_fit_validsubset/` (diagnostic fit 1),
`classification*.json`, `gifs/`.


## Update 2026-10-12 afternoon: what limits the fit (bounds, transport, shape)

### Every chunk transports the dough
Recorded centroid travel over a chunk (table plane), footprint overlap first/last frame on 5 mm cells: training chunks
02 174 mm (0.00), 04 74 (0.02), 05 83 (0.01), 08 121 (0), 21 104 (0), 23 57 (0.16), 26 123 (0); validation 07 90 (0.05),
15 174 (0), 22 127 (0); the adjoint-NaN chunks 06/09/10/14/16/17/18/27 71-188 mm (0); only chunk 25 kneads in place
(22 mm, 0.43). Chunk 02 has a 207 mm centroid jump between two consecutive frames (reconstruction glitch); it is the
chunk with the worst objective (1.07) and should be dropped or masked in future fits.

### Floor retention bound was the binding constraint
`solver.py` multiplies the tangential velocity by `floor_retention * (1 - floor_stickiness)` on every 0.2 ms floor-contact
substep, so 0.9 is a 10 %/substep loss: the dough is glued to the table unless a tool pushes it. Chunk 07, fitted E/nu/
viscosity, recorded tool motion, chamfer sim-vs-recorded (mean / last frame, mm): floor 0.80 14.7/21.8; 0.90 (bound)
11.9/15.2; 0.99 9.1/11.8; 0.999 8.2/11.1; stickiness 0.6 at floor 0.9 11.4/15.2 (no effect). Reconstruction error alone
is 5-8 mm. Recommendation for the solver: fit log(1 - floor_retention), the per-substep form is extremely nonlinear near 1.

### Re-fit with widened bounds (floor_retention 0.2-0.999, tool_stickiness 0-1)
`runs/episode11_vm1000_e20k_fit_depthoff7_widebounds` (same 7 chunks, depth term off). Identical to the first fit until
floor retention crossed 0.9 at update 10, then: update 20 floor 0.994, chunk 07 chamfer 8.9/11.6 mm; flat afterwards
(update 60: 8.7/11.6; parameters at update 68: E 25.8 kPa, viscosity 24.8, floor 0.991, friction 0.907, stickiness 0.437).
The run aborted at update 69 on TaichiDough's source-fingerprint check because the files below were edited in the shared
tree while it ran (my mistake); the other session continues it as `..._widebounds_cont69` (31 updates, fresh Adam).
Validation objective at update 68 vs the first fit's final: not yet evaluated strictly; chunk 07 chamfer 15.2 -> 11.6 mm.

### Shape: the elongation is tool contact, not material
With transport fixed, the simulated chunk 07 dough is stretched along the push direction (last frame extent 128 x 221 mm,
footprint 96 cm2, height 41 mm) while the recorded dough is compact (159 x 125 mm, 106 cm2, 33 mm). Forward sweep at the
fitted parameters (reference: E 25 kPa, yield 3000 Pa, viscosity 20, stickiness 0.434), mean particle displacement of the
last frame relative to the reference and the extent along the push:

| change | mean displacement | extent along push |
|---|---|---|
| yield stress 3000 -> 5000 Pa | 0.00 mm | 221 mm |
| yield stress 3000 -> 200 Pa | 13.1 mm | 209 |
| E 25 -> 60 kPa | 1.8 mm | 224 |
| E 25 -> 5 kPa (yield 5000) | 6.4 mm | 211 |
| viscosity 20 -> 60 | 0.3 mm | 221 |
| **tool stickiness 0.434 -> 0** | **33 mm** | **199** |

Plasticity is active (yield 200 Pa moves the dough 13 mm) but the fitted range 1-5 kPa is nearly irrelevant for this
loading, E and viscosity move particles by millimetres, and stickiness is the only parameter that changes the shape
appreciably; even at 0 the dough stays 60 % too long. The tool drags the dough along its path. Fixing the shape needs the
tool-contact model (or its stickiness/adhesion parameterisation), not material bounds.

### Yield-stress fit with a frame-referenced depth term (running)
Code (shared tree, both uncommitted): `loss.py` option `depth_reference: "frame"` (current rendered depth vs current
observed depth on all observed pixels; the default `"initial"` is unchanged; gradient checked by finite differences),
`dataset_config.py` schema v4 = v2 physical names + `von_mises_yield_stress_pa`. Dataset
`dataset_episode11_dynamics_temporal_von_mises_yieldfit_framedepth_7chunks_v1.json`: contact fixed at the re-fit's update-68
values, fitted E (20 kPa), nu, viscosity, yield (1000 Pa, bounds 200-5000), depth weight 1.0 frame-referenced, coverage
0.5, distance 0.5. Preflight passes on all 10 chunks (the frame-referenced term does not need footprint overlap).
`runs/episode11_vm_e20k_fit_yieldfit_framedepth7`, update 29/100: E 25.5 kPa, yield 1157 Pa, viscosity 23, nu 0.49 (bound);
all gradients push the material stiffer, but repeated batch compositions return the same loss to 0.003 and chunk 07
chamfer is 8.7/11.7 mm at updates 0, 10, 20: the objective is flat along this direction, as the sweep predicts.

GIFs: `download/gifs/chunk07_bounds_*.gif`, `chunk07_shape_*.gif`; per-10-update snapshots of both fits in
`download/progress/`.

## Earlier interim notes (superseded where the above differs)

`episode11_vm1000_e20k_fit_depthoff7`: the requested settings (von Mises 1000 Pa, E0 20 kPa, batch size 4,
100 updates, lr 0.01, f64) with `loss.depth_weight: 0`, on the 7 training chunks with finite adjoints
(02, 04, 05, 08, 21, 23, 26) + the 3 validation chunks. The closest fit to the request that this data allows.
(An earlier 13-chunk depth-off attempt died on the adjoint chunks.)

Both write `optimizer_state.json` per update under `experiments/differentiable_mpm/runs/`. Results (final
parameters, start/end training and validation loss, per-chunk validation rec-to-sim, completion, output size)
will be appended here when they finish. Nothing from `runs/` goes through the repository; the run
directories stay on the host at the paths above.
