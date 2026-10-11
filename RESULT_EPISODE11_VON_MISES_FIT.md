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

### The adjoint failures (06, 25, 27)

Forward passes finish with the highest final losses of the set (0.39-0.79); the backward replay then yields
non-finite adjoints for x, v, C, F, Jp from a late step backwards. Same three fail with and without the depth
term. Looks like the von Mises return-mapping gradient under large plastic flow; not investigated further.

## What is running instead (both labelled diagnostics, not the requested fit)

1. `episode11_vm1000_e20k_fit_validsubset`: the requested settings on the two valid training chunks (04, 05)
   + the 3 validation chunks (batch size 2). Shows the von Mises fit runs end to end here and gives timing.
2. `episode11_vm1000_e20k_fit_depthoff`: the requested settings with `loss.depth_weight: 0` on the 13
   training chunks that are not adjoint failures + validation (batch size 4). The closest thing to the
   requested 16-chunk fit that can run on this data.

Both write `optimizer_state.json` per update under `experiments/differentiable_mpm/runs/`. Results (final
parameters, start/end training and validation loss, per-chunk validation rec-to-sim, completion, output size)
will be appended here when they finish. Nothing from `runs/` goes through the repository; the run
directories stay on the host at the paths above.
