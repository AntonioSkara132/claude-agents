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

## What is running instead (labelled diagnostics, not the requested fit)

`episode11_vm1000_e20k_fit_depthoff7`: the requested settings (von Mises 1000 Pa, E0 20 kPa, batch size 4,
100 updates, lr 0.01, f64) with `loss.depth_weight: 0`, on the 7 training chunks with finite adjoints
(02, 04, 05, 08, 21, 23, 26) + the 3 validation chunks. The closest fit to the request that this data allows.
(An earlier 13-chunk depth-off attempt died on the adjoint chunks.)

Both write `optimizer_state.json` per update under `experiments/differentiable_mpm/runs/`. Results (final
parameters, start/end training and validation loss, per-chunk validation rec-to-sim, completion, output size)
will be appended here when they finish. Nothing from `runs/` goes through the repository; the run
directories stay on the host at the paths above.
