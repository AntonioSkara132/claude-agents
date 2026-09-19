# Learned reranker on top of cosine retrieval: no measurable win

Implemented the small learned reranker (`models/retrieval.py::LearnedReranker`): after
cosine top-K selection, a shared 2-layer MLP scores each retrieved candidate from
`[query, candidate, query-candidate, query*candidate]`; `score = cosine_similarity +
correction`; correction's final layer is zero-initialized so training starts identical
to plain cosine weighting. Wired into `CosineTopK`/`RetrievalPolicy` behind
`learned_reranker: true` (also added `freeze_residual` to `train_policy` and an
`init_checkpoint` warm-start hook in `run_experiment`, both new, minimal, config-gated).
83 existing tests still pass; 5 new tests cover the reranker and freeze/warm-start paths.

Ran 4 variants on `cartesian_residual` + `orientation_delta`, all against the
1000-epoch, no-reranker checkpoint from `REQUEST_DOM_RETRIEVAL_ORIENTATION_DELTA_IMPLEMENTED.md`
(25.25mm / 6.51°):

| variant | epochs | position | orientation | retrieval_entropy_normalized |
|---|---|---|---|---|
| baseline (no reranker) | 1000 | 25.25mm | 6.51° | 0.9960 |
| reranker, frozen encoder+residual, warm-started | 200 | 25.36mm | 6.49° | 0.9933 |
| reranker, unfrozen, warm-started | 200 | 25.25mm | 6.51° | 0.9960 (reverted to baseline) |
| reranker, unfrozen, cold-start | 200 | 30.87mm | 6.22° | 0.9416 |
| reranker, unfrozen, cold-start | 1000 | 33.27mm | 6.70° | 0.8158 |

Notes per variant:

- **Frozen, warm-started (the clean test):** only the reranker's own parameters were
  trainable (encoder and `ResidualNetwork` frozen via a `torch.state_dict` load from
  the finished checkpoint, `strict=False`, verified only `retriever.reranker.*` keys
  were missing). Validation MSE did move off the zero-init point (best at epoch 18,
  0.4062 vs 0.4107 at epoch 0), i.e. it isn't fully inert, but the final test-set effect
  is inside noise: +0.11mm position, -0.02° orientation.
- **Unfrozen, warm-started:** validation MSE at epoch 0 (the warm-started state) was
  never beaten across all 200 epochs (monotonically climbed to 0.56 by epoch 200), so
  best-checkpoint selection reverted fully to the pre-training warm-start weights —
  byte-identical metrics to the no-reranker baseline. Jointly fine-tuning an
  already-converged encoder+residual with fresh Adam state just destabilizes it here;
  this run doesn't say anything about the reranker either way.
- **Unfrozen, cold-start (200 and 1000 epochs):** worse than baseline at both budgets,
  and worse at 1000 epochs than at 200 (33.27mm vs 30.87mm). `retrieval_entropy_normalized`
  fell substantially (0.996 -> 0.82 at 1000 epochs) — the reranker does sharpen retrieval
  weighting a lot when trained jointly from scratch, but that sharper retrieval doesn't
  translate into better task metrics; if anything it correlates with worse ones here.

## Read

The reranker's own gradient signal is real (frozen-only run moved off zero-init) but
too small to matter at this data scale, and letting it interact with joint
encoder/residual training just adds instability rather than gains. I don't think this is
worth pursuing further without a different signal for why retrieval weighting should
matter more than it currently does — the flat retrieval entropy (~0.996) in every
well-performing variant suggests the `ResidualNetwork` correction head is absorbing
nearly all of the useful signal regardless of how the K=100 neighbors are weighted.

Code changes are local only (not pushed to `dom_retrieval` origin — following the same
pattern as your commits, since this session doesn't have GitHub auth for that repo). Say
the word if you want the diff packaged as a patch like `dom_retrieval-0ad5d5b-orientation-delta.patch`.

## Separate: MPM policy-control CUDA run in progress

Picked up `REQUEST_MPM_POLICY_CONTROL_CUDA_RUN.md` after you pushed the runner changes
(`4735e3b`). Two real bugs found and worked around on this host, both worth knowing about
if they generalize:

1. `load_condition_archive` requires the archive's `control_dt_s` to exactly match the
   simulator's physics `dt` (0.0002s here), not a coarser tool-control rate. I originally
   exported at 30Hz and got `Condition archive dt 0.0333333 does not match simulator dt
   0.0002`. Re-exported `export_policy_controls.py` with `--control-dt 0.0002` (same
   `_interpolate_positions`/`_interpolate_quaternions` upsample it fine, just a much
   bigger `.npz`).
2. `--end-frame 59` (the value used for the recorded-baseline run) needs archive
   coverage out to `recorded_times[59] = 1.9677s`, but the diagnostic-transformed
   prediction only covers one 44-step training segment (0.0667-1.5008s,
   duration 1.4340805858373642s = exactly `recorded_times[43]`). Used `--end-frame 43`
   instead so the predicted-condition run only simulates the span the prediction
   actually covers — will use the same endpoint for the recorded-baseline comparison run
   per your "identical... endpoint" instruction.
3. Also needed the same `LD_LIBRARY_PATH`-leak fix as before, but for `run.py` itself
   this time, not just its `--simulation-python` child: invoking `run.py` directly with
   `env LD_LIBRARY_PATH=<sysroot>` set broke the `render_python_xvfb.sh` dependency-check
   subprocess (`GLIBC_2.35 not found` in `bash` itself, since the sysroot libc leaked into
   an unrelated py311 subprocess). Fixed by invoking `run.py` through the existing
   `sim_python_wrapper.sh` + `sim_launcher.py` pair (already built for the simulation
   subprocess) instead of a raw `env`-prefixed command, since `sim_launcher.py` pops
   `LD_LIBRARY_PATH` before it starts running the target script in-process.

`predicted_xyz_recorded_orientation` condition simulation is running now
(`/workspace/TaichiDough/experiments/differentiable_mpm/runs/dom_policy_predicted_recorded_orient_diagnostic`).
Will run the matched `recorded_full_pose` baseline at `--end-frame 43` next and report
both, per your interpretation-limits note (diagnostic transform only, plumbing test).
