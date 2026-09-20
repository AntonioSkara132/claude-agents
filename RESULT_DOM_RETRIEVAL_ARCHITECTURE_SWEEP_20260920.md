# Architecture sweep on the new dataset, plus new-transform MPM re-runs

Everything below happened after `FOLLOWUP_EPISODE18_NEW_TRANSFORM_SIM_RERUN.md`. Long
session, grouping by topic. Standard for every number below unless noted: `cartesian_residual`,
orientation_delta, `k=100`, early stopping (`patience: 30`), 3 seeds (7/17/27) per
condition, on `Deformapth2_camera1024_interpolated_nodbscan`. Position error is reported
as **start / shape / raw** (start = t=0 only, shape = both trajectories aligned to their
own t=0 then compared, raw = plain unaligned) since raw alone conflates the two and they
don't decompose additively.

## Dataset transfer: now fully resolved

Re-audited after your fix: **50/50 episode directories load cleanly**, and **47/47**
overlapping with the old dataset have exactly matching frame counts. Earlier in the
session a re-audit mid-transfer showed a *different* 9/50 broken (not the same 10 as
before) — confirming it was actively syncing, not stably broken. It has since stabilized.
3 episodes (23, 68, 69) still get automatically excluded for missing temporal annotations
in `source_root` — expected, not a bug.

## New camera->mocap transform: validated, not diagnostic

The `T_M_DO_new` matrix you sent passes the point-cloud check the earlier diagnostic fit
failed: **median NN residual 1.1mm, p90 1.6mm** (vs 30.5mm/70.9mm for the old one), with
only 10/1024 points over 20mm — the same known sensor-artifact cluster from earlier
comparisons, not new outliers. Re-ran the MPM predicted-condition simulation with it for
both the old-dataset checkpoint and the new-dataset checkpoint: both completed cleanly
(7171/7171 steps, no failure). One remaining gap: I don't have this transform's own fit
provenance (data it was fit on, residual stats on its own target) since it arrived as a
matrix rather than a derivation — still provisional until that's confirmed on your end.

## Commits merged (all manually adapted around my own reranker, since I never took your
`SharedMLPReranker`/`40001b8` reranker — kept my independent `LearnedReranker` instead)

- `fba9bfb` constant residual mode (`residual_mode: constant`, one correction broadcast
  across the horizon).
- `86657ad` observation residual branch (`residual_input: observation` +
  `residual_mode: constant`): residual computed from the embedding alone, never sees the
  retrieved reference. Reference still feeds position directly; only the residual/correction
  is decoupled.
- `ae0a26e` kernel machines: `RBFTopK` (Gaussian kernel over normalized embeddings,
  `retrieval_mode: rbf`) plus `retrieval_all: true` (weight every eligible memory motion,
  no fixed K — literal Nadaraya-Watson).

All 94 local tests pass after merging. Not pushed anywhere (no GitHub auth for this repo
in this session, same as every prior local change) — say the word for a patch bundle.

## My own additions built during this thread

- `reranker_deep: true` — adds a hidden layer to `LearnedReranker` (2 Linear -> 3 Linear).
- `DeepSelectionScorer` (`deep_selection_scorer: true`) — a zero-init deep MLP
  (`Linear(4D,64)->GELU->Linear(64,64)->GELU->Linear(64,1)`) added to cosine similarity
  **before** masking/Top-K, so it can change *which* candidates get selected, not just
  reweight the ones already chosen. Verified with a deterministic test (forced a
  cosine-worst candidate to have the highest augmented score, confirmed it gets pulled
  into the Top-K).
- Extended `CosineTopK` to accept `k=None` (use every eligible candidate, matching how
  `RBFTopK`'s `retrieval_all` already worked) so `deep_selection_scorer` can run over the
  full memory bank too, not just a fixed K. Implemented, unit-tested, not yet run at scale.

## Results (new dataset, 3 seeds each unless noted)

| variant | start | shape | raw | orientation |
|---|---|---|---|---|
| baseline (per-waypoint, reference-conditioned) | - | - | 30.31mm (1 seed) | 6.02deg |
| constant, reference-conditioned | 18.99-23.23mm | 22.84-25.54mm | 22.23-26.30mm (23.83+/-2.19 mean) | 7.45-7.57deg (7.51+/-0.06) |
| constant, reference-conditioned + deep_projection | - | - | 27.93+/-3.66mm | 7.59+/-0.28deg |
| constant, observation-decoupled + no_motion_residual | - | - | 34.89+/-1.02mm | 7.36+/-0.15deg |
| constant, observation-decoupled + deep reranker (post-Top-K) | - | - | 22.60-33.58mm (28.46+/-5.53 mean) | 7.42-7.90deg |
| **constant, observation-decoupled + deep_selection_scorer (pre-Top-K)** | 15.18-22.23mm (18.82+/-3.55) | 19.62-22.52mm (21.48+/-1.60) | 20.98-25.58mm (**23.49+/-2.33**) | 7.26-7.61deg (7.45+/-0.18) |
| RBF kernel, Top-100, bandwidth=0.5 (seed 17 only so far) | 19.28mm | 22.32mm | 23.12mm | **5.72deg** |

Reading these:

- **Constant residual mode trades position for orientation** vs per-waypoint: better
  position, worse orientation, in every constant-mode variant tried. Makes sense
  architecturally — a single time-invariant rotation-vector correction can shift starting
  orientation but can't track orientation *changing* over the trajectory, which only
  per-waypoint residual can do.
- **Deep projection still doesn't help** (tested twice now: per-waypoint and constant
  residual, both times worse, one seed barely trained at all in the constant case).
- **Post-Top-K reranking vs pre-Top-K scoring, holding everything else fixed
  (observation-decoupled + constant)**: pre-Top-K (`deep_selection_scorer`) clearly wins
  (23.49+/-2.33mm vs 28.46+/-5.53mm) with much tighter variance. Letting the deep
  correction influence *which* candidates get retrieved helps more than reranking
  already-selected ones, at least once the residual can't compensate for a bad reference.
- **RBF kernel's first seed has the best orientation number of the entire investigation**
  (5.72deg) — only one seed run so far (chain still running as I write this), so treat as
  preliminary, but worth flagging immediately given how consistently ~6-8deg every other
  orientation number has landed.

## MPM: best observation-decoupled model, validated transform

Exported the best observation-decoupled checkpoint (`constant` + `deep_selection_scorer`,
seed 27, raw 20.98mm) through the same pipeline with `T_M_DO_new`. Completed cleanly
(7171/7171 steps, no failure), same episode18 segment/params/endpoint as every other run.

## Still running / not yet reported

- RBF chain: 1/3 seeds done (above), 2 more in flight.
- `deep_selection_scorer` + `retrieval_all` (all motions, no Top-K) combination: code
  merged and unit-tested, not yet run as an actual training job.
- No per-recording paired comparisons across these new conditions (same gap as the
  original ablation matrix) — say the word if you want them for any specific pair.
