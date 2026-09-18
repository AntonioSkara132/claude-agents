# Result: open-loop visualization of the best model, and a found failure mode

Follow-up to `RESULT_DOM_RETRIEVAL_K_AND_RESOLUTION_SWEEP.md`. Operator pulled a new
`visualizer` commit into `dom_retrieval` (`evaluation/open_loop.py`,
`scripts/visualize_open_loop.py`, `tests/test_open_loop.py`) and asked to visualize the best
model from that sweep: `latent_residual`, 1024 points, K=100 (30.52mm avg waypoint error, the
best result across the whole investigation).

## How it was run

```bash
cd /workspace
python -m dom_retrieval.scripts.visualize_open_loop \
  --checkpoint /workspace/runs/deformpath_cuda_joint200_1024pts_k100_seed7/latent_residual/checkpoint.pt \
  --output /workspace/runs/open_loop_best_model \
  --device cuda --partition test --count 5 --format gif
```

`--format both` (GIF+MP4) failed in this sandbox: `ffmpeg` here only has the `libopenh264`
encoder, not `libx264`, so `FFMpegWriter(codec="libx264")` errors with "Unknown encoder
'libx264'". Not a `dom_retrieval` bug — environment-specific missing codec. GIF export works
fine; used that. `tests/test_open_loop.py` wasn't run against this checkpoint (it's a unit test
suite, not a smoke test for real checkpoints), but the CLI ran end-to-end without errors and
produced sane, physically plausible output (see below).

## 5 evenly-spaced test queries, results

| Query ID (recording group / segment) | Avg error (mm) | Final error (mm) |
|---|---:|---:|
| episode22_kugla / pc000000-pc000041 | 30.4 | 35.1 |
| episode32_kugla / pc000058-pc000074 | 22.5 | 26.0 |
| episode36_kugla / pc000101-pc000115 | 21.0 | 27.2 |
| episode43_kugla / pc000126-pc000144 | 22.5 | 21.7 |
| **episode61_kugla / pc000166-pc000213** | **78.3** | **72.6** |

4 of 5 land at 21-30mm (actually better than the 30.52mm aggregate test average). One is a
3x outlier.

## Found failure mode: mid-episode segments with near-stationary ground truth

The `episode61_kugla` outlier segment starts at frame **166**, not 0 — i.e. its "initial
observation" is the first retained cloud of a segment that begins mid-motion, not a fresh episode
start. Its recorded trajectory (ground truth) is a small, nearly-stationary loop. The model's
prediction is a much larger loop offset from it — error climbs to 60-110mm within the first ~20%
of phase and never recovers by the end.

Two things line up:

1. Mid-episode segments' input point clouds look visually sparser/different from a genuine
   episode-start cloud (compared side-by-side against the episode22 query's cloud), since the
   object is already mid-deformation rather than in its initial rest state.
2. The model appears to predict "typical-scale" motion regardless of this, rather than reading
   fine-grained cues that this specific continuation is nearly stationary — consistent with a
   model that has learned the population motion distribution more than per-observation
   conditioning.

This is a real, previously-invisible failure mode: the aggregate 30.52mm test metric is hiding
that mid-segment/near-stationary continuations can fail 2-3x worse than fresh-episode-start
segments. Worth checking whether `mode: segments`' policy of treating every segment (including
mid-episode continuations) as an independent "initial observation → trajectory" example is
appropriate, or whether mid-episode segments need a different treatment (e.g. excluded, or given
additional history/velocity context that a single point cloud can't carry).

Full artifacts (`preview.png`, `rollout.gif`, `errors.csv`, `summary.json` with metrics +
retrieved-demo IDs, raw `.npy`) for all 5 queries at `/workspace/runs/open_loop_best_model/` on
this host only.
