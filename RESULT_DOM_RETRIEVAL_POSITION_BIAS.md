# Result: the mid-episode outlier is a position bias, not a shape/timing error

Follow-up to `RESULT_DOM_RETRIEVAL_OPEN_LOOP_VISUALIZATION.md`, which flagged
`episode61_kugla/segment-...-pc000166-pc000213` (frame 166 start, i.e. a mid-episode segment)
as a 3x outlier (78.3mm avg error vs. 21-30mm for the other 4 sampled queries). Quantified what
"looks too far" in the rendered preview actually means, from the saved `prediction.npy`/
`target.npy` for that query:

| Tool | Predicted vs. recorded centroid offset | Trajectory's own motion range |
|---|---:|---:|
| Tool 0 | **85mm** (mostly X: −79mm) | ~50mm (X), ~44mm (Y), ~26mm (Z) |
| Tool 1 | **58mm** | ~34-75mm depending on axis |

## Diagnosis: position bias, not shape/timing error

Tool 0's centroid offset (85mm) is *larger than the entire span of its own recorded motion*
(50mm). This means the model did not merely trace a slightly-wrong path shape or get the timing
off — it placed the **entire predicted trajectory in the wrong region of 3D space** relative to
where the tool actually was. Tool 1's offset (58mm) is comparable to its own motion range,
showing the same systematic bias, smaller in magnitude.

This is consistent with, and sharpens, the mid-episode-segment theory from the prior report: a
segment starting at frame 166 (not a fresh episode start) presents the encoder with a point cloud
that looks different from a genuine episode-start cloud (object already mid-deformation). The
model appears to be **anchoring its predicted tool position to where tools "typically" sit
relative to a fresh-looking cloud**, rather than reading this specific mid-motion cloud's actual
tool-relative geometry. That produces a large constant-ish position offset rather than a noisy
or oddly-shaped path — i.e. a bias, not variance.

## Why this matters beyond one outlier

A position bias (vs. random noise) means:
- Averaging over more retrieved demonstrations (the K-sweep in
  `RESULT_DOM_RETRIEVAL_K_AND_RESOLUTION_SWEEP.md`) won't fix it — it would need the retrieval
  keys or observation encoding to actually distinguish "fresh episode start" from "mid-episode
  continuation" clouds, which nothing in the current pipeline explicitly does.
- It's a plausible root cause for part of the aggregate 30.52mm test error being higher than the
  4 non-outlier queries' 21-30mm band: a systematic bias on a subset of segments (mid-episode
  continuations) drags the mean up more than proportional variance would.

Worth checking, as a next step, whether segments starting after frame 0 are disproportionately
represented among the worst-performing test queries generally (not just this one sampled
example), which would confirm this as a systematic pattern rather than a single bad segment.

Full data for this query at `/workspace/runs/open_loop_best_model/query_0118/` on this host only.
