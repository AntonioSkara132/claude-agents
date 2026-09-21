# Chained-anchor ("long-horizon") test: retrieval-space discontinuity found, reference_dropout mitigates it

Follow-up to `RESULT_REFERENCE_DROPOUT_TRAINING.md`. User's framing: in a real-time
closed-loop deployment, the "current pose" anchor fed into the next motion segment
is the robot's actual/predicted last endpoint from the previous segment, not a
ground-truth value pulled from a dataset. Offline evaluation always uses ground-truth
anchors, which could be hiding a real deployment risk. Tested this directly.

## Method

`/workspace/runs/reference_dropout/test_chained_anchor.py`. Two consecutive real
segments from the same episode/demo (`episode18_kugla`,
`segment-...-pc000137-pc000154` then `segment-...-pc000154-pc000172`), so no
coordinate transform is needed: predicted output before `output_to_scene` shares the
same per-demo local frame as `history_tool_positions` (observations are never
normalized/scene-transformed, only targets are — verified earlier this session).
Run the model on segment A, take its *predicted* last waypoint, overwrite segment B's
`history_tool_positions` last-valid entry with it (replacing the ground-truth
anchor), then compare segment B's prediction error normal-vs-chained.

## Result

| checkpoint | anchor drift | normal start/raw/orient | chained start/raw/orient | raw degradation |
|---|---|---|---|---|
| original loss (`rbf_bandwidth_sweep/cond_rbf_bw0p15_seed27`) | 6.1mm | 15.53 / 22.65mm / 6.09deg | 91.56 / 100.20mm / 10.00deg | **+342%** |
| decomposed start/shape loss (`start_shape_loss/cond_start_shape_seed27`) | 7.5mm | 13.87 / 21.73mm / 5.06deg | 100.06 / 96.64mm / 10.37deg | **+345%** |
| reference_dropout=0.5 (`reference_dropout/cond_refdrop_sometimes_seed27`) | 38.1mm | 41.77 / 39.89mm / 6.21deg | 61.61 / 58.26mm / 6.58deg | **+46%** |

Confirmed on two independently-trained checkpoints (original loss and decomposed
loss) that this is a general architectural property, not specific to either loss
formulation.

## Root cause (verified directly, not inferred)

Checked `output.indices` and `output.weights` from the retrieval step for the
original-loss checkpoint, normal vs. chained:

- Top-10 retrieved trajectory indices: **normal** `[21, 167, 55, 51, 158, 20, 163,
  142, 16, 402]`, **chained** `[30, 287, 415, 155, 277, 405, 169, 323, 394, 229]` —
  **zero overlap**.
- Top retrieval weight: **normal 0.05** (diffuse, spread across many neighbors) ->
  **chained 0.85** (sharply concentrated on a single match).

A small anchor perturbation (~6-7mm, *smaller than the model's own typical
~15-22mm prediction error*) flows through the GRU history encoder into the retrieval
embedding and is enough to land it in a completely different region of retrieval
space, pulling in an entirely different reference trajectory. This is a
**retrieval-space discontinuity**, not smooth interpolation drift — which is why a
tiny input perturbation produces a wildly disproportionate output error. It directly
explains why offline ground-truth-anchored evaluation can look fine while closed-loop
deployment (anchors = actual/predicted endpoints) would not.

## Mitigation

`reference_dropout=0.5` (from `RESULT_REFERENCE_DROPOUT_TRAINING.md`) reduces this
failure mode substantially too, not just the zero-reference case it was originally
tested against: +340-345% degradation (undropped checkpoints) down to +46%, even
under a *larger* (38mm vs 6-7mm) anchor perturbation in that run. Consistent
mechanism: training-time reference dropout forces the network to not over-rely on a
single sharp retrieval match, so a retrieval-embedding shift matters less.

## Follow-up: bandwidth sweep + finer dropout grid (2026-09-21)

Tested whether a wider RBF bandwidth (smoother kernel over retrieved neighbors)
fixes this more directly than dropout, plus filled in `reference_dropout` at 0.25
and 0.75 to check for a gradual tradeoff. Same episode18 segment pair, seed 27,
built on the decomposed start/shape-loss base.

| variant | normal raw/orient | chained raw/orient | anchor drift | raw degradation |
|---|---|---|---|---|
| bw=0.15, no dropout (baseline) | 21.7mm / 5.1deg | 96.6mm / 10.4deg | 7.5mm | +345% |
| bw=0.30, no dropout | 20.6mm / 5.1deg | 94.4mm / 8.7deg | 8.9mm | +359% |
| reference_dropout=0.25 | 39.4mm / 6.3deg | 57.9mm / 6.5deg | 36.6mm | +47% |
| reference_dropout=0.50 | 39.9mm / 6.2deg | 58.3mm / 6.6deg | 38.1mm | +46% |
| reference_dropout=0.75 | 40.3mm / 6.6deg | 58.2mm / 6.8deg | 38.3mm | +44% |

**Bandwidth hypothesis falsified.** bw=0.3 gives the best normal-condition accuracy
seen yet (20.6mm raw, even beating bw=0.15), but chained-anchor robustness is
unchanged-to-slightly-worse (+359% vs +345%). Checked directly: retrieval indices
still flip completely under the bw=0.3 checkpoint (zero top-10 overlap) and weight
concentration still jumps sharply (0.016 -> 0.137). The discontinuity lives in the
GRU history-encoder embedding itself, not in how sharply the RBF kernel weights
neighbors once retrieved — a smoother kernel can't fix a discontinuous embedding.

**Dropout has no gradual tradeoff in [0.25, 0.75] — it's a step, not a dial.** All
three values give essentially the same ~44-47% degradation and the same ~39-40mm
normal-condition raw error on this pair (vs ~21mm undropped). Whatever robustness
`reference_dropout` buys, it appears to saturate by 0.25 already, alongside its
accuracy cost. A gentler tradeoff point, if one exists, would have to be found below
0.25 (e.g. ~0.1) — not attempted yet.

## Root-cause fix: remove the raw tool-pose branch from the history encoder (2026-09-21)

Traced the discontinuity to a specific, checkable architectural asymmetry:
`HistoryEncoder.forward` (`models/history.py`) feeds the point cloud through
`normalize_cloud` (centered + scale-normalized) before embedding it, but feeds
`history_tool_positions` into `tool_features` (a single `Linear(pose_dim+2,
hidden_dim)`) as **raw, absolute, completely unnormalized** coordinates — the only
input in the whole pipeline with no invariance built in. (Separately: PointNet's own
`position_mlp` branch does retain the point cloud's centroid, so "point clouds
aren't informative enough" was not the actual mechanism — confirmed irrelevant since
perturbing only `history_tool_positions`, with the point cloud completely untouched,
was already enough to flip retrieval. Also checked whether more points would help:
loaded the raw source `pointclouds_interpolated.pt` directly — native camera export
is exactly 1024 points/frame, matching the dataset name (`camera1024`); `points: 1024`
in the config is already the ceiling, and `sample_points` explicitly repeats rather
than adding real points above that count, so increasing point count is a dead end
for this dataset.)

Removed the branch: `tool_features` now takes only `(elapsed_time, dt)` (2-dim,
was `pose_dim+2`), `history_tool_positions` is no longer read into `auxiliary` at
all. `models/history.py`, `tests/test_history.py` (updated
`test_current_only_ignores_past_and_full_uses_order` to assert the new invariance
instead of the old sensitivity). Full suite: 104/104 still passing.
`history_tool_positions` remains required in the observation batch (`validate_history`
still enforces it) and is still used correctly elsewhere — `current_tool_pose()`
still reads it directly to compose the final orientation-delta output relative to
the real/predicted current pose. Only the *retrieval-query* pathway lost access to
it.

Trained on the current-best config (bw=0.3, decomposed start/shape loss, seed 27)
and reran the chained-anchor test on the same episode18 pair:

| variant | normal raw/orient | chained raw/orient | raw degradation | orient degradation |
|---|---|---|---|---|
| bw=0.3, with tool-pose branch | 20.56mm / 5.13deg | 94.42mm / 8.67deg | +359% | +69% |
| **bw=0.3, branch removed** | 44.12mm / 5.78deg | 44.12mm / 6.34deg | **+0%** | **+9.7%** |

Verified directly, not just via the error numbers: `output.indices`, `output.weights`,
`output.reference`, and `output.raw_residual` are all **bit-exact identical** between
normal and chained runs. The encoder embedding is now provably invariant to the tool
anchor. The residual ~10% orientation sensitivity that remains is expected and
correct, not a leftover flaw: orientation composition (`compose_orientation_delta`)
uses `current_tool_pose(observations)` directly as the rotation base, independent of
retrieval, so it correctly reflects whatever anchor orientation is actually passed
in.

**Real cost**: normal-condition accuracy on this pair got worse (20.6mm -> 44.1mm
raw), and the dataset-wide eval log confirms this isn't pair-specific (31.27mm vs
~20mm raw overall, tool-branch vs no-tool-branch, same config otherwise). The
tool-motion signal that made retrieval sensitive to anchor drift was also carrying
real, useful information for retrieval quality under normal (ground-truth-anchored)
conditions. This fixes *position* brittleness completely (0% vs `reference_dropout`'s
+44-47% residual degradation) but at a larger accuracy cost than dropout (44mm vs
~29-40mm raw normal-condition).

## Recommendation

Add this chained-anchor test as a standard part of the evaluation suite for any
checkpoint intended for real-time/closed-loop use (not just offline ground-truth
anchored metrics) — it surfaces a failure mode that raw/shape/start error on
GT-anchored data completely hides. Wider RBF bandwidth is *not* a fix — it improves
normal-condition accuracy independently but does nothing for this failure mode.

Two real mitigations now exist, different tradeoffs:
- **`reference_dropout` in [0.25, 0.75]** (interchangeable across that range):
  raw degradation +44-47%, normal-condition raw ~29-40mm. Partial fix, moderate
  accuracy cost.
- **Remove the raw tool-pose branch from `HistoryEncoder`**: raw degradation
  **+0%** (position brittleness fully eliminated, verified bit-exact), normal-
  condition raw ~44mm. Complete fix for position, larger accuracy cost, and this is
  the actual root-cause fix rather than a regularizer papering over it.

If closed-loop position robustness matters more than raw accuracy, remove the
branch. If some residual brittleness is acceptable for better normal-condition
accuracy, `reference_dropout=0.25` is the cheaper partial fix. Worth exploring next:
whether the accuracy lost by removing the branch can be recovered by re-adding tool
*motion* information in a form that doesn't carry raw absolute position (e.g.
frame-to-frame deltas within the history window, or the pose relative to the
window's own first entry, rather than the world-frame absolute value) — not
attempted yet.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
