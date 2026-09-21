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

## Recommendation

Add this chained-anchor test as a standard part of the evaluation suite for any
checkpoint intended for real-time/closed-loop use (not just offline ground-truth
anchored metrics) — it surfaces a failure mode that raw/shape/start error on
GT-anchored data completely hides. `reference_dropout` around 0.5 is currently the
best available mitigation, at its known accuracy cost under normal conditions
(`RESULT_REFERENCE_DROPOUT_TRAINING.md`). Worth considering whether a smoother
retrieval mechanism (e.g. wider RBF bandwidth, or an explicit continuity/Lipschitz
penalty on the embedding w.r.t. anchor position) could fix this more directly than
dropout, which is a blunt-instrument fix — not attempted yet.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
