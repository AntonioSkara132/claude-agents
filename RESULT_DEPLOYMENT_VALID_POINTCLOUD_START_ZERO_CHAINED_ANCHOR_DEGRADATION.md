# Genuinely deployment-valid model: zero measured pose anywhere, exactly 0% chained-anchor degradation

Follow-up to `RESULT_POINTCLOUD_CONTRASTIVE_LOSS_FIXES_CHAINED_ANCHOR_BRITTLENESS.md`.
That report treated the `ProprioceptiveEncoder` coupling (raw tool pose feeding the
retrieval-selection embedding) as the root cause of chained-anchor brittleness. It
was a real bug, but fixing it wasn't the actual point -- a real-time/gym run never
has a ground-truth *current tool pose* to give the model as an input at all, whether
through the encoder or through decode. Every `current_pointcloud_with_tool_pose`
variant tested so far (including the `retrieval_all`/`soft_clustering` and InfoNCE
contrastive results) still read a measured `current_tool_pose(observations)` at
decode time (`relative_start_source: current_tool_pose`) -- not deployable, just
less obviously broken.

## The actually-correct architecture already existed: `relative_start_source: pointcloud`

`path_start` (the anchor used to decode `relative_to_start` candidates) is predicted
from the point-cloud embedding alone -- `self.path_start(embedding, context)`, no
measured pose input anywhere in the forward pass. This was flagged as broken back in
`RESULT_NEW_FIXED_POLICY_EVALUATION.md` (143.45deg orientation error, `residual`
force-disabled) and never fixed. Two real bugs, now both fixed
(`dom_retrieval` commit `16bbd31`):

1. **`self.residual` and `self.path_start` were mutually exclusive by construction**
   (`relative_start_source == "pointcloud"` unconditionally set `self.residual = None`
   regardless of the `residual`/`residual_input` arguments). Decoupled -- both are
   now independent options.
2. **`path_start` only ever received gradient indirectly**, through the final
   trajectory loss -- a very weak signal for something directly measurable in the
   training data. Added `PolicyOutput.predicted_start` (exposed from every
   `RetrievalPolicy` forward branch) and `LossWeights.predicted_start`, wired as a
   **training-only** auxiliary loss in `train_policy` comparing `predicted_start`
   against `current_tool_pose(x)` -- the measured pose, used purely as a supervision
   target. Verified this never reaches the forward pass with an exact-invariance
   test: perturbing `tool_positions` by +5.0 leaves `model(x).prediction` bit-identical.

Also updated the canonical config (`configs/deformpath_current_pointcloud_multimodal.yaml`,
commit `cafe92d`) from `current_pointcloud_only` (which stripped tool pose from the
data *entirely*, so there was never anything to supervise `path_start` against -- this
is very likely why the original bug went unfixed) to `current_pointcloud_with_tool_pose`
+ `proprioception.enabled: false` -- tool pose now exists in the training data for
loss purposes, but `proprioception.enabled: false` guarantees the encoder never
touches it (verified via `PointCloudOnlyEncoder`, a new adapter that lets any
point-cloud-only encoder safely ignore extra dict fields it's handed alongside the
point cloud). Also folded in `retrieval_all: true` and `soft_clustering: true` from
the same investigation thread (see previous report), and switched to decomposed
`start`/`shape` loss instead of raw `trajectory`/`orientation` MSE, matching what
worked best earlier in this whole investigation.

## Results, 3 seeds (7/17/27), full test set

| | raw | orientation | start error | shape error | chamfer<->retrieval corr |
|---|---|---|---|---|---|
| `vinn_rbf` (no residual) | ~30-36mm | **143.3-143.6deg** | -- | -- | -- |
| `cartesian_residual` (fixed) | 30.3-36.5mm (mean 33.3) | **7.19-8.08deg (mean 7.76)** | 43.6-55.2mm (mean 50.4) | 24.2-24.7mm (mean 24.5) | **0.28-0.35 (mean 0.33)** |

Compare to the best measured-pose variant (`retrieval_all`+`soft_clustering`,
`relative_start_source: current_tool_pose`, not deployable): 19.83mm raw, 4.84deg
orientation, 0.106 chamfer correlation.

**Chained-anchor test**: normal and chained results are **bit-for-bit identical**
across all 3 seeds -- not an approximate 0%, an exact 0%, because the model has no
anchor input left for the test to perturb. This is the conclusion of the whole
chained-anchor investigation thread that's run across this session and the one
before it (HistoryEncoder pose_representation, reference_dropout, relative_to_start,
ProprioceptiveEncoder decoupling) -- this is the first architecture where the
property is guaranteed by construction rather than empirically approximated.

## Two things worth flagging, not yet resolved

1. **The cost of true deployability is real and concentrated in `start` error**
   (43.6-55.2mm vs. 1-3mm for every measured-pose variant) -- `path_start` has to
   guess the current pose from geometry instead of reading a sensor, and that guess
   is meaningfully worse. Not a bug, just the honest price, and it dominates the
   raw-accuracy gap versus the non-deployable variants.
2. **`path_start`'s own predicted orientation is still ~143deg (indistinguishable
   from random) even with direct supervision** -- the entire final-orientation
   improvement (143deg -> 7.76deg) comes from the residual network, which apparently
   learns to disregard `path_start`'s orientation component almost entirely and
   reconstruct it some other way (it has its own access to the observation).
   Plausible explanation, not yet verified: the point cloud captures the *dough's*
   surface, not the *tool's* -- there may be little to no geometric signature for
   tool orientation specifically (as opposed to tool position, which correlates with
   where the dough is being deformed). If true, no amount of auxiliary-loss weight
   tuning on `predicted_start` will fix this -- it would need either a different
   observation (tool visible in the point cloud, or some other orientation cue) or
   accepting that `path_start`'s orientation output is structurally unusable and
   should perhaps not be decoded/composed against at all (position-only path_start,
   let residual own 100% of orientation). Flagging as open, not guessing further
   without testing it directly.

## Recommendation

This is now the correct default for anything meant to run in the gym/real-time --
already made the standing default in `configs/deformpath_current_pointcloud_multimodal.yaml`.
It is measurably worse on raw accuracy than the tool-pose-input variants (33mm vs
20-25mm) but that comparison was never fair to begin with -- those variants use an
input a real deployment doesn't have. Suggest the next investigation either verifies
or rules out the "point cloud doesn't encode tool orientation" hypothesis directly
(e.g., check whether `path_start`'s orientation-only loss ever meaningfully decreases
during training at all, independent of position) before spending more effort tuning
`predicted_start`'s weight or the auxiliary loss's construction.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
