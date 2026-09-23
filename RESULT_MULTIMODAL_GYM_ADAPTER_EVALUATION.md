# Multimodal / "gym adapter" commit: bug found+fixed, evaluated, and a naming clarification

Follow-up to `5e6e751 "gym adapter"` (dom_retrieval) and `bc977a4 "gym"` (TaichiDough).
Pulled both, read the diffs in full, fixed a real bug in the new multimodal loss,
trained and evaluated the new architecture against this session's established
baselines, and traced the "gym" naming — it doesn't currently mean what it sounds
like.

## What the commit actually adds

Real, well-designed architecture work, three pieces:

1. **Multimodal retrieval**: `cluster_retrieved_trajectories` (`models/retrieval.py`)
   splits the RBF-retrieved candidates into `num_modes` deterministic clusters
   (farthest-point seeding). Each mode gets its own reference + residual
   correction (`PolicyOutput.mode_predictions`/`mode_probabilities`). Training
   uses a best-of-N loss (`multimodal_trajectory_loss` in `training/losses.py`) --
   whichever mode matches the target best gets the gradient, so modes specialize
   rather than average. Verified this produces genuine (not collapsed) behavior:
   on a 3-mode, 85-example test set, `selected_mode` distribution was
   {0: 62, 2: 15, 1: 8} -- real variation across queries, not a single dominant
   mode -- and mean max-mode-probability was 0.49 (a real but soft preference, not
   sharp commitment).
2. **Shape/position separation in the encoder**: `PointNetEncoder` gained
   `include_position_features` (default `True` for backward compat). The new
   example config sets it `False`, so the retrieval embedding is built from
   centered geometry only; absolute centroid position is routed separately via
   `residual_centroid_context` directly into the residual network instead of
   contaminating the retrieval embedding. This is the same fix this session
   already made for the tool-pose branch (`RESULT_CHAINED_ANCHOR_BRITTLENESS.md`),
   now applied to the point-cloud side.
3. **`observation_mode: current_pointcloud_only`**: no history, no tool-position
   input at all -- the actual gym-compatibility piece, since a
   `DifferentiableMPMEnv` step only ever gives you the current particle state,
   never a recorded history buffer.

## Bug found and fixed

`multimodal_trajectory_loss`'s `start`/`shape` terms crash on pose (quaternion)
targets -- `RuntimeError: shape '[20, 3]' is invalid for input of size 180` --
because `pose_components` returns an extra per-tool axis that `position_loss`
(the working term right above it) correctly reduces with `dim=(1,2,3)`, but
`start`/`shape` only reduced `dim=1` / `dim=(1,2)`, leaving a stray xyz axis
unreduced before reshaping. Only shows up with pose (14D, xyz+quaternion) targets
-- the shipped example config uses `target_representation: xyz` (no orientation
at all), so this path was never exercised. Confirmed via the existing test suite:
111/111 pass both before and after the fix, meaning the multimodal tests added in
this same commit only cover the plain-xyz path -- a real coverage gap.

Fix (`training/losses.py`): replaced the hardcoded `dim=1` / `dim=(1,2)` with
`dim=tuple(range(1, predicted_xyz.ndim - 1))` / `dim=tuple(range(1, predicted_shape.ndim))`,
which correctly reduces "everything except batch" regardless of whether an extra
per-tool axis is present. Verified against both branches (plain xyz and pose) by
inspection of the resulting tensor ranks, and confirmed end-to-end by training
successfully afterward.

## Evaluation, 3 seeds each, against this session's established models

| | start | shape | raw | orientation |
|---|---|---|---|---|
| baseline (absolute repr., full history/tool-pose, 1024pts, `orientation_delta`) | 10.88mm | 19.63mm | 19.95mm | 5.46deg |
| `relative_to_start` (3-seed) | 2.36+/-0.28mm | 22.10+/-1.14mm | 22.05+/-0.97mm | 5.80+/-0.12deg |
| multimodal/gym config as shipped (64pts, xyz-only, 3-seed) | 39.13+/-1.81mm | 29.00+/-0.68mm | 38.46+/-1.21mm | N/A |
| multimodal/gym config, 1024pts + orientation added (3-seed) | 39.66+/-2.52mm | 28.94+/-0.56mm | 38.61+/-3.13mm | 8.17+/-0.24deg |

Two findings worth flagging:

- **Point density (64 -> 1024, 16x) changed nothing on position accuracy**
  (raw 38.46mm -> 38.61mm, within noise). Whatever is limiting this architecture's
  position accuracy isn't point count -- worth looking elsewhere (no history/
  tool-position at all is the more likely constraint, or general capacity for
  this harder multimodal objective).
- **Orientation (8.17deg) is worse than anything else measured this session**
  (5.46-5.80deg for the others). Root cause, not just a training issue: multimodal
  retrieval currently *requires* `pose_parameterization="absolute"` --
  `orientation_delta` is explicitly blocked (`RetrievalPolicy.forward` raises if
  `num_modes > 1` and `pose_parameterization == "orientation_delta"`). Absolute
  orientation prediction is a structurally harder task than the delta-from-
  current-pose composition our best models use. This isn't evidence multimodal
  retrieval hurts orientation prediction in general -- it's evidence that giving
  up `orientation_delta` to get multimodal support has a real, measurable cost.
  If this direction is worth pursuing, extending multimodal retrieval to support
  `orientation_delta` composition (each mode composes against
  `current_tool_pose` the same way the single-mode path does) looks like the
  actual fixable lever, not more data or more points.

## "gym" naming clarification

Checked directly rather than assume: the string "gym" / "gymnasium" appears
**nowhere in this commit's actual code** (`git show 5e6e751 | grep -i gym` only
matches the commit message itself), and there's no `import gym`/`gymnasium`
anywhere in the dom_retrieval tree. No reference to TaichiDough's
`DifferentiableMPMEnv` (from `bc977a4 "gym"` on the TaichiDough side, itself a
real, working `gymnasium.Env` subclass -- verified separately this session by
running it end-to-end on episode20 chunk07 via
`/workspace/runs/gym_test/test_gym_episode20.py`: 2200/2836 steps, real contact
confirmed via the env's own diagnostics, ~215 steps/s steady-state on cuda, ~23x
slower than real time). No script anywhere connects the two repos.

So: this commit builds real prerequisites for gym compatibility (no more
history/tool-position dependency, which would've made consuming
`DifferentiableMPMEnv`'s bare `{particles, tool_poses}` observations impossible),
but it is not itself a gym adapter and hasn't been tested against the gym
environment. The name describes intent, not current content -- worth being
precise about this distinction in future references to this commit.

## Recommendation

Don't read the evaluation numbers above as "multimodal retrieval is worse than
our other work" -- the comparison is confounded by giving up `orientation_delta`
(not a multimodal-inherent cost) and by the no-history/no-tool-position
constraint (a real gym-compatibility requirement, not a free choice). The
cleanest next test of whether multimodal clustering + shape/position separation
are good ideas on their own merits would be: extend multimodal retrieval to
support `orientation_delta`, then compare against `relative_to_start` with
matched history/tool-position settings -- that isolates the actual
mode-clustering idea from the gym-driven observation stripping it's currently
bundled with.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
