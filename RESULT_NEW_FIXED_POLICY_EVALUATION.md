# "new fixed policy" (e21dd36): confirms both prior fixes, finds two new real bugs

Follow-up to `RESULT_HISTORY_IS_NOT_THE_COST_CURRENT_TOOL_POSE_IS.md` and
`RESULT_MULTIMODAL_GYM_ADAPTER_EVALUATION.md`. Pulled `e21dd36 "new fixed policy"`.
Good news first, then two new bugs found and one fixed.

## The good news: `current_pointcloud_with_tool_pose` config now matches our best models

Both bugs from the previous evaluation are independently fixed in this commit --
`models/retrieval.py`'s clustering-distance shape bug (their fix: reshape-then-
broadcast; mine was equivalent, discarded in favor of theirs) and (mostly) the
`start`/`shape` reduction bug in `multimodal_trajectory_loss` (see below for the
part that's still broken). Trained the `current_pointcloud_with_tool_pose` +
multimodal (modes=3) + `relative_to_start` config from the prior result, 3 seeds:

| | raw | orientation |
|---|---|---|
| baseline (full history/tool-pose, no multimodal) | 19.95mm | 5.46deg |
| `relative_to_start` (3-seed, full history) | 22.05+/-0.97mm | 5.80+/-0.12deg |
| **gym-compatible, multimodal, current tool pose only (3-seed)** | **23.06+/-0.09mm** | **5.39+/-0.10deg** |

Remarkably tight across seeds and essentially matches (slightly beats on
orientation) the best non-gym-compatible model, while being genuinely
gym-compatible (no history) and multimodal. Strong confirmation of last time's
diagnosis and this commit's fix.

## New bug 1 (found, fixed): quaternion terms in start/shape still don't fully reduce

This commit's own fix for the `start`/`shape` dimension bug (my previous report)
only handled `shape`'s xyz-position term correctly, not `start`'s, and the new
quaternion-awareness it adds to both terms crashes immediately on any multimodal +
pose-target training:

```
RuntimeError: The size of tensor a (3) must match the size of tensor b (60) at non-singleton dimension 1
```

Root cause, verified directly: `pose_components` returns xyz with shape
`[batch*modes, horizon, 2, 3]` (extra per-tool axis). `start`'s position term used
`.mean(dim=1)` (only reduces the tool axis, leaves xyz unreduced -- `[N,3]`
instead of `[N]`); `shape`'s position term used `.mean(dim=(1,2))` (reduces
horizon+tool, same issue). Both then get added to correctly-reduced `[N]`-shaped
quaternion terms, causing the broadcast crash. Existing test suite (113/113,
including the 2 new tests this commit added) didn't catch it -- still a real
coverage gap for multimodal + pose-target combinations specifically.

Fixed (`training/losses.py`): replaced the hardcoded `dim=1` / `dim=(1,2)` with
`dim=tuple(range(1, tensor.ndim))`, which reduces "everything except batch"
regardless of whether the extra per-tool axis is present -- verified against both
the pose and plain-xyz branches directly, and confirmed by successful end-to-end
training afterward. Suggest checking this pattern doesn't recur -- this is the
second time a `start`/`shape` reduction over pose-shaped tensors has needed this
same fix; might be worth a shared helper instead of re-deriving the reduction
dims at each call site.

## New bug 2 (found, not fixed -- flagging for a design decision): `relative_start_source="pointcloud"` disables the residual network entirely

Tested the new example config (`current_pointcloud_only` + `relative_start_source:
pointcloud` -- predicting the current pose from the point cloud instead of
requiring a measured one, genuinely the most gym-compatible setup possible: zero
proprioception input at all). Two problems, both real:

1. **`cartesian_residual` and `vinn_rbf` produce byte-identical metrics.** Checked
   directly: `model.residual is None` for `cartesian_residual` under this
   config. In the constructor, `relative_start_source == "pointcloud"`
   unconditionally sets `self.residual = None` regardless of the `method`/
   `residual` argument -- `path_start` and the residual correction network are
   currently mutually exclusive by construction, not independent features. That
   looks like an oversight, not an intentional design choice -- there's no
   obvious reason predicting the start pose from the point cloud should preclude
   also residual-correcting the retrieved reference.
2. **Orientation error is catastrophic: 143.45deg** (essentially random --
   compare to 5.4-5.8deg for every other config tested this session). Not an
   undertraining artifact -- converged normally, 58 epochs, early-stopped at best
   epoch 27, `path_start`'s own weights are nonzero (genuinely trained, not
   stuck at zero-init). `path_start` predicts the anchor pose used to compose
   every retrieved candidate's absolute orientation via `compose_orientation_delta`
   -- if that predicted quaternion is bad, it would corrupt every downstream
   orientation prediction regardless of how good retrieval/clustering is. Haven't
   isolated whether this is a quaternion-normalization issue in `path_start`'s
   output, a genuine difficulty of predicting orientation from point-cloud shape
   alone (plausible -- dough point clouds don't obviously encode tool orientation
   the way they might encode tool position), or something else -- flagging as an
   open, unresolved problem rather than guessing further.

## Recommendation

The `current_pointcloud_with_tool_pose` (measured pose) config is solid and
matches our best models -- safe to build on. The `current_pointcloud_only` +
`relative_start_source: pointcloud` config (zero proprioception, fully
vision-based anchor) is not yet usable: needs (a) `path_start` and `residual` made
independent rather than mutually exclusive, and (b) the orientation-prediction
failure investigated before drawing any conclusion about whether pointcloud-only
anchor prediction is viable at all.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
