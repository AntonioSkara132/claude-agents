# Tool-pose transform gap recurs identically across episode26 and episode30 -- likely a shared `_temporal_practice` pipeline issue, not per-episode miscalibration

Follow-up to `BLOCKER_EPISODE26_TOOL_POSE_CALIBRATION_MISMATCH.md`. User asked to try
`episode30_kugla_temporal_practice` (appeared in the data directory mid-session,
20 chunks, `status: valid`, `reconstruction_run: true`, `simulation_run: false` --
prepared but never forward-simulated, same state episode3 was in before this
investigation). Chunk<->dom_retrieval-segment correspondence verified exactly (all
20 frame counts match 1:1, unlike episode26's off-by-one on chunk01).

Ran the same point-cloud + tool-pose diagnostic used to catch the episode26 bug,
before burning a full simulation:

- Point-cloud transform: **0.0000157mm median residual** (chunk07 frame 0 vs the
  simulator's own reconstruction) -- essentially exact, same as episode26.
- Tool poses, same `scene_from_source` matrix: tool0 lands ~165-251mm *below* the
  dough bounding box (z-axis), tool1 lands ~150-236mm *above* it, both x/y roughly
  plausible. **Same directional signature, same rough magnitude, as episode26.**

## Ruled out as the cause

- **Pose-array misparsing**: checked the raw `[T,2,14]` array directly. Indices
  `[3:7]` are a valid unit quaternion (norm ~1.0 exactly), confirming `[0:3]` really
  is position and the parsing (`poses[:,:,:7]` via `load_observation_sequence`) is
  correct.
- **Tool-geometry marker offset**: `episode18_registered_tools_v1/tool_geometry.json`'s
  `marker_from_mesh`/`marker_from_collider` translations are only ~1-3cm, nowhere
  near the 15-27cm gap observed.

## What this means

Two independent `_temporal_practice` episodes (26, 30) show the identical failure
shape. That consistency is stronger evidence than a one-off per-episode
miscalibration -- it points to a gap shared by this specific data-prep pipeline
(`_temporal_practice`), as opposed to the fully-validated `_v1`/`_v2` preparations
(episode18, episode20) which don't have this problem. Likely candidate, by analogy
to the ACT/BeT fix earlier in this investigation: an extra frame hop needed for
poses that isn't needed for points (ACT/BeT needed `T_M_DO_new` applied to poses,
not just point clouds, while TaichiDough's own `_v1`/`_v2` exports needed no such
hop). Not confirmed -- no fix attempted yet, flagging the pattern rather than
guessing further down the same unproductive path episode3 went down.

## Recommendation

If more held-out MPM validation is wanted, this needs either (a) working out
whatever extra pose-frame transform `_temporal_practice` exports require (a fresh
investigation, likely comparable effort to the unresolved episode3 case), or (b)
checking whether any *other* episode uses the validated `_v1`/`_v2`-style
preparation instead of `_temporal_practice`. episode20 is the only other one seen
so far and it's training data, not held-out.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
