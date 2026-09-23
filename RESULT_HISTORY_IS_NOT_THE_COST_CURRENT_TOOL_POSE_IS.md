# History barely matters on the real best model; what's actually expensive is losing relative_to_start/orientation_delta

Follow-up to `RESULT_MULTIMODAL_GYM_ADAPTER_EVALUATION.md`. That result left an open
question: is the multimodal/gym config's accuracy gap (35-39mm raw, 8.17deg
orientation, vs. this session's best of ~22mm/5.8deg) caused by dropping the
8-step/0.7s history window (a real gym-loop constraint -- `DifferentiableMPMEnv`
never gives you a recorded past-history buffer), or by something else? Ran two
targeted ablations to find out.

## Ablation 1: modes=1 vs modes=3 on the gym config

Isolated whether multimodal clustering itself was the problem. Retrained the
1024pt/+orientation gym config with `retrieval_modes: 1` (bypasses
`cluster_retrieved_trajectories` entirely, falls back to standard single-reference
RBF retrieval), everything else identical, 3 seeds.

| | start | shape | raw | orientation |
|---|---|---|---|---|
| modes=3 | 39.66+/-2.52mm | 28.94+/-0.56mm | 38.61+/-3.13mm | 8.17+/-0.24deg |
| modes=1 | 36.85+/-1.30mm | 26.47+/-0.32mm | 35.58+/-1.13mm | 8.17+/-0.24deg |

Orientation identical to two decimal places; position ~7-9% *better* without
clustering. Modes are not the problem, and multimodal clustering isn't helping on
aggregate accuracy either way (may still matter for specific genuinely-multimodal
cases -- not tested here, this is an aggregate-accuracy read only).

## Why orientation is bad: a hard architectural lock, not a data cost

Tried turning on `trajectory_representation: relative_to_start` and
`pose_parameterization: orientation_delta` on top of modes=1 (both now unblocked
since the num_modes>1 restriction no longer applies). It doesn't just perform
worse -- it **crashes immediately**:

```
ValueError: orientation_delta requires measured 14D tool poses
```

`current_tool_pose(observations)` -- needed by both `orientation_delta` (final
composition) and `relative_to_start` (recomposing the retrieved reference against
the actual current pose) -- has nothing to read under
`observation_mode: current_pointcloud_only`, which excludes tool position
entirely, not just tool-position *history*.

**This is stricter than gym compatibility actually requires.** Checked
`DifferentiableMPMEnv` (TaichiDough, `bc977a4`) directly: its
`observation_space` includes `tool_poses` (shape `(2,7)`) on *every* step --
the gym env always gives you the current tool pose, it just never gives you a
recorded history of past ones. So `current_pointcloud_only` throws away more
than gym-compatibility demands: it should exclude history (a real constraint) but
currently also excludes current tool position (not a real constraint -- the gym
env has it).

## Ablation 2: does history actually matter, on the architecture that isn't locked out?

Tested directly on this session's actual best model (`relative_to_start` +
`orientation_delta`, bw=0.3, 1024pts, decomposed loss) -- set
`history.mode: current_only` (an existing, already-supported `HistoryEncoder`
option -- ignores the temporal window, uses only the last valid frame), 3 seeds,
compared to the established full-history result.

| | start | shape | raw | orientation |
|---|---|---|---|---|
| full history (8-step/0.7s) | 2.36+/-0.28mm | 22.10+/-1.14mm | 22.05+/-0.97mm | 5.80+/-0.12deg |
| **current pose only, no history** | **1.97+/-0.11mm** | **22.50+/-0.99mm** | **22.61+/-1.04mm** | **5.86+/-0.06deg** |

**Essentially indistinguishable** -- raw +2.5%, start actually *better*
current-only, orientation flat. On the architecture that still has
`relative_to_start`/`orientation_delta` available, dropping the entire temporal
history window costs almost nothing.

## Conclusion

The gym config's accuracy gap is not a history-loss problem. It's specifically the
consequence of `current_pointcloud_only` excluding *current* tool position (not
just history), which hard-blocks both `relative_to_start` and `orientation_delta`
-- the two things this session already found actually matter
(`RESULT_RELATIVE_TO_START_TRAJECTORY_REPRESENTATION.md`,
`RESULT_STOP_GRAD_REFERENCE_EXPOSES_START_POINT_SHORTCUT.md`). Losing the temporal
window alone would have cost almost nothing, per the current-only ablation above.

## Recommendation

Add a new observation mode -- current point cloud + current tool pose, no
history -- that matches exactly what `DifferentiableMPMEnv` actually provides
each step, rather than reusing `current_pointcloud_only` which excludes more than
necessary. This is a real code change (`data.py`/`observations.py`/
`trainer.py`'s `observation_mode` handling), not a config tweak, since
`current_pointcloud_only` currently forbids tool positions outright
(`make_encoder` raises if `include_tool_positions`/proprioception is set alongside
it). Given the current-only ablation above, a model built this way should land
close to ~22mm/5.8deg rather than the ~35-39mm/8.2deg the existing gym config
gets -- worth validating once that observation mode exists.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
