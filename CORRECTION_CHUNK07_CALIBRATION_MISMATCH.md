# Correction to RESULT_CHUNK07_MID_EPISODE_SIM.md: tool calibration was wrong, now fixed

The chunk07 mid-episode sim reported in `RESULT_CHUNK07_MID_EPISODE_SIM.md` used the
wrong tool calibration for the control archive. Flagged by inspection (tool placement
looked visibly off) and now fixed.

## What was wrong

That run used:
- `scene_from_source`: identity, from chunk07's own local `scene_calibration_v2.json`
- `marker_from_tool_frames`: the real registered-tool offsets from
  `episode18_registered_tools_v1/tool_geometry.json`'s `marker_from_mesh` (nonzero
  rotation/translation per tool)

Every other run on the hosted page instead uses:
- `scene_from_source`: the real calibration matrix from
  `episode18_registered_tools_v1/scene_calibration_v2.json`
- `marker_from_tool_frames`: **identity**

These are two different, inconsistent combinations. I don't know for certain whether
the simulator's SDF tool-collision stack already bakes `marker_from_mesh` in
internally when placing the tool mesh (which would mean my chunk07 version was
double-applying that offset to the control trajectory) — I didn't verify that in code,
just matched what every other successful run already does.

## Fix and result

Rebuilt chunk07's control archive using `episode18_registered_tools_v1`'s real
`scene_from_source` with identity `marker_from_tool_frames`, matching every other run.
Reran: completed cleanly again (2669/2669 steps, no failure, 0.5338s simulated).
Updated on the hosted page along with a "classic" open-loop rollout for the matching
dom_retrieval query (validation-split index 8, 30.0mm avg / 7.6deg orientation error
for that specific query) shown side by side with the corrected sim:
https://claude.ai/artifact/MmxhTp5Gxscb4MGajC2sk6

## Open question for you

Is `marker_from_tool_frames` ever supposed to be nonidentity for control-archive
export in this SDF/registered-tools setup, or does the simulator already account for
`marker_from_mesh` when placing the tool collision geometry (making identity correct
for the control stream, with the registration only mattering for collision-mesh
placement)? If nonidentity is sometimes correct, I'd want to know when, since chunk07
was the first run where I had a `tool_geometry.json` on hand and reached for it by
default per `REGISTERED_TOOLS.md`'s guidance for collision meshes — which turned out
not to apply to the control-pose transform.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
