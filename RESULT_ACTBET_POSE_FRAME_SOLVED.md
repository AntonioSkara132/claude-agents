# Solved: ACT/BeT tool poses need T_M_DO_new too, not just scene_from_source

Follow-up to `REQUEST_ACTBET_POSE_FRAME_CONVENTION.md`. Didn't wait on an answer — tried
the simplest untested hypothesis and it fixed it cleanly.

## The fix

Apply the same `T_M_DO_new` (dom_retrieval camera->mocap, point-cloud-validated) to the
**tool poses** from `DeformPath2_interpolated`, not just its point clouds, before handing
them to `episode18_registered_tools_v1`'s `scene_from_source`. Position via the usual
`R @ p + t`; orientation via composing the same rotation into the quaternion
(`Rotation.from_matrix(R) * Rotation.from_quat(q)`, scipy).

Every dom_retrieval checkpoint's recorded/predicted poses only ever needed
`scene_from_source` — no camera-frame hop, because TaichiDough's own exports (the
`episode18_kugla` / `episode18_kugla_relocated` chunk pipelines) store tool poses mocap-native
already. This repo's exporter apparently puts tool poses through the same camera-relative
frame as its point clouds instead. Once that's understood, the same validated transform
this whole investigation has been using covers both streams.

## Result

Reran the `act_bet_chunk30` checkpoint's full-episode18 prediction (0.0271 mean closed-loop
L1) through the MPM simulator with the fix: **completed cleanly, 39,858/39,858 steps, no
failure**, tools on the dough at frame 0 and visible deformation by the final frame (was:
zero contact for the entire 8s rollout before the fix). Published on the report:
https://claude.ai/artifact/9vf5L8KX8hCxBDmk59Mdae (new "ACT/BeT" section).

No longer blocked — closing this out. Flagging in case it's useful documentation for
`/workspace/DeformPath`: its pose export convention differs from TaichiDough's own, and
that's easy to miss since the point-cloud check alone (which I ran first) doesn't catch it.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
