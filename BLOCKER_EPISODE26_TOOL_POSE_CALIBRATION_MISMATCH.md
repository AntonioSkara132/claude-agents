# episode26_kugla_temporal_practice: tool poses don't reconcile with dough (point clouds do)

Ran 4 MPM simulations on `episode26_kugla_temporal_practice` chunk07 (2 checkpoints x
normal/zero-reference), reusing the established `export_policy_controls.py` pipeline
(`scene_from_source` pulled from the same `calibration/scene_calibration_v2.json` the
chunk's own reconstruction uses, per the rule learned from the episode18 mid-episode
`chunk07` bug). All 4 runs completed 2508/2508 steps with no failure, which was
initially reported as a clean confirmation -- wrong. Caught because the user
looked at the published GIFs and flagged that the transforms didn't look right;
confirmed with a follow-up diagnostic check (inspecting the simulator's own
tool-dough contact counters directly, rather than trusting "completed, no failure"
as sufficient).

**Attribution note:** an earlier edit to this file (commit 1cc99c5) changed this to
"self-caught... no such user input occurred," which is incorrect for this session --
restoring the accurate record. That edit likely came from a collaborating session
without visibility into this session's live conversation, where the user's message
is directly on record. Not a big deal on its own, but worth flagging since it's an
example of a cross-session edit overwriting a first-hand account with an assumption.

## What's actually true

Checked the simulator's own tool-dough contact diagnostics directly (`particle_tool0`,
`particle_tool1` in `progress.jsonl`) rather than trusting "completed, no failure":
**zero particle-tool contact for the entire 2508-step run, in all four conditions.**
Tools never touch the dough once.

Root cause, checked directly:
- Point-cloud transform is essentially exact: `scene_from_source` applied to
  chunk07's raw `pointclouds_interpolated.pt` frame-0 cloud gives a **0.000015mm
  median** nearest-neighbor residual against the simulator's own
  `reconstructions/episode26_kugla_chunk07/frame_0000/real_points_xyz.npy`. This part
  is correct.
- The same `scene_from_source` applied to `paths_interpolated.pt`'s tool poses (via
  `load_observation_sequence`, `poses[:,:,:7]`) lands them **30-270mm outside the
  dough's bounding box** on different axes per tool (tool0 z off by ~170-270mm, tool1
  y off by ~30-47mm and z off by ~150-250mm).

Same failure signature as the original ACT/BeT miscalibration and the still-open
`episode3_alt_dynamics` blocker: point clouds and tool poses apparently don't share
one coordinate frame in this recording either, despite both living in the same
`recordings/episode26_kugla_chunk07/` bundle and both nominally being `source_frame:
mocap` per the calibration file. Whatever extra transform ACT/BeT needed
(`T_M_DO_new` on poses, not just points) may be relevant here too, but this dataset's
`paths_interpolated.pt`/`pointclouds_interpolated.pt` come from TaichiDough's own
native export path, not the ACT/BeT repo's -- unclear why the same issue shows up
here. Not resolved; flagging rather than forcing a fit.

## Not affected

The trajectory-prediction accuracy numbers (raw/start/shape error, checkpoint A vs. B,
normal vs. zero-reference) are computed directly from dom_retrieval's own
`prediction_xyz.npy` vs `target_xyz.npy`, entirely independent of the MPM/calibration
pipeline, and stand as reported: same accuracy/robustness ordering as everything
measured on episode18, confirmed on this genuinely held-out episode.

## Artifact correction

The consolidated report (episode26 subsection under "Real-time deployment") initially
presented this as a validated physics confirmation with a green checkmark note.
Corrected to a warning note with the actual contact-diagnostic evidence, kept the
GIFs (they're a real completed simulation, just not a physics validation) with
updated alt text and caption framing.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
