# Policy adapter exercised end-to-end (data side); run.py wiring still missing

At the user's explicit direction ("i think its ok to test it in a simulation"), ran
`policy_adapter.py` / `export_policy_controls.py` (commit `832c6cb`) against a real
prediction, using the diagnostic `T_M_DO` fit from
`RESULT_EPISODE18_SOURCE_FRAME_PROVENANCE.md` **only as a test input**, not as an
accepted calibration. The blocker in `BLOCKER_MPM_POLICY_EXPORT_SOURCE_MISMATCH.md`
/ `FOLLOWUP_MPM_POLICY_EXPORT_SOURCE_MISMATCH.md` remains open for anything beyond
this test.

## What was done

1. `python3 -m unittest experiments.differentiable_mpm.tests.test_policy_adapter -v`
   — 3/3 pass.
2. Loaded `relocated_episode18` raw marker poses/times via
   `reference/deformpath_dynamics.load_observation_sequence` (388 frames, `poses[...,:7]`,
   times relative to frame 0).
3. Took the existing `cartesian_residual` prediction export
   (`/workspace/runs/policy_export_episode18`, checkpoint
   `history_deformpath_full_1000_seed7/seed_7/full/cartesian_residual/checkpoint.pt`,
   `prediction_xyz.npy` shape `[32,2,3]` in `camera_depth_optical_frame`) and applied
   `T_M_DO` (rotation + translation only, no reflection) to get tool positions in the
   `mocap`/source frame that `recorded_poses.npy` and `scene_from_source` expect.
   - Sanity check: tool-to-tool distance is preserved by the rigid transform as
     expected (0.4907 m before -> 0.4911 m after; the ~0.49 m separation just
     rotates from the camera frame's X axis into the mocap frame's Z axis).
4. Ran `export_policy_controls.py` with:
   - `--recorded-poses` / `--recorded-times` from step 2
   - `--policy-export` = the diagnostic-transformed prediction from step 3
   - `--scene-from-source` = `episode18_registered_tools_v1/scene_calibration_v2.json`'s
     `scene_from_source` (`mocap -> table-aligned`)
   - `--marker-from-tool-frames` left default (identity), matching
     `tool_geometry.json`'s `marker_from_collider` = identity for both tools
   - `--control-dt 0.0333` (30 Hz), `--duration 1.4340805858373642` (the segment's
     `duration_s` from its manifest)
   - Output: `/workspace/runs/mpm_policy_test/conditions_episode18_diagnostic/`
     (all 5 condition `.npz` files + manifest, schema
     `taichidough/mpm-policy-controls/v1`)

## Result worth reporting

At `t=0` in the table-aligned scene frame, the diagnostic-transformed predicted
trajectory sits **~2.0 cm** from the recorded trajectory for both tools
(1.99 cm / 1.79 cm), versus the ~13 cm mismatch that motivated the blocker. This is
consistent with the diagnostic fit's tight tool-pose residuals (0.569 mm RMS) and is
a much better sign than the point-cloud alignment failure — but it is one frame of
one episode and does not on its own validate the transform generally.

## Where this stops

`forward_video_v2/run.py` has no argument or code path that consumes
`ConditionControls` / the `.npz` files `save_conditions` writes — grepped for
`policy_adapter`, `ConditionControls`, `--controls` and found nothing. So the
condition files above are ready, but nothing currently drives an actual MPM
simulation from them. Per earlier instruction this integration is left to you
rather than implemented here.

## Request

Wire one of the five conditions (`predicted_xyz_recorded_orientation` is probably
the most informative first case) into `run.py`/`forward_video_v2` as an alternate
tool-control source, so a real simulation can be run and compared against
`recorded_full_pose` under matched physics parameters
(`youngs_modulus=17072.5`, `poisson_ratio=0.4896`, `viscosity=0.3367`,
`plastic_min=0.7278`, `plastic_max=1.0920`, from
`dom_policy_identified_20260919T100203Z_recorded/forward_config.json`). The prepared
condition files and scripts are at `/workspace/runs/mpm_policy_test/` on this host if
useful as a reference for the expected data shapes.

Independent of this: obtaining a real (non-diagnostic) `camera_depth_optical_frame
-> mocap` transform is still needed before any paired-condition result can be
trusted as more than a plumbing test.
