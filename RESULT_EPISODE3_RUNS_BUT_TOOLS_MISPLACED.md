# Episode3 alt-dynamics: crash fixed, but tool poses land nowhere near the dough

Follow-up to `RESULT_EPISODE3_INPUT_VALIDATION.md`. Relabeled
`table_aligned_identity_calibration.json`'s `source_frame` to `"mocap"` in a local
scratch copy (per your call that it's the same frame) and propagated the resulting
fingerprint change into a local `reconstruction_metadata.json` copy too, since
`load_reconstruction_metadata` cross-checks a `calibration_fingerprint` baked into the
metadata against the calibration actually loaded. Also found and worked around a real,
separate issue: the source `training`/`validation` windows this run.py derives are
`[1,42]`/`[43,43]` regardless of what's in the per-chunk config file (edits to the
config's own window fields are silently ignored — run.py recomputes them from something
else, frame count probably), and this specific chunk
(`episode3_altdynamics_473a99dace623f9e_pc000066_pc000138`) has a genuine 0.133s tool
timestamp gap at index 40->41, just over the 0.1s `replay_max_gap_s` default. Used
`--end-frame 39` to stay inside the continuous stretch.

**With both fixed, the simulation runs**: 6505/6505 steps, no failure, chunk_config
mass/physics params applied correctly.

## But: tools render nowhere near the dough, for the whole 40-frame window

Not a "tool hasn't reached the object yet" situation — checked frame 0 and frame 39
(full window), tools sit in the same distant spot both times, not approaching the dough
at all. Dug into why with the raw numbers for this chunk's frame 0:

- Dough reconstruction bbox: x=[0.225, 0.315], y=[0, 0.033], z=[0.348, 0.480]
  (`sampled_particles_xyz.npy`, in `episode3-table-aligned` scene frame).
- Raw tool0/tool1 positions from `paths_interpolated.pt`, frame 0:
  `[-0.078, 0.731, -0.431]` and `[0.069, 1.010, -0.443]`.

Tried both transforms I have access to on those raw positions:

1. **Identity** (i.e. trusting the README that poses are pre-transformed, matching the
   "identity" calibration name): y stays ~0.73-1.0 (dough's y range is 0-0.033, so this
   reads as ~700-1000mm off) and z stays negative (-0.43) while the dough's z is
   positive (0.35-0.48) — different sign entirely, not just an offset.
2. **`coordinate_transform.json`'s matrix** (`scene_from_camera`, in case the raw poses
   are actually still in `camera_depth_optical_frame` despite the calibration's identity
   label): result is `[0.154, 0.966, 1.105]` / `[0.302, 1.003, 1.380]` — also nowhere
   near the dough (y and z both far outside its bbox).

Neither transform I have access to reconciles the tool poses with the dough
reconstruction. This reads like the raw pose stream in this export is in a third frame
(robot base frame? uncalibrated mocap world frame?) that needs its own transform,
similar to how episode18's registered-tools setup needed a dedicated
`scene_from_source` specifically for **mocap tag poses**, separate from
the camera-to-mocap chain used for point clouds. Or there's a scale/unit issue I'm not
seeing. Not confident guessing further without risking exactly the kind of unverified
"looks plausible" fix that caused the chunk07 calibration bug earlier today.

## What I need

Is there a mocap-tag-to-scene calibration for episode3 (analogous to
`episode18_registered_tools_v1/scene_calibration_v2.json`'s `scene_from_source`, used
specifically for tool poses, not point clouds) that I'm missing? Or is the raw
`paths_interpolated.pt` pose data itself off for this export (wrong units, wrong
sign convention, or genuinely a different source than what got reconstructed for the
dough)?

Sim GIF (tools floating away from the untouched dough, for reference) and full repro are
under `/workspace/runs/episode3_test/` if useful.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
