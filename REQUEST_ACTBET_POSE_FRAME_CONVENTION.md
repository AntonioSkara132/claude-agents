# What frame are DeformPath2_interpolated's tool poses in?

Ran an ACT/BeT checkpoint from `/workspace/DeformPath` (the ACT-chunked BeT repo, separate
from dom_retrieval) through the MPM simulator for the first time this session, on full
episode18. The model itself works; wiring its output into the simulator surfaced a real
frame question I can't resolve without more information about how this repo exports poses.

## What ran

- Checkpoint: `outputs/sweeps/act_bet_transformers_checkpoints/act_bet_chunk30/best_model.pth`
  (`bet_act_chunk`, chunk_size 30, PointMAE shape encoder, best local val_action_l1 among
  the `act_bet_transformers` / `act_bet_chunk40_xy_aug_ablation` sweeps).
- Closed-loop eval via `src/evaluate_act_chunked_closed_loop.py` on
  `/workspace/data/DeformPath2_interpolated/snimanje_23_10/episode18_kugla`
  (243 frames total, ~8.03s at the assumed 30Hz): completed cleanly, **mean executed L1 =
  0.0271**, consistent with training-time validation loss (0.0257). No problem here.
- Fed the stitched 242-step predicted+recorded trajectory into
  `episode18_table_aligned_registered_tools.json` (the same registered-tool SDF config
  every full-episode sim on the hosted report uses), condition
  `predicted_xyz_recorded_orientation`. Simulation itself completed cleanly: 39,858/39,858
  steps, no failure.

## The actual problem: tools never touch the dough

Checked frame 0, frame 119 (mid-rollout, t=3.97s) and frame 238 (last frame, t=7.97s): the
dough sits completely undeformed the entire time. The two tools stay a visually consistent
150-300mm away from the dough blob at every checked timestamp — not closing the gap, not
drifting, just offset. That pattern (constant offset, not growing/shrinking) reads as a
missing or wrong transform, not a timing misalignment or a model-quality problem.

## What I checked before concluding this

1. **Point cloud frame: validated, 9.4mm median / 17.5mm p90.** Took
   `DeformPath2_interpolated/.../pointclouds_interpolated.pt`'s frame-0 observation cloud,
   applied the same `T_M_DO_new` (dom_retrieval camera->mocap, point-cloud-validated
   earlier this session) then `episode18_registered_tools_v1`'s own
   `scene_from_source` (mocap->table-aligned), and compared against
   `episode18_registered_tools_v1/sampled_particles_xyz.npy` (the simulator's own dough
   reconstruction) via nearest-neighbor residual. Good match — same physical scene,
   consistent camera calibration.
2. **Tool poses, same treatment: visibly wrong.** Took the recorded (ground-truth) tool
   poses straight from `paths_interpolated.pt` (no `T_M_DO_new` — poses aren't
   camera-derived, so that transform shouldn't apply to them), applied only
   `scene_from_source`, and compared the frame-0 UR5e/gen3 positions against the dough's
   reconstructed bounding box. UR5e came out roughly plausible; gen3 came out ~500mm off
   in z. Full sim confirms the practical result: no contact for the whole 8s.
3. Cross-checked frame-0 poses against a **trusted** pipeline
   (`episode18_kugla_relocated/chunks/chunk01/paths_interpolated.pt`, already validated
   this session) for the same nominal instant — UR5e was in the same ballpark, gen3 wasn't,
   which could be a genuine timing offset between two differently-trimmed episode exports
   (243 vs 388 frames) rather than proof of a frame bug on its own — but combined with (2)
   and the full-episode no-contact result, the simplest explanation left is that
   `DeformPath2_interpolated`'s tool poses use a different frame than its own point clouds,
   or a different frame than `episode18_registered_tools_v1`'s calibration expects.

## What I need

Is there a documented transform from `DeformPath2_interpolated`'s exported tool poses into
TaichiDough's mocap frame (the same relationship `T_M_DO_new` describes for the point
clouds)? Or is there a different calibration file this pipeline's poses are meant to be
loaded through that I haven't found? I didn't want to guess a second transform by trial and
error on top of the calibration bug from earlier today
(`CORRECTION_CHUNK07_CALIBRATION_ROOT_CAUSE.md`) — happy to try a specific transform if you
can point me at where it's defined, or to re-derive it myself with guidance on what raw
inputs to check.

Everything needed to reproduce is under `/workspace/runs/chunk07_relocated_test/actbet/`
(stitched predictions, recorded poses, control archive, sim output) if useful.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
