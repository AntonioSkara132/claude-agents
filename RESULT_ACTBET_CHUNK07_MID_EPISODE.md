# ACT/BeT also works mid-episode: chunk07, same frame-index trick as dom_retrieval

Follow-up to `RESULT_ACTBET_POSE_FRAME_SOLVED.md`. Ran the same `act_bet_chunk30`
checkpoint starting mid-episode at chunk07, not just full-episode from frame 0.

## How the frame alignment worked out

`DeformPath2_interpolated/.../episode18_kugla/sequence_metadata.json` has a
`pointcloud_indices` array mapping each of its 243 stored samples back to the *same* raw
source-frame numbering used by the TaichiDough chunk pipeline (`sample_dt_sec` is
literally identical to chunk07's own `paths_interpolated.pt` dt, `0.033351898193359375`
both places). Checked it directly: stored positions 123-139 map to raw frames 137-153,
gap-free, exactly chunk07's own `raw_start_frame`/`raw_end_frame_exclusive` (137/154).

Reran `evaluate_act_chunked_closed_loop.py` with `--start-step 123 --max-steps 17
--replan-every 17`, applied the same fix as the full-episode run (`T_M_DO_new` on both
position and orientation of the pose stream, then chunk07's own `scene_from_source`), and
fed it through chunk07's reconstruction.

## Result

Completed cleanly: 2669/2669 steps, no failure, tools aligned with the dough on frame 0
(correct on the first attempt, using the already-solved calibration). mean_executed_l1
0.0179 for this 17-step window (model's own metric; tighter than the full-episode
0.0271, expected for a shorter local prediction). Published on the report's ACT/BeT
section: https://claude.ai/artifact/9vf5L8KX8hCxBDmk59Mdae

## A precise, scoped feature request: decouple `total_steps` from the observation window

User asked to run the chunk07 replay longer than its own ~0.53s (17 recorded frames).
Traced why that's not currently possible even though the physics itself doesn't need it:

```python
# data.py, prepare_experiment()
frames = observation_schedule(sequence.times, sequence.original_indices, selected_end, dt)
total_steps = frames[-1].completed_substeps
```

`total_steps` is hard-derived from the last **recorded observation's** timestamp, and
`selected_end` (`--end-frame`) must satisfy `0 <= end_frame < len(times)` —
`observation_schedule`'s own assertion, `replay.py` line ~36. For chunk07 that caps
`total_steps` at ~2669 steps (0.5338s) no matter how long the controls-archive duration
is (tried 1.5s explicitly — control archive built fine, `total_steps` didn't move).

This path exists to align simulation checkpoints with real observations for **loss
scoring** during calibration — reasonable for that use case. But `forward_video_v2`'s
`run.py`/`forward.py` also use this same path for pure `--condition`-driven forward
replay (`predicted_xyz_recorded_orientation` etc.), where there's no scoring happening
and the physics doesn't need any observation past the first frame to keep stepping.

**Ask**: would a mode where `total_steps` can come from the controls-archive's own
duration (or an explicit `--total-steps`/`--replay-duration` flag) instead of
`frames[-1].completed_substeps`, specifically for condition-driven replay with no loss
scoring, be reasonable to add? Only needed when running a `--condition`, not the
scoring/calibration path — happy to be more specific about the call sites if useful.
Not attempting this myself since it's a change to `data.py`'s core step-count logic.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
