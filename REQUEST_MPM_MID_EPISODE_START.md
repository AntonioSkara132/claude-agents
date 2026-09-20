# Request: a way to start the predicted-condition MPM sim mid-episode

Operator wants the next predicted-condition simulation to start from somewhere in the
middle of episode18 rather than the very first segment (everything run so far has used
`segment-2d750affdcc9d983-pc000000-pc000046`, start_time_s 0.07).

## What I checked before asking

- `run.py` has no `--start-frame` (or equivalent) argument — `raw["training"]["start_frame"]`
  is hardcoded to `1` in `forward_video_v2/run.py:260`. The simulation always initializes
  particles from the episode's frame-0 reconstruction and steps forward to `--end-frame`.
- `episode18_chained_comparison.py` does carry full particle state across a control switch
  (`run_chain`, `Stepper.load_state`/`.state(slot)`), which is the right primitive for
  this, but it's built specifically around your existing fixed 4-chunk breakdown of the
  episode for the article comparison (`prepared[start_index:4]`, hardcoded
  `0 <= start_index < 4`) — not something I can safely point at an arbitrary mid-episode
  start with a new controls-archive condition without risking getting the SDF/particle
  setup wrong. Didn't want to reverse-engineer and repurpose that internals myself given
  the earlier instruction to leave simulator-adapter work to you.

## What's needed

Some way to run `predicted_xyz_recorded_orientation` (or any condition) starting from a
mid-episode frame with the dough already in the correct deformed state at that point.
Two options that seem reasonable, your call which fits the existing pipeline better:

1. Warm-up mode: replay `recorded_full_pose` controls from frame 0 up to a chosen start
   frame (carrying full particle state, like `run_chain` already does), then switch to
   the requested condition's controls for the remainder, controlled by something like
   `--warm-up-condition recorded_full_pose --start-frame N --end-frame M`.
2. Direct state initialization: load a pre-saved particle state (e.g. from an existing
   chunk's `terminal_state.npz`, if one already lands near the frame we want) as the sim's
   initial state instead of the frame-0 reconstruction.

## For reference: candidate mid-episode segment

If it's useful for picking a frame, the exported policy segments for episode18 (same
seed-27 `deep_selection_scorer` checkpoint used in the last MPM report) include, e.g.:

```text
segment-2d750affdcc9d983-pc000193-pc000210, start_time_s=6.44, end_time_s=6.97 (17 samples)
segment-2d750affdcc9d983-pc000210-pc000230, start_time_s=7.00, end_time_s=7.67 (20 samples)
```

roughly the midpoint of the ~13s / 388-frame episode (compare to the first segment's
0.07-1.50s used in every prior run). Happy to export the exact predicted trajectory for
whichever start frame ends up supported once there's a way to warm the sim up to it.
