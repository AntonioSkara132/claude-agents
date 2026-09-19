# Instructions: build the policy-export adapter (handoff, not for the CUDA host to implement)

Operator's call: I (CUDA host) do the environment/data/calibration groundwork and verification,
you implement new capabilities in the repos — matching the pattern for every `dom_retrieval`
feature this thread (proprioception, history, pose, dough-frame). This adapter should be your
implementation, not mine. Everything below is the groundwork already done so you don't have to
redo it.

## Status: baseline simulation + rendering both confirmed working

See `RESULT_TAICHIDOUGH_BASELINE_SIMULATION.md` and `RESULT_TAICHIDOUGH_RENDER_FIX.md`. Recorded-
action replay for episode18 (frames 1-59) completes cleanly (`simulation_result.json`,
`completed_steps: 9839, failure: null`) using the operator-vouched material parameters
(E=17072.497582525557 Pa, ν=0.489644253859494, viscosity=0.33672455451768313 Pa·s, plastic
0.7277641506821126-1.091951442087265) passed as explicit CLI flags. Rendering works via
`xvfb-run` + direct GIF assembly (this host has no GPU EGL/working GLX and no `libx264`).

## Coordinate frame: verified, no unknown transform needed

Checked this carefully before handing off, since it was the biggest risk. The exact
hash-verified calibration episode18's `sequence_metadata.json` references
(`DeformPath3/snimanje_23_10/episode18_kugla/scene_calibration_v2.json`, sha256
`d23150aa6fe54d0ca3c48d2c2c0eb6bb7b0e91f7e06be7176fbd5aedc487c68b`, matches exactly) has:

```
source_frame: mocap, scene_frame: mocap
scene_from_source: identity matrix
floor_plane_scene: [0, 1, 0, 0]
```

`scene_from_source = identity` — `dom_retrieval`'s raw tool XYZ+quaternion (straight from
`record["path"][...,:7]`, exactly what the model trains on and predicts) is already in the
simulator's scene frame. No rotation/translation to derive for tool marker poses. The only
remaining offset is `marker_from_tool_frame` (marker → tool geometry), which is already
calibrated and already used identically by the recorded-action baseline — the adapter just needs
to reuse it, not derive anything new.

One thing to flag, not blocking: `preprocessed_dataset/episode18_kugla`'s own
`scene_calibration_v2.json` has a different `floor_plane_scene` offset (−0.0368 vs. 0.0 above) —
a different, independently-recomputed calibration run, not an interchangeable copy of the
authoritative one above. Use the `DeformPath3`-referenced file, not the `preprocessed_dataset` one,
for anything calibration-sensitive.

## Checkpoint to use

Per the original request: **Group B full-history Cartesian residual, 24.38mm** (position-only,
`target_representation: xyz`, not the new quaternion/pose model — that one's orientation output is
still undertrained per `RESULT_DOM_RETRIEVAL_POSE_ORIENTATION.md`, not ready for this).

```python
import json
from pathlib import Path
manifest = json.loads(Path('/workspace/runs/open_loop_groupB_full_cartesian_residual/manifest.json').read_text())
checkpoint = Path(manifest['checkpoint'])
```
Resolves to `/workspace/runs/history_deformpath_full_1000_seed7/seed_7/full/cartesian_residual/checkpoint.pt`
with its `dataset.pt` snapshot alongside it (`checkpoint.parent.parent / "dataset.pt"`).

## Still needed (the actual adapter work)

Everything from the "Policy export requirements" section of the original
`REQUEST_DOM_RETRIEVAL_RECONSTRUCTED_MPM.md` still applies as written — not re-copying it all
here, just flagging what's NOT yet done:

1. **Segment selection** matching the reconstruction's timeline (episode18, frames 1-59). Match by
   actual retained frame indices/timestamps, not annotation ordinal as an array index. Episode18
   is in the policy's *training* partition — any run here is an integration/calibration
   diagnostic, not held-out accuracy, exactly as the original request already flagged.
2. **The adapter script itself**: `load_experiment`/`batch_data`/`physical_output` in eval mode,
   predict once, `[T,6]` → `[T,2,3]` (UR5e then Gen3), map predicted phase (0→1) to real time via
   the segment's actual `start_time_s`/`duration_s` (not predicted), interpolate to simulator
   control times, apply the (now-confirmed-identity, but still explicit — don't hardcode away the
   step) `scene_from_source` transform and the existing `marker_from_tool_frame` offset.
3. **The 5 paired conditions** from the original request (recorded full-pose, recorded
   XYZ+fixed-orientation, predicted XYZ+fixed-orientation, hold-position, optional
   predicted-XYZ+recorded-orientation).

## Environment notes for whoever runs this

If run on this host: working Taichi runtime is a patched interpreter at
`/root/.claude/jobs/c912ad55/tmp/py310runtime/python3.10` (not the documented GLIBC sysroot
loader, which fails here) — needs `LD_LIBRARY_PATH` set at its own process startup but NOT leaked
into subprocess calls (`render_python`, the spawned `--simulation-python` re-invocation). Working
wrapper scripts already exist at `/workspace/runs/sim_launcher.py` and
`/workspace/runs/sim_python_wrapper.sh` if reused. Rendering needs `xvfb-run` wrapping the render
Python, and video encoding needs direct GIF assembly (`ffmpeg -i frame_%05d.png output.gif`), not
the tool's own MP4 path (`-crf` isn't supported by this host's `libopenh264`-only ffmpeg build).

Report back here when the adapter's ready to run; happy to run it on this host if that's easier
than you running it yourself.
