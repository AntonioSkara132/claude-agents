# Result: episode18 baseline recorded-action simulation succeeded

Follow-up to `CORRECTION_TAICHIDOUGH_EPISODE18_FINGERPRINT.md`. The blocker is resolved — with a
real fix, not a bypass.

## Found the correct data

Operator pointed to `/workspace/data/relocated_episode18` (the exact path referenced as
`relocated_episode` in `episode18_registered_tools_v1.verification.json`, which I'd noted early on
but hadn't followed up). Computed its sequence fingerprint directly:

```
sha256(sha256(pointclouds_interpolated.pt) + sha256(paths_interpolated.pt))
= b5b2225335813b6c61f1777194c92cfa5de93b082dcf1a4974cd9cf992b825cb
```

**Exact match** against `episode18.json`'s `expected_sequence_fingerprint`. This is the genuine
source data the calibration was built from.

## Baseline simulation: completed

```bash
cd /workspace/TaichiDough
export EPISODE18=/workspace/data/relocated_episode18
# (same command as REQUEST_TAICHIDOUGH_EPISODE18_DATA_BLOCKER.md, using the unmodified
#  dataset_episode18_only.json / episode18.json — no fingerprint override needed)
```

```json
{"status": "completed", "completed_steps": 9839, "sim_time_s": 1.9678, "last_saved_source_frame": 59, "failure": null}
```

Full 60-frame recorded-action replay, no failures, ~59 steps/s once warmed up (CUDA, f32).
Output at
`/workspace/TaichiDough/experiments/differentiable_mpm/runs/dom_policy_identified_20260919T100203Z_recorded/`
on this host (`simulation/simulation_result.json`, `last_valid_particles.npy`, `snapshots/`).

## Separate, unrelated failure: rendering

Video rendering failed after the simulation succeeded, but for a purely environmental reason —
no working GPU OpenGL driver for `pyvista` in this sandbox:

```
libGL error: failed to create dri screen
libGL error: failed to load driver: nouveau
libGL error: No matching fbConfigs or visuals found
libGL error: failed to load driver: swrast
X Error of failed request: GLXBadContext
```

Simulation states are saved to disk regardless (`recover.py` is suggested by the tool's own error
message for rendering from saved state later). Not a blocker for the numeric/physics work; will
revisit if a rendered preview is needed.

## Next

Baseline recorded-action replay is validated. Ready to move on to the actual policy-export
adapter and the paired-conditions comparison (recorded vs. predicted-XYZ vs. hold-position) from
the original request — will scope and report that as a separate step rather than combining it
with this baseline confirmation.
