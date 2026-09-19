# Result: pose-history fix tested at full 1000-epoch budget — doesn't help cartesian_residual

Follow-up to `RESULT_DOM_RETRIEVAL_POSE_ORIENTATION.md`. You committed the suggested fix
(`1834ca3`, "Feed pose history to orientation policies") — matches the proposed diff essentially
verbatim (same 5 files, same logic, minor formatting differences). 83/83 tests still pass. Ran the
full matched 1000-epoch `cartesian_residual` comparison this asked for, not just the earlier
200-epoch smoke test.

## Complete campaign table (all 4 methods, unfixed, for context) + the fixed run

| Method | Position (mm) | Orientation (deg) |
|---|---:|---:|
| direct (unfixed) | 26.85 | 9.50 |
| cartesian_residual (unfixed) | 27.53 | 9.35 |
| latent_residual (unfixed) | 25.53 | 9.32 |
| transformer_residual (unfixed) | 26.55 | 9.28 |
| **cartesian_residual (fixed, 1000ep)** | **28.51** | **9.28** |

## Honest read

At the full, properly matched 1000-epoch budget (not the earlier 200-epoch smoke test), the fix:

- **Doesn't help position** — 28.51mm vs. 27.53mm unfixed, actually slightly worse.
- **Barely moves orientation** — 9.35°→9.28°, within noise, and still above the 8.81°
  do-nothing baseline established in the earlier report (mean true rotation is only 7.74°).

This isn't the fix earning its keep for `cartesian_residual` specifically. Two ways to read it:

1. **The fix is correct but this method doesn't benefit from it.** `cartesian_residual` mixes a
   Top-K retrieved reference (via `weighted_pose_mean`, hemisphere-aware quaternion averaging)
   plus a residual correction. If the retrieval keys/encoder embedding aren't shaped around the
   *new* orientation-aware history signal (still trained end-to-end from scratch, same as
   position was originally), the extra input may just add noise to an already-noisy retrieval
   selection rather than sharpening it. Worth comparing against `direct` (simplest, no retrieval)
   at the same full budget — the smoke test's only positive signal (9.50→9.14°) was on `direct`,
   not `cartesian_residual`.
2. **There's a second limiting factor beyond input availability**, as already flagged: the
   `tool_features` branch is a single 2-layer MLP (`Linear(pose_dim+2, hidden) → ReLU`) processing
   raw concatenated position+quaternion+time — might not be expressive enough to extract useful
   rotation-direction signal from that representation, independent of which policy method
   consumes it.

## Suggested next step

Run `direct` at the full 1000-epoch budget with the fix (only tested at 200 epochs so far) before
concluding either way — it's the one method that showed a real, if small, improvement, and it's
the simplest architecture to rule in/out as the source of the effect. If `direct` also fails to
clear the do-nothing baseline at full budget, the input-completeness fix on its own likely isn't
sufficient and the `tool_features` architecture itself is the next thing to look at.

Full logs at `/workspace/runs/deformpath_pose_trajectory_fixed_cartesian_residual_seed7/` on this
host.
