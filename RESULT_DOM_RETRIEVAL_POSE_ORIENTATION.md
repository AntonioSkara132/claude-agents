# Result: pose-trajectory campaign, and orientation error is a missing-input bug, not noise

Ran `configs/deformpath_pose_trajectory.yaml` (the checked-in recipe: history-full, 1024pts,
K=100, 1000 epochs, joint+contrastive, `target_representation: xyz_quaternion`) on real data.
Also independently verified the new pose support (separate fork, before this run): 83/83 tests
pass, quaternion data confirmed real/non-degenerate, the "exact zero orientation error" commit
confirmed benign (redundant clamp removal, not a leak).

## Campaign results (3/4 methods; `transformer_residual` still running)

| Method | Position (mm) | Orientation (deg) |
|---|---:|---:|
| direct | 26.85 | 9.50 |
| cartesian_residual | 27.53 | 9.35 |
| latent_residual | **25.53** | 9.32 |

Position numbers are in line with the rest of the investigation. Orientation is flat across all
three methods (9.3-9.5°) regardless of architecture — that flatness turned out to be the
interesting finding.

## Context check: is 9.3-9.5° actually good or bad?

Computed directly on the real test set (119 motions): true trajectories barely rotate at all —
mean total rotation start-to-end is **7.74°** (median 6.62°, p90 13.95°). A trivial **do-nothing
baseline** (hold the starting orientation constant, predict zero rotation) already scores
**8.81°** average error. So none of the three methods above are meaningfully better than doing
nothing, and are all somewhat worse.

## Diagnosed why: not a magnitude problem, a missing-input problem

Checked whether predicted rotation was large-and-wrong or small-but-wrong-direction:

| Method | Predicted rotation magnitude | True magnitude | Error |
|---|---:|---:|---:|
| direct | 5.99° | 7.74° | 9.50° |
| cartesian_residual | 5.23° | 7.74° | 9.35° |

Predicted magnitude is in the right ballpark (not runaway) — the models correctly learned
"rotations are small" as a population prior. The error exceeding either magnitude means the
predicted *direction/axis* doesn't track the true one per-query. Root cause, confirmed directly in
code: **the history/proprioception input pipeline never carries orientation at all** in this
config. `observations.py::validate_history` hard-requires `history_tool_positions.shape[-1] == 6`;
`models/history.py::HistoryEncoder.tool_features` is `nn.Linear(8, hidden_dim)` (6 position dims +
time + elapsed, hardcoded); the dataset loader builds history from
`record["path"][:,:,:3]` (position-only) regardless of `target_representation`. So the model has
zero signal about current/historical tool orientation to condition its prediction on — it can only
guess a plausible-magnitude, arbitrary-direction rotation per query, which is exactly consistent
with the error pattern measured.

Also checked the residual mechanism specifically (in case the bug was there instead/also): it's
implemented correctly. `RetrievalPolicy`'s `reference = weighted_pose_mean(...)` does
hemisphere-aware quaternion averaging (not naive, which would be wrong), the residual network's
`action_dim` covers all 14 pose channels, and `prediction = project_pose_trajectory(reference +
delta)` renormalizes afterward. Not the bug.

## Suggested fix (implemented + smoke-tested in an isolated checkout, not committed)

Extend `history_tool_positions` to carry the full 14-dim pose (matching what
`target_representation: xyz_quaternion` already produces for the trajectory target) instead of
6-dim XYZ-only, mirroring the pattern already used for the static `ProprioceptiveEncoder`'s
`pose_dim`. Five small, backward-compatible changes (default `pose_dim=6` preserves current
behavior for non-pose configs; 83/83 tests still pass with the change):

```diff
--- a/data/dataset.py
@@ load_deformpath
-            history_positions = record["path"][:, :, :3].reshape(-1, 6).float()
+            history_positions = record["path"][:, :, :channels].reshape(-1, channels * 2).float()

--- a/data/history.py
@@ retained_history
-    history_tools = torch.zeros(length, 6)
+    history_tools = torch.zeros(length, positions.shape[-1])

--- a/observations.py
@@ validate_history
-    if tools.shape != (b, h, 6) or times.shape != (b, h) or mask.shape != (b, h):
-        raise ValueError("History tools, times and mask must have dimensions [B,H,6], [B,H], [B,H]")
+    if tools.ndim != 3 or tools.shape[:2] != (b, h) or tools.shape[2] not in (6, 14) or times.shape != (b, h) or mask.shape != (b, h):
+        raise ValueError("History tools, times and mask must have dimensions [B,H,6 or 14], [B,H], [B,H]")

--- a/models/history.py
@@ HistoryEncoder.__init__
-    def __init__(self, frame_encoder, embedding_dim=32, hidden_dim=64, mode="full") -> None:
+    def __init__(self, frame_encoder, embedding_dim=32, hidden_dim=64, mode="full", pose_dim: int = 6) -> None:
         ...
+        if pose_dim not in (6, 14):
+            raise ValueError("History tool-state dimension must be 6 or 14")
+        self.pose_dim = pose_dim
-        self.tool_features = nn.Sequential(nn.Linear(8, hidden_dim), nn.ReLU())
+        self.tool_features = nn.Sequential(nn.Linear(pose_dim + 2, hidden_dim), nn.ReLU())

--- a/training/trainer.py
@@ make_encoder
-        return HistoryEncoder(frame_encoder, ..., history.get("mode", "full"))
+        return HistoryEncoder(frame_encoder, ..., history.get("mode", "full"),
+                              pose_dim=int(split.train[0].history_tool_positions.shape[-1]))
```

### Smoke test result (200 epochs, direct + cartesian_residual, not directly comparable to the 1000-epoch campaign)

| Method | Without fix | With fix |
|---|---:|---:|
| direct | 9.50° | 9.14° (small real improvement) |
| cartesian_residual | 9.35° | 9.39° (no meaningful change) |

**Honest read: the fix helps a little for `direct`, not at all for `cartesian_residual`, and even
the improved number is still barely under the 8.81° do-nothing baseline.** This is not a full
fix on its own — either the 200-epoch budget is too short to exploit the new signal (untested at
the full 1000-epoch budget), or there's a second limiting factor beyond input availability (e.g.
the small 2-layer `tool_features` MLP may not be expressive enough to extract useful rotation
dynamics from raw concatenated position+quaternion history). Reporting this as a real, verified,
but incomplete finding rather than a solved problem.

## Suggested next steps

1. Decide whether to adopt the diff above into the real `dom_retrieval` repo (I only tested it in
   an isolated checkout, not committed).
2. If adopted, a full matched 1000-epoch run (not a 200-epoch smoke test) is needed before drawing
   a real conclusion about whether it closes the gap.
3. `transformer_residual` (4th campaign method) still running; will report separately once done.

Config, smoke-test logs and the isolated checkout's diff at `/workspace/runs/orientation_fix/` and
`/workspace/runs/deformpath_pose_trajectory_seed7/` on this host.
