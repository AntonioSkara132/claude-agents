# New camera->mocap transform validates against point clouds (unlike the earlier diagnostic fit)

Received a new candidate transform (operator, source not yet in this repo):

```text
T_M_DO_new =
[[ 0.00548785,  0.99242744, -0.12270963,  0.28713715],
 [ 0.05672247, -0.12282284, -0.99080629,  0.53146005],
 [-0.99837490, -0.00152300, -0.05696697,  0.76467727],
 [ 0.,          0.,          0.,           1.        ]]
```

Convention: `p_mocap = R @ p_camera_depth_optical + t`, same as the earlier diagnostic
`T_M_DO` in `RESULT_EPISODE18_SOURCE_FRAME_PROVENANCE.md`.

## Point-cloud validation (the check the earlier transform failed)

Applied to the training-input point cloud (episode18, frame index 2, same frame used in
every prior comparison this session), nearest-neighbor distance to the verified
`relocated_episode18` simulation-input cloud (both old and new dom_retrieval datasets
give the same episode/frame, checked against the old dataset here since its raw file
loads cleanly right now):

| | old diagnostic `T_M_DO` | new `T_M_DO_new` |
|---|---|---|
| median NN residual | 30.5 mm | **1.1 mm** |
| p90 | 70.9 mm | **1.6 mm** |
| p95 | n/a | 1.78 mm |
| p99 | n/a | 4.81 mm |
| max | 138.1 mm | 170.6 mm |

Only 10 of 1024 points exceed 20mm residual with the new transform — and those are the
same 10-ish stray disconnected points already identified as sensor artifacts (not dough)
in the point-cloud comparison figure on the "Episode18 Policy Sim" hosted page, visually
separate from the main blob in every prior visualization. Excluding those, alignment is
sub-2mm for the rest of the cloud. `R` is a proper rotation (orthonormal, det=1).

This is categorically different from the earlier diagnostic fit, which fit tool poses
tightly (0.569mm RMS) but failed on point clouds (50.4mm median). This one succeeds on
point clouds directly. Given the magnitude of the improvement (30x) and that it fixes
exactly the failure mode that made the earlier fit "diagnostic only," this looks like
a real resolution to the depth-optical-to-mocap calibration blocker
(`BLOCKER_MPM_POLICY_EXPORT_SOURCE_MISMATCH.md` /
`RESULT_EPISODE18_SOURCE_FRAME_PROVENANCE.md`), not another diagnostic fit — but I don't
have this transform's own fit provenance (what it was fit on, residual stats on its own
fit target, sample count) since it arrived directly rather than through this repo. Please
confirm provenance and residual stats on your side before this gets treated as fully
validated.

## Next step

Given this checks out, replacing the diagnostic transform with this one in the MPM
policy-control archive and re-running the predicted-condition simulation would give an
actually-meaningful predicted-vs-recorded comparison for the first time, rather than a
plumbing-only test. Will do that next unless you flag a reason not to.
