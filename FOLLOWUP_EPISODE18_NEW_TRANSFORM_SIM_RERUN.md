# Follow-up: MPM re-run with the new validated transform, both completed

Follow-up to `RESULT_EPISODE18_NEW_TRANSFORM_VALIDATED.md`. Re-ran the
`predicted_xyz_recorded_orientation` MPM simulation with `T_M_DO_new` in place of the
old diagnostic `T_M_DO`, for both checkpoints on this host: the original
`cartesian_residual` checkpoint (old dataset, 25.25mm/6.51deg) and the new
`cartesian_residual` checkpoint trained on `Deformapth2_camera1024_interpolated_nodbscan`
(30.31mm/6.02deg, see `RESULT_DOM_RETRIEVAL_ABLATION_MATRIX.md` context for the dataset
audit that preceded training it). Same episode18 segment, same physics parameters
(youngs_modulus 17072.5, poisson_ratio 0.4896, viscosity 0.3367, plastic_min 0.7278,
plastic_max 1.0920), same endpoint (frame 43), same archive/export pipeline as before,
only the transform changed.

Both completed cleanly:

```text
old-dataset checkpoint:  completed_steps 7171/7171, sim_time_s 1.4342, failure null
new-dataset checkpoint:  completed_steps 7171/7171, sim_time_s 1.4342, failure null
```

Output directories:

```text
/workspace/TaichiDough/experiments/differentiable_mpm/runs/corrected_olddata_predicted
/workspace/TaichiDough/experiments/differentiable_mpm/runs/corrected_newdata_predicted
```

Rendering hit the same known non-fatal ffmpeg `-crf` limitation on this host as every
prior run (`Unrecognized option 'crf'`, only `libopenh264` available) — 17 preview PNGs
rendered fine in both cases, assembled into GIFs directly via `ffmpeg -framerate 6
-i frame_%05d.png output.gif`, same workaround as always.

Both GIFs, plus a point-cloud overlay/residual-histogram figure for the new transform,
are on the hosted page (private artifact, link available on request from the operator
side since this repo doesn't carry hosted-page links).

Given the point-cloud validation already reported (median 1.1mm, p90 1.6mm, only the
known 10-point sensor-artifact cluster as outliers), these two runs are the first
predicted-vs-recorded MPM comparisons on this episode that aren't plumbing-only tests.
Haven't done a quantitative predicted-vs-recorded physics comparison (e.g. depth/coverage
loss) yet — happy to if there's a specific metric you want, otherwise treating this as
"the pipeline now produces a meaningfully-placed simulation" rather than a policy-quality
verdict.

One open item carried over: this transform's own fit provenance (data it was fit on,
sample count, residual stats on its own target) hasn't been confirmed on this end since
it arrived as a matrix rather than a documented derivation. The point-cloud check is
strong evidence it's real, but flagging this the same way the earlier diagnostic fit's
limitations were flagged, for the same reason.
