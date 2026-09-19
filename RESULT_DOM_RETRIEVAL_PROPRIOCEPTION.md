# Result: proprioception input fixes most of the position bias

In reply to `INSTRUCTIONS_PROPRIOCEPTION.md`. Thanks for pushing the actual implementation to
`dom_retrieval` (commit `993e5b2`) — the first `INSTRUCTIONS_PROPRIOCEPTION.md` referenced
`include_tool_positions`/`proprioception.enabled`/`configs/deformpath_proprioception.yaml` before
that commit existed on this host; verified via grep + `load_deformpath`'s signature that none of
it was present yet, held off running it, then pulled again once told to and found the real
commit. Reviewed the diff (`ProprioceptiveEncoder` in `models/perception.py`,
`include_tool_positions` in `data/dataset.py`, memory-bank wiring in `models/retrieval.py`,
`observations.py`) and ran the full test suite first: **71/71 pass**, including the new
`test_open_loop`/proprioception-adjacent tests.

## Run

Used `configs/deformpath_proprioception.yaml` as given, only pointing `root`/`source_root` at
this host's absolute paths (`Deformapth2_predownsampled1024_interpolated` +
extracted metadata tree from `deformpath2_source_metadata.tar.gz`), `device: cuda`. Same
611-motion / 451-41-119 split as every other real-data result in this thread.

```bash
python -m dom_retrieval.scripts.train \
  --config <adapted deformpath_proprioception.yaml> \
  --output /workspace/runs/deformpath_proprioception_joint200
```

## Results vs. the best non-proprioception config (1024pts, K=100, same joint+contrastive setup)

| Method | Without proprioception | **With proprioception** | Change |
|---|---:|---:|---:|
| Direct | 30.99mm | 29.89mm | −3.5% |
| Cartesian + residual | 30.69mm | **27.91mm** | **−9.1%** |
| Latent + residual | 30.52mm | 29.00mm | −5.0% |

`cartesian_residual` is now the best result across the entire investigation (27.91mm,
0.0004031 m² test MSE), beating the prior best (`latent_residual` @ 30.52mm) by ~9%. Every method
improved, consistent with the position-bias diagnosis: giving the model the actual measured
tool-position-at-segment-start removes the need to guess it from object geometry alone, which is
exactly what caused the `episode61_kugla` mid-episode outlier reported in
`RESULT_DOM_RETRIEVAL_POSITION_BIAS.md` (85mm/58mm centroid offsets exceeding the trajectories'
own motion range).

## Still running

A second job (`cartesian`, `latent_residual`, same proprioception config, 200 epochs) is running
now to fill out the method comparison further; will follow up with those numbers. Have not yet
re-run the open-loop visualizer on the `episode61_kugla`-style mid-episode segment specifically to
directly confirm the position-bias metric closed — planned as the next check.

Full artifacts at `/workspace/runs/deformpath_proprioception_joint200/` on this host only.
