# Result: corrected Episode18 v3 three-chain simulation, 2mm tool offset

This request arrived after I had already run this exact configuration once (the user directed me
to act ahead of your "wait for the follow-up" instruction on REQUEST_EPISODE18_CHAIN_V3_TOOL_OFFSET.md).
Verified the script hash matches your stated expectation exactly, so I ran your verification script
against the existing output rather than re-running (per "do not overwrite an existing run").

## Source verification

```
sha256sum experiments/differentiable_mpm/episode18_chained_comparison.py
f8e9e09df8334ed100e170a1ade8f2b7354365a6d952a7f26eb5e151926cafc7  (matches expected exactly)
```

## Run

```
Output:  /workspace/TaichiDough/experiments/differentiable_mpm/runs/episode18_chained_v3_tooloffset2mm_20260915T160026Z
Archive: /workspace/TaichiDough/experiments/differentiable_mpm/runs/episode18_chained_v3_tooloffset2mm_20260915T160026Z_simulation_states.tar.gz
Bytes:   44252816
SHA256:  d7e0d21e29c6bb818ba3d0c2f443145a98c535e265c33306a712447f959cab52
Python:  3.11.13, Taichi 1.7.4, confirmed "[Taichi] Starting on arch=cuda", CUDA device: NVIDIA GeForce RTX 4090
```

`--stage simulate` ran with no bag/conversion-metadata arguments, as required.

## Manifest checks (all passed)

- `schema == "taichidough/episode18-chained-comparison/v3/run"` ✓
- `stages.simulate.status == "completed"` ✓
- `calibration_run is False` ✓
- `tool_pose_offset_m == {"axis": "scene_y", "value": -0.002, "applied_to_both_tools": true}` at both
  root manifest and every chain manifest ✓
- Chain chunk sequences match exactly: `chain_a` [1,2,3,4], `independent_from_chunk2` [2,3,4],
  `from_chain_a_chunk1_end` [2,3,4] ✓
- Every `initial_state.npz`/`terminal_state.npz` has exactly fields `{x,v,C,F,Jp}`, all finite ✓
- Every particle snapshot shape `(24000, 3)`, all finite ✓
- `source_identity.unchanged == True` (tracked Python source did not change during the run) ✓

## Snapshot bounds (all finite, compact — no domain-filling instability)

```
chain_a 1 start:  min [0.4140, 0.0000, 0.4080]  max [0.5730, 0.0250, 0.5220]
chain_a 1 end:    min [0.4090, 0.0002, 0.4040]  max [0.5796, 0.0453, 0.5201]
chain_a 2 end:    min [0.4178, 0.0010, 0.4192]  max [0.5639, 0.0539, 0.5416]
chain_a 3 end:    min [0.4064, 0.0003, 0.4378]  max [0.5514, 0.0551, 0.5506]
chain_a 4 end:    min [0.4093, 0.0023, 0.4520]  max [0.5468, 0.0661, 0.5703]

independent_from_chunk2 2 start: min [0.4200, 0.0000, 0.3901]  max [0.5610, 0.0271, 0.5220]
independent_from_chunk2 4 end:   min [0.4253, 0.0045, 0.4553]  max [0.5402, 0.0720, 0.5985]

from_chain_a_chunk1_end 2 start: min [0.4090, 0.0002, 0.4040]  max [0.5796, 0.0453, 0.5201]
from_chain_a_chunk1_end 4 end:   min [0.4093, 0.0023, 0.4520]  max [0.5468, 0.0661, 0.5703]
```

## Replay comparison — tolerance check FAILS

```
replay all_exact: False
replay all_within_tolerance: False   <-- fails your required assertion
```

Per-field differences between `chain_a` and `from_chain_a_chunk1_end` (which should replay
identically from Chain A's exact C1 terminal state), growing through the chain:

```
chunk2 end:  x max_abs=9.89e-05  v max_abs=0.0349  C max_abs=0.00486  F max_abs=0.00129  Jp exact
chunk3 end:  x max_abs=5.39e-04  v max_abs=0.0521  C max_abs=0.02457  F max_abs=0.00896  Jp exact
chunk4 end:  x max_abs=5.59e-04  v max_abs=0.0765  C max_abs=0.02519  F max_abs=0.00818  Jp exact
```

(rmse values are much smaller than max_abs in every case — e.g. chunk4 end x: rmse=9.99e-06 vs
max_abs=5.59e-04 — so this is a small number of particles with larger deviation, not a uniform
shift. `Jp` is bit-exact throughout.)

This matches your own caveat that CUDA atomic P2G can prevent bitwise equality, but the divergence
here exceeds the script's tolerance (`REPLAY_ATOL=REPLAY_RTOL=1e-5`) — velocity differences reach
~0.077 m/s and position differences ~0.56mm by chunk4, both several orders of magnitude above that
tolerance and growing chunk over chunk. This looks like accumulating CUDA nondeterminism (atomic
add ordering in P2G) rather than a bug in the replay/state-transfer logic itself, since `x`, `C`,
`F` are bit-identical at each chunk's start (only `end` states diverge, then get carried forward
as the next chunk's start — consistent with per-step floating-point nondeterminism compounding).

Not making any change to the tolerance values or the comparison logic — flagging for you to decide
whether `1e-5` is the right bar given CUDA's atomic execution model, or whether determinism should
be enforced some other way (e.g. deterministic atomics, CPU backend for this check, or a looser
tolerance).

## Prior runs/archives

Not modified — this run reused the existing output from earlier in the session, nothing deleted or
overwritten.
