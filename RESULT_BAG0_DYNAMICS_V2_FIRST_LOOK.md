# bag0 dynamics v2: first look on the GPU host (2026-10-25)

Package `bag0_dynamics_v2_gpu_package.zip` installed in an isolated checkout `/workspace/TaichiDough_bag0` at
`origin/bounded-shear-adhesion-v3` = `9f50bf1` (four commits past the package's `dc2cedd`; the package's source
snapshot is identical to the tip for every file that enters the SDF cache identity, so the shipped 256^3 fields
load as-is: `SDF cache hit` for both blades, arrays checksum-verified). Environment notes: Open3D 0.19 was
installed into the py311 env (numpy pinned, nothing else changed) because the cached fields are re-qualified
with it on load; two local, uncommitted patches in the worktree only: `sdf_cache.py` pins the numpy/Open3D
version strings in the cache identity to the ones the cache was built with (`TAICHIDOUGH_SDF_CACHE_NUMPY_VERSION`,
`..._OPEN3D_VERSION`), and `forward_video_v2/run.py` replays at the dataset's shared initial parameters when no
parameter source is given (its explicit flags do not know `tool_adhesion` / `floor_friction_coefficient`).
Scripts and outputs: `/workspace/runs/bag0/`.

## Forward replays (starting parameters: E 20 kPa, nu 0.47, viscosity 1, tool friction 0.6, adhesion 5, floor friction 0.5)

Five chunks rendered in perspective (recorded tool motion, f32, 545 steps/s, 17 s per chunk):
ep03c_squeezes_seg44_c00, ep01_pushes_squeeze_c01, ep03b_spatula_kneading_c00, ep05_left_edge_cross_cut_c00,
ep06_fast_squeezes_c01 -- GIFs on the results page (top section) and in `/workspace/runs/bag0/*_perspective.gif`.
All five simulate without failure; the blades close on the dough as recorded.

## Gradient identifiability (all 34 chunks, f64 adjoint, atomic P2G, at the starting parameters)

`grad_ident.py`: one adjoint per chunk, d(objective)/d(parameter) for the six fitted parameters plus the von
Mises yield stress (differentiable, log-scale parameter in the code; the v2 dataset just does not list it as a
shared fit parameter). Reported as scaled sensitivity s = p * dL/dp (change of the objective per 100 % change
of the parameter). f32 adjoints are rejected by the recompute-consistency check (same as Episode 14); f64 works:
~110 s per chunk, 16 GB GPU memory.

30 of 34 chunks give finite gradients for every parameter. The other four (ep03c c01, c02, c03, c04) cannot be
scored at the start point: the initial-referenced depth term has zero pixel support ("Fixed initial-referenced
depth support 0 is below 1"), the same loss limitation as on Episode 11. No non-finite adjoints on this dataset.

| chunk | objective | E | nu | viscosity | tool friction | tool adhesion | floor friction | yield |
|---|---|---|---|---|---|---|---|---|
| ep03c_squeezes_seg44_c00 | 0.145 | -0.0016 | -0.0024 | -0.0001 | -0.0187 | -0.0020 | +0.0885 | -0.0004 |
| ep03c_squeezes_seg44_c05 | 0.090 | -0.0061 | -0.0071 | -0.0000 | +0.0114 | -0.0005 | +0.0080 | -0.0063 |
| ep03c_squeezes_seg44_c06 | 0.047 | +0.0016 | +0.0000 | +0.0000 | -0.0001 | -0.0001 | +0.0006 | +0.0002 |
| ep07_fast_squeezes_b_c00 | 0.269 | -0.0257 | +0.0799 | +0.0001 | +0.1893 | +0.0016 | -0.1805 | +0.0878 |
| ep07_fast_squeezes_b_c01 | 0.114 | -0.0024 | -0.0014 | +0.0000 | -0.0012 | -0.0002 | +0.0074 | +0.0000 |
| ep01_pushes_squeeze_c00 | 0.106 | +0.0002 | -0.0032 | +0.0000 | -0.0005 | -0.0000 | +0.0015 | -0.0000 |
| ep01_pushes_squeeze_c01 | 0.202 | -0.0119 | +0.0219 | +0.0000 | -0.0320 | -0.0036 | +0.1559 | +0.0100 |
| ep01_pushes_squeeze_c02 | 0.320 | +0.0004 | +0.0424 | +0.0003 | -0.0283 | -0.0042 | +0.2237 | -0.0092 |
| ep01_pushes_squeeze_c03 | 0.174 | -0.0056 | -0.0005 | -0.0000 | -0.0138 | -0.0031 | +0.1500 | +0.0023 |
| ep01_pushes_squeeze_c04 | 0.167 | -0.0070 | +0.0184 | +0.0000 | -0.0114 | -0.0019 | +0.1242 | -0.0008 |
| ep01_pushes_squeeze_c05 | 0.221 | -0.0083 | -0.0115 | -0.0000 | -0.0214 | -0.0037 | +0.1398 | +0.0029 |
| ep01_pushes_squeeze_c06 | 0.036 | +0.0001 | -0.0002 | +0.0000 | +0.0000 | -0.0000 | -0.0000 | +0.0000 |
| ep01_pushes_squeeze_c07 | 0.554 | -0.0149 | -0.0124 | -0.0000 | -0.0010 | -0.0005 | +0.0390 | -0.0006 |
| ep02_turned_spatula_squeeze_c00 | 0.335 | +0.0074 | +0.0137 | +0.0000 | +0.0000 | +0.0000 | +0.0029 | +0.0000 |
| ep02_turned_spatula_squeeze_c01 | 0.602 | -0.0015 | -0.0191 | +0.0000 | -0.0007 | -0.0013 | -0.0101 | +0.0132 |
| ep02_turned_spatula_squeeze_c02 | 0.042 | +0.0003 | +0.0001 | +0.0000 | +0.0000 | +0.0000 | +0.0001 | +0.0000 |
| ep03a_squeeze_end_seg42_c00 | 0.229 | -0.0002 | -0.0148 | +0.0000 | +0.0003 | -0.0003 | +0.0135 | -0.0001 |
| ep03a_squeeze_end_seg42_c01 | 0.199 | -0.0024 | -0.0146 | -0.0000 | +0.0099 | -0.0012 | -0.0132 | +0.0531 |
| ep03b_spatula_kneading_c00 | 0.055 | +0.0003 | -0.0009 | +0.0000 | -0.0002 | -0.0000 | +0.0005 | +0.0001 |
| ep03b_spatula_kneading_c01 | 0.128 | -0.0033 | -0.0021 | +0.0000 | -0.0049 | -0.0001 | +0.0055 | +0.0004 |
| ep03b_spatula_kneading_c02 | 0.133 | +0.0029 | -0.0020 | -0.0000 | +0.0009 | -0.0004 | +0.0022 | +0.0003 |
| ep03b_spatula_kneading_c03 | 0.119 | +0.0009 | -0.0012 | +0.0000 | -0.0030 | +0.0001 | +0.0040 | +0.0057 |
| ep05_left_edge_cross_cut_c00 | 0.294 | -0.0039 | -0.0092 | -0.0000 | +0.0000 | +0.0000 | -0.0010 | +0.0000 |
| ep05_left_edge_cross_cut_c01 | 0.162 | +0.0038 | +0.0040 | -0.0000 | -0.0001 | +0.0001 | -0.0032 | +0.0004 |
| ep05_left_edge_cross_cut_c02 | 0.091 | -0.0002 | -0.0030 | +0.0000 | +0.0001 | -0.0000 | +0.0031 | -0.0000 |
| ep06_fast_squeezes_c00 | 0.179 | -0.0042 | -0.0035 | +0.0000 | +0.0026 | -0.0003 | -0.0045 | +0.0012 |
| ep06_fast_squeezes_c01 | 0.148 | +0.0028 | +0.0089 | -0.0000 | +0.0017 | -0.0000 | -0.0014 | +0.0019 |
| ep06_fast_squeezes_c02 | 0.108 | +0.0024 | +0.0033 | -0.0000 | +0.0019 | -0.0002 | -0.0016 | +0.0013 |
| ep06_fast_squeezes_c03 | 0.127 | -0.0014 | -0.0032 | +0.0000 | -0.0044 | -0.0002 | +0.0072 | +0.0004 |
| ep06_fast_squeezes_c04 | 0.144 | +0.0045 | +0.0193 | -0.0002 | +0.0170 | -0.0006 | -0.0350 | +0.0064 |

| parameter | mean abs s | max abs s | chunks with abs s > 0.005 | sign + / - |
|---|---|---|---|---|
| E | 0.0043 | 0.0257 | 27% | 43% / 57% |
| nu | 0.0108 | 0.0799 | 47% | 37% / 63% |
| viscosity | 0.0000 | 0.0003 | 0% | 60% / 40% |
| tool friction | 0.0126 | 0.1893 | 33% | 40% / 53% |
| tool adhesion | 0.0009 | 0.0042 | 0% | 13% / 80% |
| floor friction | 0.0409 | 0.2237 | 53% | 67% / 33% |
| yield stress | 0.0068 | 0.0878 | 27% | 63% / 27% |

Column correlations of the chunk x parameter Jacobian with |r| > 0.6: nu ~ tool friction (r = +0.65), tool friction ~ floor friction (r = -0.72), tool friction ~ yield stress (r = +0.83), tool adhesion ~ floor friction (r = -0.92).
Singular values of the column-normalised Jacobian: [2.794, 2.141, 1.275, 0.88, 0.709, 0.268, 0.192].

Reading: at the starting point the objective is driven by **floor friction** (active on half the chunks, up to
s = 0.22, consistently positive = lower floor friction would reduce the loss on the push/squeeze chunks) and
**tool friction**; **nu**, **yield stress** and **E** are weakly but genuinely informative (yield mainly on the
fast squeezes and ep03a); **viscosity and tool adhesion are not identifiable** (scaled sensitivities 1e-4 and
1e-3, three orders below floor friction, active on no chunk). Two near-collinear directions: tool adhesion vs
floor friction (r = -0.92) and tool friction vs yield stress (r = +0.83); the two smallest singular values (0.19,
0.27 vs 2.8) say a six-parameter fit will determine four combinations well and two poorly. Many chunks are flat
in every parameter (ep01 c06, ep02 c02, ep03c c06, ep05, ep03b): their objective is dominated by the
reconstruction mismatch and barely moves with physics. These are gradients at one point; the sweep below maps
the landscape away from it.

## 4096-set parameter sweep (running)

`sweep/sweep_bag0.py` (sharded fork of the Episode 18 sweep): forward-only f32 evaluation of the fit's own
objective for every combination of 8 values of yield stress (100-10 000 Pa, log), E (2-60 kPa, log), tool
friction and floor friction (0.1-1.5), the other parameters at their starting values, on the five priority chunks
that pass the loss (ep03c c00, c05, c06; ep07_b c00, c01). 6 shards, resumable; results `sweep/wide4_s*.jsonl`.
~20-30 s per set and chunk when alone on the GPU, i.e. about 125 GPU-hours in total.


## Why some chunks are flat: it is tool contact, not grid resolution (2026-10-25, afternoon)

Tests run after the gradient table:

- **Grid 48 vs 96** on the two flattest chunks (9 friction combinations each): cross-cut ep05 c00 range 0.0015 at both
  grids; kneading ep03b c01 0.013 -> 0.015. Finer grid does not open the landscape (the grid-96 run needed the SDF
  qualification tolerance pinned to the grid-48 value, `TAICHIDOUGH_SDF_QUALIFICATION_GRID=48`, because the shipped
  256^3 fields fail the 0.05*dx tolerance at dx = 10.4 mm; diagnostic only).
- **Shape sensitivity on ep03c c00** (chamfer between final particle sets vs the start-parameter run): E 2 kPa 1.15 mm,
  60 kPa 0.92, 200 kPa 1.05 (objective 0.1456 / 0.1437 / 0.1411 vs 0.1455); yield 200 Pa 1.84 mm (0.1566); floor
  friction 0.2 1.95 mm (0.1111); tool friction 1.5 0.95 mm (0.1378). E changes the shape by ~1 mm over two decades
  (10 mm wider footprint at 2 kPa); the loss reacts ~4x less per mm to that spread than to the friction-driven shift.
- **Viscosity scan 0-60 Pa s** on ep03c c00 / ep07_b c00: objective 0.1455 -> 0.1439 and 0.2686 -> 0.2713 (opposite
  signs, ~1 %). Adhesion 0 -> 20 on ep07_b c00: < 0.001. Neither is identifiable at any value in range.
- **Scored-frame stride** 1 vs 4: same wall time per evaluation (the cost is the time integration, not scoring);
  objective changes (0.156 -> 0.166), so frame skipping buys nothing.
- **Simulator contact counters** (forward run of every chunk at the start point, `progress.jsonl` diagnostics):

| chunk | split | loss | max particles in contact tool1 | tool2 | yielded | chamfer rec-sim mean / last mm | observed cloud beyond dough mm | max scaled sensitivity | note |
|---|---|---|---|---|---|---|---|---|---|
| ep03c_squeezes_seg44_c00 | train | 0.145 | 15 | 464 | 30 | 4.9 / 5.9 | 93 | 0.088 |  |
| ep03c_squeezes_seg44_c01 | train | n/a | 331 | 1314 | 654 | 7.2 / 8.7 | 198 | n/a |  |
| ep03c_squeezes_seg44_c02 | train | n/a | 771 | 378 | 15 | 6.4 / 6.6 | 147 | n/a |  |
| ep03c_squeezes_seg44_c03 | train | n/a | 351 | 1074 | 885 | 6.5 / 9.2 | 152 | n/a |  |
| ep03c_squeezes_seg44_c04 | valid | n/a | 166 | 418 | 4 | 6.2 / 6.8 | 170 | n/a |  |
| ep03c_squeezes_seg44_c05 | train | 0.090 | 453 | 833 | 1380 | 6.5 / 7.1 | 73 | 0.011 |  |
| ep03c_squeezes_seg44_c06 | train | 0.047 | 897 | 1209 | 37 | 5.3 / 6.5 | -2 | 0.002 |  |
| ep07_fast_squeezes_b_c00 | train | 0.269 | 180 | 258 | 5058 | 7.0 / 8.4 | 15 | 0.189 |  |
| ep07_fast_squeezes_b_c01 | train | 0.114 | 95 | 99 | 98 | 6.3 / 11.1 | 6 | 0.007 |  |
| ep01_pushes_squeeze_c00 | valid | 0.106 | 0 | 479 | 4548 | 4.6 / 7.3 | 18 | 0.003 |  |
| ep01_pushes_squeeze_c01 | train | 0.202 | 0 | 1169 | 111 | 5.4 / 5.9 | 69 | 0.156 |  |
| ep01_pushes_squeeze_c02 | train | 0.320 | 222 | 0 | 34 | 7.7 / 8.7 | 111 | 0.224 |  |
| ep01_pushes_squeeze_c03 | train | 0.174 | 0 | 416 | 0 | 5.3 / 5.4 | 72 | 0.150 |  |
| ep01_pushes_squeeze_c04 | train | 0.167 | 379 | 0 | 374 | 5.0 / 6.8 | 97 | 0.124 |  |
| ep01_pushes_squeeze_c05 | valid | 0.221 | 0 | 177 | 1 | 5.8 / 7.8 | 94 | 0.140 |  |
| ep01_pushes_squeeze_c06 | train | 0.036 | 69 | 0 | 0 | 3.7 / 3.6 | -2 | 0.000 | weak contact |
| ep01_pushes_squeeze_c07 | train | 0.554 | 608 | 260 | 547 | 12.5 / 13.4 | 5 | 0.039 |  |
| ep02_turned_spatula_squeeze_c00 | train | 0.335 | 0 | 0 | 0 | 4.6 / 4.4 | -1 | 0.014 | no tool contact in simulation |
| ep02_turned_spatula_squeeze_c01 | train | 0.602 | 368 | 287 | 1121 | 12.6 / 10.8 | 7 | 0.019 |  |
| ep02_turned_spatula_squeeze_c02 | valid | 0.042 | 0 | 0 | 0 | 5.4 / 4.9 | -2 | 0.000 | no tool contact in simulation |
| ep03a_squeeze_end_seg42_c00 | train | 0.229 | 393 | 252 | 11219 | 6.3 / 15.1 | 11 | 0.015 |  |
| ep03a_squeeze_end_seg42_c01 | train | 0.199 | 1222 | 1177 | 3028 | 8.5 / 8.6 | 0 | 0.053 |  |
| ep03b_spatula_kneading_c00 | train | 0.055 | 87 | 163 | 325 | 4.9 / 5.4 | 3 | 0.001 |  |
| ep03b_spatula_kneading_c01 | train | 0.128 | 841 | 870 | 2853 | 7.1 / 6.7 | 7 | 0.005 |  |
| ep03b_spatula_kneading_c02 | valid | 0.133 | 121 | 206 | 12 | 6.3 / 6.5 | 13 | 0.003 |  |
| ep03b_spatula_kneading_c03 | train | 0.119 | 450 | 787 | 14 | 6.2 / 6.6 | 8 | 0.006 |  |
| ep05_left_edge_cross_cut_c00 | train | 0.294 | 0 | 0 | 0 | 13.4 / 18.2 | 77 | 0.009 | no tool contact in simulation |
| ep05_left_edge_cross_cut_c01 | train | 0.162 | 0 | 237 | 99 | 8.6 / 9.3 | 76 | 0.004 |  |
| ep05_left_edge_cross_cut_c02 | train | 0.091 | 0 | 621 | 0 | 8.3 / 5.2 | 28 | 0.003 |  |
| ep06_fast_squeezes_c00 | valid | 0.179 | 149 | 197 | 18 | 7.6 / 10.8 | 22 | 0.004 |  |
| ep06_fast_squeezes_c01 | train | 0.148 | 615 | 294 | 0 | 6.3 / 5.6 | 15 | 0.009 |  |
| ep06_fast_squeezes_c02 | train | 0.108 | 195 | 189 | 4 | 6.0 / 7.3 | 11 | 0.003 |  |
| ep06_fast_squeezes_c03 | train | 0.127 | 426 | 752 | 468 | 6.8 / 7.7 | 10 | 0.007 |  |
| ep06_fast_squeezes_c04 | train | 0.144 | 1074 | 961 | 88 | 9.5 / 21.9 | 19 | 0.035 |  |

Reading: the chunks with no sensitivity are the chunks where the **simulated tools never touch the dough** (ep05 c00,
ep02 c00, ep02 c02; ep01 c06 barely) while the recordings show the dough being moved; in ep05 c00 the Kinova blade
hovers above the dough's left edge and the parked UR blade is 300 mm away, yet the recorded cloud loses its left half
and shifts 40 mm. That is a tool-pose / collision-geometry problem (the blade-only meshes omit the mounts) on those
motions, not a material or resolution problem. The priority chunks all have strong contact; their flat members
(ep03c c06, ep07_b c01) have small recorded motion instead. Separately, 15 of 34 observation clouds extend 70-200 mm
beyond the dough (the stripe at x = 0.64 m in ep05 is the parked UR blade): tool points are being scored as dough,
including on 6 of the 9 priority chunks.

The 4096-set sweep (yield x E x tool friction x floor friction, 8 each, on the five scorable priority chunks) was
stopped at 2 075 of 20 480 evaluations on the user's decision after the first chunk showed E flat (marginal objective
0.173-0.175 across 2-60 kPa) while floor friction (0.109 at 0.2 -> 0.249 at 1.5), tool friction (0.196 -> 0.160) and
yield (0.164 at 100-200 Pa -> 0.206 at 1.4 kPa) were not; best set 0.092 vs 0.145 at the start
(`sweep/wide4_s*.jsonl`, resumable).

Gallery of all 34 chunks (perspective render + recorded-vs-simulated overlay, annotated): results page link
"bag0 chunk gallery"; files `/workspace/runs/bag0/all_renders/`.
