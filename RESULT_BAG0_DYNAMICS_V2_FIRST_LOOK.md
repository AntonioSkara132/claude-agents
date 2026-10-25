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


## bag0 dynamics v4 (2026-10-25, 16:45): 15 motion parts with Kalman-refined poses

Installed into the same worktree (package commit = tip `9f50bf1`; its source snapshot carries uncommitted edits in
solver/state/checkpoint/calibrate/dataset_config, mainly an optional `grid_box_lower/cells` sub-box of the lattice,
unused by the v4 configs; my three local patches kept). Same SDF cache keys as v2 (same meshes).

Renders of all 15 parts (11 dataset + 4 test) at the starting parameters, perspective + recorded-vs-simulated overlay,
with contact counters: gallery "bag0 v4 parts gallery" on the results page. Every dataset part now has tool contact
in simulation (v2's no-contact chunks are gone); the two cut parts barely touch the dough (17 / 53 particle contacts
over 36-38k steps, nothing yielded; the README notes the Kinova pose is filled on the deep part of the cut), chamfer
recorded-vs-simulated 5-9 mm on pushes/kneads, peaks of 14-21 mm on the squeezes and cuts.

Gradient sensitivities, all 11 dataset parts, f64 adjoint at the starting point (s = p * dL/dp; viscosity and
adhesion omitted from the per-part table where |s| < 0.001):

| part | split | loss | steps | E | nu | viscosity | tool friction | adhesion | floor friction | yield |
|---|---|---|---|---|---|---|---|---|---|---|
| p03_E1.3_push | train | 0.104 | 11175 | -0.0119 | -0.0314 | +0.0001 | -0.0108 | -0.0015 | +0.0954 | +0.0011 |
| p05_E1.5_push | train | 0.208 | 11342 | -0.0012 | -0.0046 | +0.0000 | -0.0177 | -0.0019 | +0.1016 | +0.0033 |
| p06_E1.6_squeeze | train | 0.328 | 19014 | -0.0018 | -0.0158 | +0.0000 | -0.0070 | +0.0009 | +0.0451 | -0.0015 |
| p07_E2.1_squeeze | train | 0.313 | 27354 | +0.0092 | -0.0087 | +0.0000 | +0.0013 | -0.0003 | +0.0105 | +0.0020 |
| p09_E4.1_knead | train | 0.087 | 14344 | -0.0005 | -0.0084 | +0.0000 | -0.0005 | -0.0001 | -0.0003 | +0.0003 |
| p11_E4.3_knead | train | 0.088 | 4504 | +0.0023 | +0.0024 | -0.0000 | +0.0044 | -0.0000 | +0.0001 | +0.0004 |
| p02_E1.2_push | valid | 0.182 | 10508 | +0.0033 | -0.0110 | -0.0002 | -0.0085 | -0.0021 | +0.0727 | -0.0147 |
| p10_E4.2_knead | valid | 0.085 | 5838 | -0.0003 | -0.0020 | +0.0000 | +0.0006 | -0.0001 | +0.0003 | -0.0000 |
| p13_E4.5_knead | valid | 0.086 | 5004 | +0.0021 | -0.0019 | -0.0000 | +0.0003 | -0.0000 | -0.0012 | +0.0000 |
| p15_E6.1_cut | train | 0.130 | 38362 | -0.0000 | -0.0019 | +0.0000 | +0.0001 | -0.0000 | -0.0001 | +0.0000 |
| p16_E7.1_cut | train | 0.200 | 36027 | -0.0033 | -0.0096 | +0.0000 | -0.0006 | -0.0001 | +0.0007 | -0.0000 |

| parameter | mean abs s | max abs s | parts with abs s > 0.005 |
|---|---|---|---|
| E | 0.0033 | 0.0119 | 18% |
| nu | 0.0089 | 0.0314 | 55% |
| viscosity | 0.0000 | 0.0002 | 0% |
| tool friction | 0.0047 | 0.0177 | 36% |
| tool adhesion | 0.0007 | 0.0021 | 0% |
| floor friction | 0.0298 | 0.1016 | 45% |
| yield | 0.0021 | 0.0147 | 9% |

Pairs with |r| > 0.7: viscosity ~ yield (r = +0.92), tool friction ~ floor friction (r = -0.95), tool adhesion ~ floor friction (r = -0.76). Singular values of the column-normalised Jacobian: [2.757, 1.753, 1.001, 0.83, 0.667, 0.211, 0.067].

Reading, compared with v2: the refined poses did not make the material parameters more visible. Floor friction is
still the dominant parameter and only on the pushes and p06 (0.05-0.10); nu is the most broadly active (6 of 11
parts, but all below 0.03); E, yield, tool friction are weak (max 0.012 / 0.015 / 0.018); viscosity and adhesion
remain unidentifiable (< 0.0005). The kneads (p09-p13) and cuts are flat in every parameter (max |s| < 0.01) even
though the tools now touch the dough: the recorded motion changes the top-view loss too little. The two squeezes
have the largest losses (0.31-0.33) but small gradients: at the start point the simulated squeeze differs from the
recording in a way no parameter direction removes (chamfer peaks 12-21 mm; see overlays). The smallest singular
values (0.07, 0.21) again mean two parameter combinations are undetermined; tool friction and floor friction are
almost perfectly anti-correlated here (r = -0.95). The v2 squeeze ep07_b c00, which carried most of the material
signal in v2, has no counterpart in v4's parts.

Sweep: viscosity x E x yield (8 x 8 x 8 = 512 sets, same axes as the Episode 18 sweep) running on the two squeeze
parts only (p06, p07; 19k and 27k steps), 6 shards, forward f32, `/workspace/runs/bag0/v4/sweep/sweep_s*.jsonl`.


## Evening 2026-10-25: why E is invisible, and what makes it visible

**Recorded springback.** Width across the blades (5-95 % extent) and height from the recorded clouds: p06 58 -> 72 mm
after release (+14 mm, height 49 -> 36), p07 55 -> 78 mm (+23, height 40 -> 30). The simulation at the starting
parameters recovers 0-2 mm and bulges to 31 mm instead of 49. The recovery is far above the 5-8 mm reconstruction
noise, i.e. the data does contain E/yield information.

**Springback vs parameters (p06, forward runs, width across blades):** E 20 kPa / yield 1, 5, 10 kPa: +2 mm each;
E 60 kPa / yield 10 kPa: +1 mm; E 5 kPa / yield 1.5 kPa: +6 mm; E 3 kPa / yield 1 kPa: +8 mm; recorded +14. Yield has
no effect on the recovery; softer E recovers more but scores worse; no combination reaches the recording. Several
hundred simulated particles are squeezed out sideways past the blades at maximum squeeze while the real dough
bulges up: the escape route under/along the blades (tool contact) is what the simulation gets wrong, not the
elastic storage.

**Grid 96 on p06 (9 E x yield sets):** objective 0.291-0.354 (span 0.063, twice the grid-48 span), monotone: stiffer
and higher yield better (E 60 kPa / yield 5 kPa 0.291 vs 0.327 at the start). Needs `TAICHIDOUGH_SDF_QUALIFICATION_GRID=48`
(the shipped 256^3 fields fail the 0.05 dx tolerance at dx 10.4 mm; diagnostic override in the worktree's data.py).

**Why the loss hides it:** per-frame terms show a 0.03 floor at frame 1 rising to 0.3 during compression before any
release (reconstruction starts 4 mm lower than the recording; recorded height reads 24-27 mm before contact); the
initial-referenced depth term loses its pixel support as the dough moves; the recovery is ~30 of 115-164 frames.
Frame stride 1 vs 4 costs the same time (integration dominates) and changes the objective, so frame skipping is no
speed-up.

**New loss: shape descriptors** (`/workspace/runs/bag0/shape_loss.py`, wrapper around the rollout's loss, no tree
change): width along x, width along z and height from band means (particles ranked 90-97.5 % on each side; robust to
leaked particles; exact piecewise gradient, finite-difference error 1e-10), squared error vs the same descriptors of
the recorded cloud, scale 10 mm, optionally relative to the first frame (cancels the constant offset), optional
late-frame weighting, optional mix with the original loss. First result (p06, v4, grid 48, absolute descriptors):

| loss | value | E | nu | yield | tool friction | floor friction |
|---|---|---|---|---|---|---|
| original, all frames | 0.328 | -0.002 | -0.016 | -0.002 | -0.007 | +0.045 |
| shape descriptors | 1.390 | **-0.062** | **+0.086** | **-0.056** | +0.136 | -0.402 |

E and yield go from 0.5 % to 4-5 % effects relative to the loss (about 9x), with physically sensible signs.

**v5 package + branch tip 45423a4:** same data as v4 plus a per-part grid box (p06: 22x10x21 nodes = 4 % of 48^3);
box results are bit-identical to the full cube at grid 48; with the box, grid 96 costs ~2x grid 48 instead of 8x and
grid 144 about the same as 96 (p06 forward: 27/45 s full-48, 22/16 s box-48, 69/42 s box-96, 46/16 s box-144, under
GPU contention). Objective at 48 / 96 / 144: 0.336 / 0.321 / 0.310 (yield 300) and 0.327 / 0.315 / 0.322 (yield
3000): not converged between 96 and 144. From now on all runs use v5 at grid 96 (box indices scaled x2:
`dataset_bag0_all_von_mises_1000pa_coulomb_v5_g96.json`; a g144 variant exists too).

Running: gradient tests of all loss variants (original, first+last frame, frame-referenced depth, four shape variants)
on p06/p07 at v5/grid 96, then the viscosity x E x yield sweep (512 sets) on the two squeezes at v5/grid 96
(`/workspace/runs/bag0/v5/`). The grid-48 v4 sweep was stopped at 40 % (`v4/sweep/`, resumable).

### Correction (19:30): the springback is a resolution effect, and grid 96 resolves it

Same springback measurement on p06 at v5 / grid 96 (box): simulated width recovery after release, recorded +14 mm:

| E | yield | grid 48 | grid 96 | loss at g96 |
|---|---|---|---|---|
| 20 kPa | 1 kPa (start) | +2 mm | **+11 mm** | 0.318 |
| 20 kPa | 5 kPa | +2 | +12 | 0.315 |
| 60 kPa | 10 kPa | +1 | +9 | 0.291 |
| 60 kPa | 1 kPa | - | +8 | 0.307 |
| 3 kPa | 1 kPa | +8 | +17 | 0.347 |

At grid 96 the simulator recovers 8-17 mm depending on E (softer = more, as expected) with the recorded 14 mm inside
the range; grid 48 suppressed the recovery entirely, which is what produced the earlier "no parameter can reproduce
it" reading. So the squeeze parts are informative about E at grid 96 (the v5 grid box makes that cost ~2x grid 48),
and the fit setup should be: v5, grid 96, shape-descriptor loss (relative to first frame, late-weighted) + original
terms. Grid-144 check of the same five sets running; v5/grid-96 gradients of all loss variants on p06/p07 running.

v4 grid-48 shape-loss gradients on both squeezes (scaled s): p06 shape_rel E -0.023, nu +0.084, yield -0.029, tool
friction +0.120, floor friction -0.356; p07 shape_rel E +0.089, nu -0.101, yield +0.014, frictions ~0.015. E is a
leading material term on p07 (10-15x the original loss); the two squeezes disagree in sign on E and nu at grid 48.

Grid-144 (box) check of the same five sets on p06, springback recorded +14 mm: start +12 (loss 0.320), E 60k/yield 10k
+16 (0.280), E 20k/yield 5k +16 (0.322), E 3k/yield 1k +22 (0.346), E 60k/yield 1k +7 (0.298). Grid 96 -> 144 changes
the start point little (+11 -> +12) but the high-yield sets a lot (+9 -> +16): at 144 the yield stress starts to act on
the recovery (elastic storage), so the recovery is not converged at 96 for stiff/high-yield dough. With the box, 144
costs about the same as 96 per the benchmark; a fit intended to resolve yield should use 144.

### Thin-blade contact (19:45)

Renders of p06/p07 at grid 96 and 144 (page "bag0 squeezes at grid 96/144"): at maximum squeeze the recorded dough is
one block 52 mm tall between the blades; the simulated dough is 36-37 mm tall and a sheet of particles is extruded
along the floor past one blade; by the end of p06 material lies on both sides of the blades, in p07 a trail follows
the retracting Kinova blade. Leaked fraction (outside the recorded footprint + 20 mm) is small (1-3 %), the height
deficit is the main effect. Cause in the code: the grid-node contact band is `tool_contact_padding` = 1.3 mm
(`tool_grid_contact_padding` unset), against 2.6 mm blades and 10-21 mm cells, so the blade is porous on the grid and
dough is held mainly by the per-particle push-out within 1.3 mm.

Widening the grid band (`tool_grid_contact_padding`, SDF rebuilt with the larger extent) on p06 at grid 96:

| band | start point: loss / springback / height / leaked | E 60k, yield 10k |
|---|---|---|
| 1.3 mm (default) | 0.318 / +11 mm / 36 mm / 1.7 % | 0.291 / +9 / - / - |
| 5.2 mm (dx/2) | 0.303 / +10 / 28 / 1.0 % | 0.292 / +22 / 51 / 35 % leaked |
| 10.4 mm (dx) | 0.244 / +3 / 37 / 1.3 % | 0.632 / +44 / 84 / 0 % |

A wider band holds the dough (best p06 loss so far, 0.244) but acts as a 10-20 mm thick blade (over-squeezes to 42 mm
instead of 58) and launches stiff/high-yield dough. Not usable as a fit knob; the thin-blade contact needs a solver-side
treatment (detect in a band, enforce at the true surface, or a thicker proxy with compensated offset). The sweep and
gradient tests keep the default band.


### Loss variants at v5 / grid 96 (20:10): gradients on both squeezes

Scaled sensitivity s = p dL/dp at the starting parameters (f64 adjoint, grid-96 box). Variants: baseline = original
all-frames loss; firstlast_* = only first and last frame scored (initial- or frame-referenced depth); allframes_framedepth;
shape_* = shape-descriptor loss (abs; relative to first frame; relative + late weighting; relative + original).

| part | loss variant | value | E | nu | viscosity | tool friction | floor friction | yield |
|---|---|---|---|---|---|---|---|---|
| p06 | baseline | 0.319 | -0.014 | -0.063 | -0.000 | +0.002 | +0.036 | -0.005 |
| p06 | firstlast_initdepth | 0.362 | -0.026 | -0.139 | -0.000 | -0.001 | +0.083 | -0.022 |
| p06 | firstlast_framedepth | 0.537 | -0.031 | -0.170 | -0.000 | -0.012 | +0.091 | -0.016 |
| p06 | allframes_framedepth | 0.449 | -0.017 | -0.078 | -0.000 | -0.002 | +0.039 | -0.003 |
| p06 | shape_abs | 1.332 | -0.108 | -0.463 | +0.000 | +0.197 | -0.246 | -0.156 |
| p06 | shape_rel | 0.954 | -0.074 | -0.375 | +0.001 | +0.222 | -0.248 | -0.114 |
| p06 | shape_rel_late | 1.500 | -0.130 | -0.651 | +0.001 | +0.351 | -0.384 | -0.201 |
| p06 | shape_rel_plus_orig | 1.272 | -0.088 | -0.438 | +0.000 | +0.224 | -0.213 | -0.118 |
| p07 | baseline | 0.298 | +0.009 | -0.035 | +0.000 | -0.002 | +0.009 | +0.006 |
| p07 | firstlast_initdepth | 0.247 | +0.012 | -0.033 | +0.000 | -0.004 | +0.012 | +0.003 |
| p07 | firstlast_framedepth | 0.262 | +0.000 | -0.069 | +0.000 | -0.007 | +0.014 | +0.011 |
| p07 | allframes_framedepth | 0.279 | -0.002 | -0.080 | +0.000 | -0.005 | +0.011 | +0.015 |
| p07 | shape_abs | 1.340 | -0.059 | -0.031 | -0.000 | +0.086 | +0.019 | +0.189 |
| p07 | shape_rel | 1.606 | -0.014 | +0.012 | -0.000 | +0.069 | +0.024 | +0.186 |
| p07 | shape_rel_late | 1.983 | -0.022 | -0.019 | -0.000 | +0.092 | +0.023 | +0.294 |
| p07 | shape_rel_plus_orig | 1.905 | -0.005 | -0.023 | -0.000 | +0.067 | +0.032 | +0.192 |

Reading: at grid 96 even the original loss sees E and nu more (p06 E -0.014 vs -0.002 at grid 48, nu -0.063 vs -0.016).
First/last-frame scoring doubles that again on p06 (E -0.03, nu -0.17, yield -0.02). The shape-descriptor variants make
the material parameters leading terms: p06 E -0.07..-0.13, nu -0.38..-0.65, yield -0.11..-0.20 (8 %, 40 %, 13 % of the
loss); p07 yield +0.19..+0.29 (12-15 % of the loss), tool friction +0.07..0.09. E now has the same sign on both parts
(both want a stiffer dough), unlike at grid 48; yield still disagrees between the parts (p06 wants higher, p07 lower),
which is what the sweep at grid 144 will map. Viscosity and adhesion stay at zero in every variant.

### Why the two squeezes disagree (21:00): initial volumes, not material

Recorded first-frame cloud vs initial particle set (97.5 % extents) and the package's reconstructed volume (mass fixed
0.3 kg): p03 push 126x104 h 24 mm / 127x106 h 21, fill 0.84, 1253 kg/m3; p05 push 125x101 h 23 / 127x104 h 21, 0.84,
1297; p09 knead 102x92 h 30 / 105x97 h 27, 0.81, 1341; **p06 squeeze 124x102 h 23 / 127x106 h 19, fill 0.71, 1648**;
**p07 squeeze 105x101 h 36 / 105x103 h 33, 0.83, 992**. p06's reconstruction is ~20 % under-filled (4 mm lower than
its cloud, fill 0.71 vs 0.81-0.84 elsewhere); p07's cloud describes a 300 cm3 lump that at 0.3 kg has an impossible
density of 992 (different episode E2: wrong mass for that dough, or non-dough in the first-frame cloud). The grid-144
sweep (original loss) therefore prefers stiff/high-yield dough on p06 (best E 60 kPa, yield 10 kPa, 0.280; it wants
the small dough to stand up) and soft dough on p07 (best E 2-7 kPa at yield 200 Pa, 0.22-0.24; it wants the oversized
dough to squash), with the joint mean led by compromises (E 7-20 kPa, yield 200, 0.282). These are volume errors being
absorbed by material parameters. Recommended data fix: one dough density (~1270 kg/m3) with mass from the reconstructed
volume, and a check of p06's first-frame fill and p07's mass. Until then, material fits on the squeezes are not
trustworthy; the fits launched tonight (fit A locally, fit B on Colab) should be read with this caveat.

Why the squeezes start with the wrong volume (21:10, from the recorded clouds): per 5 mm cell top height at frame 0,
p03 push mean 17.0 mm (most cells 15-20), p05 16.7 (15-20), **p06 13.4 (most cells 10-15)** with the same footprint
(136-141 cm2): the p06 top surface is ~3.5 mm low across the whole dough, a depth/table offset in that part, and the
floor-fill reconstruction therefore starts ~20 % short. **p07** frames 0-2 show a lump 25-40 mm tall (220 of 414
cells at 30-40 mm, visible volume 293 cm3) that by frame 10 is 15-25 mm (187 cm3): the first frames contain ~100 cm3
of non-dough (blade/arm/hand or a depth error), and the initial particles were built from them. Fixes: check the
per-part depth/table offset (p06) and reconstruct p07 from a later clean frame (~10); then re-derive mass from a
common density.

### Grid-144 sweep (4x4x4, original loss, v5 box) complete; squeeze fits on hold (21:45)

p06 (64 sets, start 0.320): best 0.280 at E 60 kPa / yield 10 kPa / viscosity 0; marginal by E 0.349 (2 kPa) -> 0.292
(60 kPa), monotone, optimum on the grid's E bound; yield 10 kPa > 3 kPa > 1 kPa/200 once E is stiff; viscosity 0.002.
p07 (64 sets): best 0.221 at E 2 kPa (any yield, viscosity 40); marginal by E 0.225 (2 kPa) -> 0.299 (60 kPa), the
opposite direction; yield 200 best (0.242) vs 0.274 above. Joint mean: best 0.275 at E 60 kPa / yield 200 / viscosity
5 (p06 0.302, p07 0.247), a compromise that is neither part's optimum. Renders of both parts at p06's best set (page
"bag0 squeezes"): p06 stands 45 mm (36 at start, 52 recorded), p07 stands 39 mm (31 recorded).

Agreed diagnosis with the host-side analysis of the recorded clouds: p06 is a sound squeeze (solid ~11 mm pancake, no
step from p05, 182 cm3 consistent with its own cloud); p07's first ~10 frames carry stale depth after the preceding
lift (height decays 30 -> 18 mm while the RGB footprint is current), so its initial particles are built from
inflated frames (302 cm3): start p07 at frame ~10. The floor-fill volume of the same lump wanders 122-225 cm3 over
ep01 (p01 worst), so every part's starting volume carries a ~25 % error; a common fixed lump volume (scale each
reconstruction in height, keep footprint) is the consistent convention if the mass/lump is the same for the bag.
Decision: the 100-update squeeze fits (fit A local, fit B Colab) are on hold until p07's start and the volume
convention are fixed; the sweep, re-scoring and renders stand as diagnostics. The p06 optimum on the E bound is the
loss asking for more volume during the squeeze (recorded block at max squeeze ~280 cm3, likely dough + blade points).
