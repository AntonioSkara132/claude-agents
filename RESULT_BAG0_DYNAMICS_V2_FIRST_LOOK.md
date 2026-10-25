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
