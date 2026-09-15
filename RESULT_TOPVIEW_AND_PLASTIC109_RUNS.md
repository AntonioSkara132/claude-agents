# Results: topview simulate stage (13 chunks) + corrected plastic_max=1.09 diagnostics/chain

All three pending requests done: REQUEST_RGB_TOPVIEW_SIMULATION_ONLY.md,
REQUEST_INDEPENDENT_FORWARD_DIAGNOSTIC.md (run with the corrected parameter set per
REQUEST_CORRECTED_PLASTIC109_RUNS.md, since that supersedes it), and the corrected chained run
from REQUEST_CORRECTED_PLASTIC109_RUNS.md section 2.

Environment: host `steffy`, `/workspace/TaichiDough`, conda env `py311` (Python 3.11.13), taichi
1.7.4, CUDA, NVIDIA GeForce RTX 4090. Same GLIBC-2.39-via-conda-forge-sysroot loader trick from
RESULT_CHAIN_SIMULATION_TAICHI174.md.

## New environment fixes needed for the full topview/forward pipeline (not just plain simulate)

`forward_video_v2/run.py` (invoked internally by both `episode18_temporal_rgb_topview.py` and
directly) forks its own simulate/render subprocesses using a bare python path — it does NOT
inherit the GLIBC loader trick automatically. Fixed by passing `--simulation-python` /
`--render-python` pointing at a wrapper script that invokes python through the sysroot's own
`ld-linux-x86-64.so.2` loader (same trick, just applied to the subprocess python too).

Additional missing pieces found and fixed, none by editing `.py` source:
- `pyvista` was not in `requirements.txt` but is required by the render dependency check — installed
  into `py311`.
- No working GLX/DRI on this headless host (`DISPLAY=:0` errors: `failed to create dri screen`,
  `GLXBadContext`). Fixed with a dedicated Xvfb display (`Xvfb :99 ...`) + `LIBGL_ALWAYS_SOFTWARE=1`
  (forces Mesa's software rasterizer) — this lets pyvista/VTK render offscreen images successfully.
- The base conda `ffmpeg` (4.3, from `pkgs/main`) only has `libopenh264`, not `libx264`, so
  `render_support.encode_video`'s `-c:v libx264 -crf 19` command failed with
  `Unrecognized option 'crf'`. Fixed by installing `imageio-ffmpeg` (bundles a static ffmpeg 7.0.2
  with libx264) into `py311` and putting it first on `PATH` (env's `bin/` takes priority over the
  base conda one). `ffprobe` still resolves to the base conda one — fine, it only reads.
- `run.py` has no CLI flag to skip only the render step while still running the simulation
  (`--prepare-only` skips simulation *and* render). So getting the full pipeline (as
  `episode18_temporal_rgb_topview.py --stage simulate` actually invokes it) required fixing
  rendering rather than bypassing it.

## 1. Topview simulate stage — all 13 chunks

```
Output:  /workspace/TaichiDough/experiments/differentiable_mpm/runs/episode18_rgb_topview_article_best_20260915T150944Z
Archive: /workspace/TaichiDough/experiments/differentiable_mpm/runs/episode18_rgb_topview_article_best_20260915T150944Z_states.tar.gz
Size:    97M
SHA256:  2842f26f0576539bf18db2053702d7700f93cc47ad19ef8fadbc8e2428598c86
Python:  3.11.13, Taichi 1.7.4, CUDA device: NVIDIA GeForce RTX 4090
```

Verification script from the request ran clean — all 13 `simulation_result.json` present with
`status: completed`, all referenced particle snapshots present. Frame counts per chunk:

```
completed chunks and frame counts: [(1, 19), (2, 18), (3, 18), (4, 17), (5, 19), (6, 17), (7, 18),
(8, 21), (9, 17), (10, 20), (11, 17), (12, 19), (13, 21)]
```

Each chunk also produced a debug video (`chunks/chunkNN/perspective/requested_material_perspective.mp4`)
as a side effect of `forward_video_v2/run.py`'s normal path — not requested, but harmless and left
in place. All simulation states preserved.

## 2. Independent forward diagnostic — C01/C02, corrected plastic_max=1.09

(Superseding the plastic_max=1.00 version requested in REQUEST_INDEPENDENT_FORWARD_DIAGNOSTIC.md —
only ran the corrected version per REQUEST_CORRECTED_PLASTIC109_RUNS.md.)

```
Output:  /workspace/TaichiDough/experiments/differentiable_mpm/runs/episode18_independent_plastic109_20260915T150145Z
Archive: /workspace/TaichiDough/experiments/differentiable_mpm/runs/episode18_independent_plastic109_20260915T150145Z_states.tar.gz
Size:    14M
SHA256:  47cbee34e9ff1cc2004fe9968c8976fbba1b3313d51894913ea09be805ad8775
```

Particle bounds, frame 0 and last frame, both chunks — compact, bounded, physically plausible
(dough stays near its initial ~0.4-0.58 m region, does NOT expand to fill 0.04-0.96 m):

```
chunk01 status=completed frames=19
  frame 0:  min=[0.4140, 0.0000, 0.4080]  max=[0.5730, 0.0250, 0.5220]
  frame 18: min=[0.4107, 0.0001, 0.4059]  max=[0.5805, 0.0439, 0.5231]
chunk02 status=completed frames=18
  frame 0:  min=[0.4200, 0.0000, 0.3901]  max=[0.5610, 0.0271, 0.5220]
  frame 17: min=[0.4324, 0.0004, 0.3857]  max=[0.5483, 0.0438, 0.5528]
```

All finite. **Diagnostic conclusion: independent C01 stays compact under the corrected parameters —
matches the corrected chained run's chunk01 bounds closely (see below), which points at the
`plastic_max=1.00` typo (not the chained-stepping/state-transfer logic) as the cause of the earlier
domain-filling instability.**

## 3. Corrected chained run — plastic_max=1.09

```
Output:  /workspace/TaichiDough/experiments/differentiable_mpm/runs/episode18_chained_plastic109_taichi174_20260915T150145Z
Archive: /workspace/TaichiDough/experiments/differentiable_mpm/runs/episode18_chained_plastic109_taichi174_20260915T150145Z_simulation_states.tar.gz
Size:    31M
SHA256:  af87f52ad176059ad60eb2e835f80294856126cb097c237f7954fc7d2d91a285
```

Both chain manifests present (`chains/from_chunk1/chain_manifest.json`,
`chains/from_chunk2_replayed/chain_manifest.json`). All `initial_state.npz`/`terminal_state.npz`
finite for every chunk. Particle position (`x`) bounds across all chunks in both chains stay in
roughly x:[0.407-0.583], y:[0.0-0.072], z:[0.406-0.582] — compact and bounded throughout, no
expansion. Full per-chunk bounds available on request; summary:

```
from_chunk1/chunk01: min≈[0.411,0.000,0.406] max≈[0.580,0.044,0.523]   (matches independent C01 closely)
from_chunk1/chunk04: min≈[0.413,0.001,0.446] max≈[0.555,0.070,0.572]
from_chunk2_replayed/chunk04: min≈[0.423,0.002,0.461] max≈[0.568,0.072,0.580]
```

Failed as expected on the placeholder RGB path after both chains completed:
`FileNotFoundError: /nonexistent/episode18_conversion_not_used_for_simulation.json`.

**This confirms the plastic_max correction (1.00 → 1.09) fixes the domain-filling instability seen
in the earlier chained run reported in RESULT_CHAIN_SIMULATION.md / RESULT_CHAIN_SIMULATION_TAICHI174.md.**
The earlier `plastic_max=1.00` runs/archives were preserved and not overwritten, per instructions.
