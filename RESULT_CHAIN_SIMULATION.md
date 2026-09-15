# Result: Episode18 chained simulation states — run on isolated /workspace/TaichiDough checkout

This ran on the `steffy` agent session's own local checkout (`/workspace/TaichiDough`), not the
`/mnt/Data/studenti/antonio_skara` mounted checkout — that mount is still not accessible from this
session. The user explicitly asked to run it here using this checkout's own `DATASET`/`RANGES`
paths (which were copied in during the earlier RGB top-view exchange) instead of waiting for the
mount.

## Environment had to be fixed first

- This machine only had Python 3.8.12 (conda `base`); `state.py` uses `float | None` syntax,
  which needs Python 3.10+.
- Created a new conda env `py311` (Python 3.11.13).
- taichi 1.7.4 (pinned in `requirements.txt`) fails to import here: its compiled `.so` requires
  GLIBC 2.32, but this host (Ubuntu 18.04) only has GLIBC 2.27.
- Installed **taichi 1.6.0** instead (oldest available on PyPI, works on this GLIBC) — this is a
  deviation from the pinned requirement. The script imported and ran correctly against it, but no
  extensive taichi-API-compat testing was done beyond this one run.
- Also installed `rosbags`, `scipy==1.15.3`, `scikit-image==0.25.2`, and the rest of
  `requirements.txt` into `py311`.
- Did not edit any `.py` source files.

## Run

```
DATASET=/workspace/TaichiDough/experiments/differentiable_mpm/data/episode18_kugla_temporal_v1/dataset_episode18_kugla_temporal_stretch_clamp_v1.json
RANGES=/workspace/TaichiDough/experiments/differentiable_mpm/data/episode18_kugla_temporal_v1/range_selection.json
OUTPUT=/workspace/TaichiDough/experiments/differentiable_mpm/runs/episode18_chained_custom_20260915T141631Z
```

Same parameter set and placeholder `--bag-dir`/`--conversion-metadata` paths as requested. Ran on
`--backend cuda`. Confirmed log line: `[Taichi] Starting on arch=cuda`. Finished in well under a
minute (small 4-chunk chains), then hit the expected failure on the placeholder RGB path:

```
FileNotFoundError: [Errno 2] No such file or directory: '/nonexistent/episode18_conversion_not_used_for_simulation.json'
```

## Verification

- `chains/from_chunk1/chain_manifest.json` — present ✓
- `chains/from_chunk2_replayed/chain_manifest.json` — present ✓
- `initial_state.npz` / `terminal_state.npz` — present for every chunk (01–04 in chain A, 02–04 in
  chain B) ✓
- **Discrepancy**: particle snapshot files are present but are NOT named exactly
  `particles_start.npy` / `particles_end.npy` as the request's verification checklist expected.
  The script actually writes them with frame-index suffixes, e.g.
  `particles_start_000000.npy`, `particles_end_000018.npy`. Same data, different filenames — the
  request's exact `test -f .../particles_start.npy` checks would fail even though the underlying
  simulation output is complete. Flagging this in case the downstream RGB-extraction step expects
  the exact unsuffixed filenames.

## Archive

```
Path:   /workspace/TaichiDough/experiments/differentiable_mpm/runs/episode18_chained_custom_20260915T141631Z_simulation_states.tar.gz
Size:   31M
SHA256: 4f71c26b6e169e8c6908ef6df436c9c56981bdfeed3c09db422d86954dbb5820
```

Not rerun into this output directory; all state files preserved as generated.
