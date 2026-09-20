# Mid-episode MPM replay works: chunk07, driven by dom_retrieval's best RBF model

Follow-up to `REQUEST_MPM_MID_EPISODE_START.md`. Found the chunked-episode
infrastructure from your recent commits (`bf35ff2` "episode 18 chunked",
`945281c` "new data", `4735e3b` "conditions for the retrieval simulation") already
materialized locally at
`/workspace/data/preprocessed_dataset/episode18_kugla/chunks/` (28 chunks,
`dataset.json` manifest, each chunk reconstructed as its own independent episode per
`README_taichi.md` section 6). Used it to run the first simulation on this host that
doesn't start from episode18's frame 0.

## What ran

- Episode/chunk: `episode18_kugla/chunks/chunk07` — raw source frames 171-187
  (~5.7s-6.2s into the recording), reconstructed with its own initial particle
  sampling and calibration, not inheriting dynamics from earlier chunks.
- Policy: best plain-RBF checkpoint (seed27, 21.24mm raw / 5.68deg — see
  `RESULT_RBF_KERNEL_ORIENTATION_WIN.md`), dom_retrieval segment
  `segment-2d750affdcc9d983-pc000172-pc000193` (closest match to chunk07's frame
  range), transformed with the validated camera->mocap transform
  (`RESULT_EPISODE18_NEW_TRANSFORM_VALIDATED.md`).
- Physics: chunk07's own shared-default parameters from its `differentiable_mpm.json`
  (Young's modulus 130,579.32 Pa, Poisson 0.3, plastic 0.9/1.1, floor retention 0.4,
  tool friction 0.5) — these are the pipeline's defaults for this chunk, not a
  material fit specific to it.
- Tool geometry: the registered-tool SDF setup from `episode18_registered_tools_v1`
  (same one documented in `REGISTERED_TOOLS.md`), resolved via `--path` overrides
  since the dataset's embedded paths are absolute to your machine
  (`/home/antonio/diplomski_antonio/...`) and don't exist on this host.
- `scene_from_source` for this chunk is identity (chunk-local calibration already in
  scene frame); `marker_from_tool_frames` came straight from `tool_geometry.json`.

## Result

Completed cleanly: 2669/2669 steps, no failure, 0.5338s simulated (control_dt=0.0002,
duration=0.5340s matching chunk07's own recorded span). Rendered and GIF'd (ffmpeg
`-crf` still unavailable on this host, same workaround as always — raw
`-framerate 4` assembly from the 8 preview PNGs). Published on the hosted page's new
"mid-episode start" section: https://claude.ai/artifact/MmxhTp5Gxscb4MGajC2sk6

## Caveats

- Only checked the plumbing (chunk loading, path overrides, control-archive timing,
  simulation stability) — did not attempt a shape/dynamics accuracy comparison against
  chunk07's own recorded reconstruction. Happy to add that next if useful.
- Physics params are shared chunk defaults, not calibrated for chunk07 specifically —
  don't read deformation magnitude as a physical claim yet.
- dom_retrieval's segment boundaries (retained_indices 170-190) don't align exactly to
  chunk07's raw frame range (171-187) — off by a few frames at each end since the two
  systems chunk episode18 independently. Close enough for this plumbing test; flag if
  you want frame-exact alignment for a real accuracy comparison.
- One local workaround needed: `run.py`'s default `--simulation-python`/
  `--render-python` (sys.executable) breaks under this host's GLIBC/LD_LIBRARY_PATH
  setup once invoked through the wrapper that clears `LD_LIBRARY_PATH` for `run.py`
  itself — had to pass both flags explicitly pointing back at the sysroot wrapper
  scripts. Not a code issue on your end, just noting it in case it surfaces elsewhere.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
