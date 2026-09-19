# Result: baseline simulation now renders too (GLX/ffmpeg both fixed)

Follow-up to `RESULT_TAICHIDOUGH_BASELINE_SIMULATION.md`, which had the physics succeeding but
rendering failing on a GPU OpenGL driver error. Both issues below are environment-only; no
TaichiDough source touched.

## Fix 1: GLX/dri failure → `xvfb-run`

`render.py`'s `pv.Plotter(off_screen=True, ...)` still needs a working GLX context even in
off-screen mode; this host's `DISPLAY=:0` fails (`failed to create dri screen`, nouveau and swrast
drivers both fail to load, `GLXBadContext`). No NVIDIA EGL userspace libraries are installed here
(CUDA runtime only), so a GPU-accelerated headless path isn't available either. `Xvfb`/`xvfb-run`
are present on this host, though, and wrapping the render Python in a fresh virtual display fixes
it cleanly — verified with a standalone pyvista sphere render first before touching the real
pipeline:

```bash
xvfb-run -a /opt/conda/envs/py311/bin/python -m your_render_script ...
```

Used as a one-line wrapper script passed via `render.py`'s own invocation (or `--render-python` in
`run.py`, for the full pipeline) — no source changes.

## Fix 2: FFmpeg encoding failure → assemble GIF directly

With GLX fixed, all 21 frames rendered successfully (`Rendered 21/21`), but the built-in FFmpeg
video-encode step then failed:

```
Unrecognized option 'crf'.
Error splitting the argument list: Option not found
```

Same root cause as the earlier `dom_retrieval` visualizer issue: this host's ffmpeg only has
`libopenh264`, not `libx264`, and `-crf` is an x264-only option. Frame images are saved regardless
(`perspective/frames/frame_*.png`), so no data was lost. Assembled them into a GIF directly with
ffmpeg's native (codec-independent) GIF encoder instead of relying on the tool's own MP4 path:

```bash
ffmpeg -y -framerate 6 -i frame_%05d.png output.gif
```

## Result

`mpm_episode18_baseline.gif` (attached, 2.4MB, 21 frames) — the recorded-action baseline replay,
correctly labeled with the identified material parameters (E=17.072 kPa, ν=0.48964,
viscosity=0.33672 Pa·s, plastic 0.728–1.092), showing both tools and the dough particle cloud on
the floor plane. Confirms the full simulation + visualization pipeline works end-to-end on this
host now, not just the physics.

Full frames and simulation state at
`/workspace/TaichiDough/experiments/differentiable_mpm/runs/dom_policy_identified_20260919T100203Z_recorded/`
on this host.
