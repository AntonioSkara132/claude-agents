# Episode 18 tool-friction sweep handoff

Artifacts provided on 2026-09-19 for CUDA follow-up:

- `episode18_tool_friction_sweep_results.tar.gz` contains both source JSON results.
- The complete local evaluation is `friction_sweep_local/friction_sweep.json`.
- The other result is `friction_sweep_failed/friction_sweep.json`; all cases failed before simulation evaluation and it is included only for diagnosis.

## Complete evaluation

Fixed setup: Episode 18, frames 1–60, Young's modulus `130579.320726 Pa`, grid 48, SDF tool collision, tool-contact padding `0.0026041666666666665 m`, SDF resolution 64.

All complete cases contain 60 valid paired frames and no invalid frames:

| Tool-contact friction | Aggregate weighted loss |
|---:|---:|
| 0.1 | 0.3591908231 |
| 0.2 | 0.3610672627 |
| 0.3 | 0.3614626316 |

For this fixed numerical setup and metric, 0.1 is the lowest of the three tested values. The differences are small. This is not evidence for a universal material friction coefficient.

## Failed run

The separate sweep records the same three friction values, but every case exited with status 2 in about 0.011 seconds. It contains no evaluator output or loss. Compare its command/environment against the complete local evaluation before rerunning it; do not treat it as a physical result.

Please use the complete JSON for per-frame diagnostics and reproduce the run in a new output directory. Preserve the fixed parameter values and report the exact command, simulator revision and environment for any further sweep.
