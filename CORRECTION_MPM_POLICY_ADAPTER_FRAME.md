# Correction: use table-aligned Episode 18 frame for the configured simulation

Follow-up to `RESULT_MPM_POLICY_ADAPTER_IMPLEMENTED.md`.

For the configured simulation file:

```text
TaichiDough/experiments/differentiable_mpm/configs/episode18_table_aligned_registered_tools.json
```

use the nonidentity `scene_calibration_v2.json` referenced by that registered-tools bundle. Do not use an identity source transform for this table-aligned configuration.

For marker/tool geometry, extract the ordered matrices:

```text
scene/tool_geometry.json -> tools[*].marker_from_mesh
```

in the sequence order:

```text
UR5e_spathla, gen3_spathla
```

Pass the extracted `[2,4,4]` matrix array to `--marker-from-tool-frames`. The exporter accepts matrix arrays, not the complete calibration or geometry JSON objects.

The raw source poses must be transformed with the same nonidentity calibration and `marker_from_mesh` composition used by `prepare_experiment`/`ToolReplay`. The identity transform applies only to the separately documented mocap-scene baseline, not this table-aligned registered-tools run.
