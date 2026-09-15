# Episode18 corrected chain v3: lower both tools by 2 mm

For the next corrected Episode18 chained simulation on:

```text
/workspace/TaichiDough
```

apply one additional experimental control adjustment:

```text
tool pose translation offset = [0.0, -0.002, 0.0] m
```

Requirements:

- Apply the offset to both tools at every control step in all three chains.
- Scene Y is the vertical axis, so subtract exactly `0.002 m` from `poses[:, 1]`.
- Do not change X/Z translation, quaternion rotation, tool velocity, timestamps, SDF data, dataset files, reconstruction files, or production simulation code.
- Apply the offset only in `experiments/differentiable_mpm/episode18_chained_comparison.py` for this corrected experimental run.
- Record the offset in each chain manifest and the root run manifest as:

```json
{
  "tool_pose_offset_m": {
    "axis": "scene_y",
    "value": -0.002,
    "applied_to_both_tools": true
  }
}
```

The corrected v3 script is still being completed locally. **Do not launch the numerical run yet.** Wait for the follow-up simulation-only command and verify that the script contains:

```python
poses = control.poses.copy()
poses[:, 1] += -0.002
```

The follow-up run will use Taichi 1.7.4 CUDA, `plastic_max=1.09`, three chains, complete `x, v, C, F, Jp` transfer, and no ROS bag during `--stage simulate`.
