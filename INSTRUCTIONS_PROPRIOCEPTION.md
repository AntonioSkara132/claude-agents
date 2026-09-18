# Instructions: train with segment-start tool positions

The local `dom_retrieval` framework now supports measured current tool positions as an input for non-temporal models. This addresses the position bias seen on mid-episode segments.

## What the new input is

`tool_positions` is a six-value tensor containing the XYZ positions of the two tools at the first retained sample of the segment:

```text
[x_tool_0, y_tool_0, z_tool_0, x_tool_1, y_tool_1, z_tool_1]
```

These are measured current state values. They are not future target values. The loader takes them from the first retained raw trajectory sample before target resampling.

## Config

Use the new config:

```text
dom_retrieval/configs/deformpath_proprioception.yaml
```

Important settings:

```yaml
data:
  kind: deformpath
  root: Deformapth2_predownsampled1024_interpolated
  source_root: DeformPath2_release/DeformPath/DeformPath2Bags
  points: 1024
  mode: segments
  include_tool_positions: true

proprioception:
  enabled: true

k: 100
freeze_encoder: false
loss:
  contrastive: 0.1
```

Run from the framework parent directory:

```bash
cd /workspace
python -m dom_retrieval.scripts.train \
  --config dom_retrieval/configs/deformpath_proprioception.yaml \
  --output /workspace/runs/deformpath_proprioception_joint200
```

The output directory must be new or empty according to the existing training rules.

## Model behavior

The non-temporal PointNet path now combines:

1. The initial point-cloud embedding.
2. A small MLP embedding of the six current tool coordinates.
3. A projection back to the configured embedding size.

Retrieval keys use the same point cloud and segment-start tool positions as the query. This prevents the retrieval memory from ignoring the proprioceptive input.

Temporal models should not enable this additional flag. Their `history_tool_positions` already contains the current tool pose:

```yaml
history:
  enabled: true
proprioception:
  enabled: false
```

## Checkpoint and evaluation

Keep each checkpoint together with its adjacent `dataset.pt` snapshot. Load and evaluate it using the normal commands:

```bash
python -m dom_retrieval.scripts.evaluate \
  --checkpoint /workspace/runs/deformpath_proprioception_joint200/direct/checkpoint.pt \
  --output /workspace/runs/eval_proprio_direct
```

To render open-loop predictions:

```bash
python -m dom_retrieval.scripts.visualize_open_loop \
  --checkpoint /workspace/runs/deformpath_proprioception_joint200/latent_residual/checkpoint.pt \
  --output /workspace/runs/open_loop_proprio_latent \
  --partition test \
  --count 5 \
  --format gif
```

Old checkpoints without `tool_positions` remain loadable, but they do not use proprioception. They must be retrained with `include_tool_positions: true` and `proprioception.enabled: true` to use this input.

No framework source files were changed on the CUDA host by this instruction. Pull the latest `claude-agents` main branch to read this file.
