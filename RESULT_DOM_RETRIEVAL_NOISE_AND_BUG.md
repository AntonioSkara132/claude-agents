# Result: position-noise ablation, outlier still open, and a checkpoint-reload bug

Follow-up to `RESULT_DOM_RETRIEVAL_PROPRIOCEPTION.md`. Covers three things: the `cartesian`/
`latent_residual` completion, the position-noise commit (`f93fd51`), and a real bug found while
visualizing.

## cartesian/latent_residual completion (config from `993e5b2`)

| Method | mm |
|---|---:|
| **Cartesian** | **26.70** (best of the whole investigation) |
| Latent + residual | 29.00 (reproduces the earlier run's number exactly — good determinism check) |

## Bug: `packed_demo()` drops `tool_positions`, breaking checkpoint reload

`training/trainer.py::packed_demo()` handles `HISTORY_FIELDS` but was not updated for the new
`tool_positions` field, so `dataset.pt` never persists it. `load_experiment()` (used by
`scripts/evaluate.py` and `scripts/visualize_open_loop.py`) then rebuilds a `Split` with
`tool_positions=None` for every demo, so `MemoryBank` never registers its `tool_positions`
buffer, and `model.load_state_dict(checkpoint["state_dict"], strict=True)` fails:

```text
RuntimeError: Error(s) in loading state_dict for RetrievalPolicy:
    Unexpected key(s) in state_dict: "memory.tool_positions".
```

Fix should mirror the existing `HISTORY_FIELDS` handling in `packed_demo`:

```python
def packed_demo(d: Demonstration) -> dict[str, Any]:
    return {"id": d.id, "group": d.group, "observation": d.observation.cpu(),
            "trajectory": d.trajectory.cpu(), "metadata": d.metadata,
            **{name: getattr(d, name).cpu() for name in HISTORY_FIELDS if getattr(d, name) is not None},
            **({"tool_positions": d.tool_positions.cpu()} if d.tool_positions is not None else {})}
```

Did not patch this in the shared repo myself (didn't want to collide with concurrent work there);
worked around it locally for this host's visualization runs by reconstructing the split from raw
data (deterministic given the checkpoint's own saved config+seed) instead of the lossy
`dataset.pt` snapshot, verified against the checkpoint's `split_manifest`/hash exactly as
`load_experiment` does. Every checkpoint saved from a `proprioception.enabled: true` config on
this host so far needs this workaround to evaluate/visualize.

## Position-noise ablation (`position_noise_m: 0.003`, commit `f93fd51`)

Ran `cartesian` (the current best method) with the new ±3mm training-only input noise, same
1024pts/K=100/200-epoch setup:

| Config | Aggregate test avg | `episode61_kugla` outlier (mid-episode, near-stationary) |
|---|---:|---:|
| No noise | 26.70mm | 85.73mm (t=0 error ~16-30mm, drifts to 80-110mm by phase 0.5) |
| **+ noise** | 27.40mm (−2.6%, worse) | 84.31mm (no meaningful change) |

Noise augmentation neither improves the aggregate nor touches the specific outlier. Makes sense
given the diagnosis: this outlier's error grows from a near-correct t=0 anchor into a large
drift because the model keeps predicting typical-scale motion shape for what's actually a nearly
stationary continuation — that's a trajectory-shape/dynamics failure, not an imprecise-input
failure, so regularizing the tool-position input doesn't touch it.

## Where this leaves things

Proprioception (measured tool_positions, no noise) remains the best change found so far —
`cartesian` at 26.70mm is the overall best result. The `episode61_kugla`-style outlier (mid-episode
segment, near-stationary ground truth) is **not fixed** by either proprioception alone or
proprioception+noise: the model correctly anchors the start but still defaults to a
larger-than-true motion shape. This looks like it needs something that lets the model recognize
"this continuation should be small" from the object cloud/history rather than an input-side fix —
possibly relevant to the temporal/history path (`history_tool_positions`) already in the repo,
since a short history of recent tool motion would directly signal "this segment isn't moving
much," which a single static point + single static pose cannot.

Full artifacts: `/workspace/runs/deformpath_proprioception_cart_latent_seed7/`,
`/workspace/runs/deformpath_proprioception_noise_seed7/`,
`/workspace/runs/open_loop_proprio_cartesian/`, `/workspace/runs/open_loop_proprio_noise_cartesian/`
on this host only.
