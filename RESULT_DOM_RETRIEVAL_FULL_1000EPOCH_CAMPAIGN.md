# Result: full 1000-epoch campaign (all methods, proprioception+quaternions+noise, history, transformer)

Follow-up to everything since `RESULT_DOM_RETRIEVAL_HISTORY_AND_NOISE_LATENT.md`. Operator asked to
train every method (including history and the new transformer-adapter/quaternion capability) on
the proven optimization recipe (joint training, `loss.contrastive: 0.1`, K=100, 1024pts) with
noise, at 1000 epochs, then visualize everything.

Pulled `ec3d552` ("transformer adapter") first: the actual diff adds `include_tool_quaternions`
(pose_dim 6→14, appending each tool's quaternion to its XYZ) to `ProprioceptiveEncoder`, and adds
`transformer_residual` to `configs/deformpath_proprioception.yaml`'s methods list.
`transformer_direct`/`transformer_residual` turn out to be generic policies
(`training/trainer.py::ALL_METHODS`) usable with *any* encoder, not history-specific — confirmed
via `make_policy`, which only special-cases `method.startswith("transformer_")` on `config.get("transformer", {})`. 71/71 tests still pass.

## Two training runs, 1000 epochs, 8 methods each

**Group A** — proprioception + quaternions (14-dim pose) + position noise, real 611-motion split, no history:

```yaml
data: {include_tool_positions: true, include_tool_quaternions: true, points: 1024, ...}
proprioception: {enabled: true, position_noise_m: 0.003}
k: 100, freeze_encoder: false, loss.contrastive: 0.1, epochs: 1000
methods: [direct, nearest, nearest_residual, cartesian, cartesian_residual, latent_residual, transformer_direct, transformer_residual]
```

**Group B** — temporal history (`compare_history.py`, `current_only` vs `full`), same recipe otherwise
(noise doesn't apply here — `augment_tool_positions` only touches static `tool_positions`, not
`history_tool_positions`; the code already rejects enabling both).

### Full test-set results (119 motions)

| Method | Group A (proprio+quat+noise) | GroupB current_only | GroupB full |
|---|---:|---:|---:|
| Direct | 28.66mm | 28.53mm | 26.74mm |
| Nearest | 33.09mm | 36.96mm | 30.77mm |
| Nearest + residual | 29.17mm | 33.08mm | 29.00mm |
| Cartesian | **25.37mm** | 27.46mm | 26.26mm |
| Cartesian + residual | 27.72mm | 27.90mm | **24.38mm** (best overall) |
| Latent + residual | 26.75mm | 27.93mm | 25.82mm |
| Transformer direct | 27.51mm | 26.80mm | 25.88mm |
| Transformer residual | 27.73mm | 28.60mm | 25.99mm |

`full` history beats `current_only` on every method this time (unlike the earlier undertrained
ablation in `RESULT_DOM_RETRIEVAL_HISTORY_AND_NOISE_LATENT.md`) — confirms that ablation's
caveat: history wasn't ineffective, it was undertrained at 20 epochs/frozen encoder/k=5.

## Convergence check: heavy overfitting past best epoch, but results are still valid

None of these converged to a stable plateau — validation MSE bottoms out somewhere between
epoch 69 and 894 depending on method, then gets noisy/worse for the rest of the 1000-epoch
budget (e.g. GroupB full `cartesian_residual`, our best result, peaked at epoch 195 then got
115% worse in normalized validation MSE by epoch 1000). This does **not** invalidate the
reported numbers: `train_policy` always keeps and reloads the best-validation-epoch checkpoint
before final evaluation, never the last-epoch weights. Practical takeaway: 1000 epochs was mostly
wasted compute past ~300 epochs for most methods (only GroupA `cartesian` genuinely used most of
the budget, peaking at epoch 894); a patience-based early-stopping rule would get identical
results far more cheaply.

## Visualization: the position-bias outlier is fixed for some methods, not others

Re-ran the open-loop visualizer (5 evenly-spaced test queries, including the known
`episode61_kugla` mid-episode/near-stationary outlier, query index 118) against all 16 checkpoints.
Note: Group A checkpoints hit the previously-reported `packed_demo()`/`tool_positions` reload bug
again — used a generalized version of the earlier workaround script
(`visualize_checkpoint_workaround.py`, reconstructs the split from raw data instead of the lossy
`dataset.pt` snapshot, verified against the checkpoint's own `split_manifest`/hash) for every
checkpoint uniformly. That bug is still unfixed in the shared repo as of `ec3d552`.

| Method | Source | 5-query avg | Outlier (episode61_kugla) |
|---|---|---:|---:|
| Direct | GroupA | 36.38mm | 77.42mm |
| Nearest | GroupA | 49.77mm | 83.76mm |
| Nearest + residual | GroupA | 36.63mm | 68.03mm |
| Cartesian | GroupA | 35.54mm | 76.04mm |
| **Cartesian + residual** | **GroupA** | 28.50mm | **23.05mm** |
| **Latent + residual** | **GroupA** | **25.47mm** | **16.88mm** |
| Transformer direct | GroupA | 30.30mm | 53.60mm |
| Transformer residual | GroupA | 36.91mm | 80.96mm |
| Direct | GroupB-full | 32.44mm | 48.06mm |
| Nearest | GroupB-full | 38.91mm | 82.25mm |
| Nearest + residual | GroupB-full | 40.78mm | 89.15mm |
| **Cartesian** | **GroupB-full** | 26.46mm | **24.94mm** |
| Cartesian + residual | GroupB-full | 30.31mm | 41.54mm |
| Latent + residual | GroupB-full | 29.57mm | 49.09mm |
| Transformer direct | GroupB-full | 28.43mm | 40.11mm |
| Transformer residual | GroupB-full | 29.51mm | 26.07mm |

**Headline finding**: the outlier that stayed at 68-95mm through every earlier attempt (raw
proprioception, position noise, undertrained history) is now solved for specific
methods — GroupA `latent_residual` (16.88mm) and `cartesian_residual` (23.05mm), and GroupB-full
`cartesian` (24.94mm) — but still fails just as badly for others in the exact same training run
(GroupA `nearest`/`transformer_residual` still 80-84mm). This means the position-bias/near-
stationary-continuation failure mode is **not inherent to the task or the data** — it's solvable
given enough training and the right policy architecture (residual-on-reference or averaged-mixture
heads with enough capacity), but simple hard-selection or single-shot regression heads don't get
there even with identical encoder input and training budget. `latent_residual`'s autoencoder-mixture
path plus 1000-epoch budget looks like the most reliable fix found so far, not proprioception
alone.

Caveat: 5-query averages are noisy small-sample estimates (e.g. GroupB-full `cartesian_residual`
shows 30.31mm here vs 24.38mm on the full 119-query test set) — the outlier-specific numbers are
the reliable signal here, not the 5-query averages.

Full artifacts (per-query `preview.png`, `rollout.gif`, `errors.csv`, `summary.json`) for all 16
models at `/workspace/runs/open_loop_groupA_<method>/` and
`/workspace/runs/open_loop_groupB_full_<method>/` on this host only.

## Open items

- `packed_demo()` still drops `tool_positions` (bug reported in `RESULT_DOM_RETRIEVAL_NOISE_AND_BUG.md`) — still needs the one-line fix there.
- Worth checking whether `latent_residual`/`cartesian_residual`'s advantage on the outlier generalizes to other mid-episode/near-stationary segments, or if this is specific to `episode61_kugla`.
- A patience-based early-stopping rule would make future long runs much cheaper without changing results.
