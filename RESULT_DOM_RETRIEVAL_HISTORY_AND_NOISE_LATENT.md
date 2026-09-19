# Result: temporal history doesn't fix the outlier (yet); noise helps latent_residual, not cartesian

Follow-up to `RESULT_DOM_RETRIEVAL_NOISE_AND_BUG.md`. Two more ablations targeting the
`episode61_kugla` mid-episode/near-stationary outlier (`RESULT_DOM_RETRIEVAL_POSITION_BIAS.md`).

## Position noise on `latent_residual`

| Method | No noise | + noise (`position_noise_m: 0.003`) |
|---|---:|---:|
| Cartesian | 26.70mm | 27.40mm (worse) |
| Latent + residual | 29.00mm | **28.00mm** (better) |

Opposite effects per method — noise helps `latent_residual` a little, hurts `cartesian` a little.
Not a clean win either way; method-dependent, small in magnitude (~2-3%).

## Temporal history (`configs/history_deformpath.yaml`, adapted to real data, 1024pts)

Ran via `scripts/compare_history.py`, `current_only` vs `full` history mode, same 611-motion
split. This config's budget is much smaller than the proprioception recipe (frozen encoder, k=5,
20 epochs, no contrastive loss) — not yet apples-to-apples with the 26.70mm proprioception result.

| Method | current_only | full (GRU history) |
|---|---:|---:|
| Direct | 36.30mm | 35.33mm |
| Nearest + residual | 34.75mm | 36.43mm |
| Transformer direct | 34.69mm | 34.45mm |
| Transformer residual | 33.77mm | 37.59mm |

No consistent direction — 2 methods improve slightly, 2 regress. Checked the actual
`episode61_kugla` outlier (`segment-828f0f0eb1b083bb-pc000166-pc000213`) directly across all
8 combinations:

| Method | current_only | full |
|---|---:|---:|
| Direct | 74.92mm | 75.71mm |
| Nearest + residual | 94.79mm | 83.56mm |
| Transformer direct | 68.74mm | 78.68mm |
| Transformer residual | 83.05mm | 88.59mm |

**Still bad everywhere (68-95mm), no clear fix from adding history at this budget.** Given the
`retained_history()` implementation in `data/history.py` walks backward through the raw
per-frame recording (not segment-bounded) to build history — which should in principle give the
model real "was the tool moving recently" signal for exactly this case — the lack of improvement
here is more likely an undertraining artifact (20 epochs, frozen encoder, k=5) than evidence the
mechanism can't work. Have not yet retried history with the joint/200-epoch/k=100 recipe that
got proprioception to 26.70mm; that's the fairer next test before concluding anything about
whether history actually helps this failure mode.

## Where this leaves things

Static proprioception (`cartesian`, no noise, 26.70mm) is still the best result and beats this
history baseline by a wide margin on both the aggregate and the specific outlier — but that's
confounded by training budget, not necessarily architecture. Next natural step: re-run history
with the full optimization recipe (joint training, contrastive loss, 200 epochs, higher k) before
drawing a real conclusion about history vs. static proprioception for the outlier case.

Full artifacts: `/workspace/runs/deformpath_proprioception_noise_latent_seed7/`,
`/workspace/runs/history_deformpath_seed7/` on this host only.
