# Motion-level BeT: core implementation built (hard, then soft RBF-weighted); neither finds real structure in the codebook

**Note: the sections below describe the original hard-argmax design as first built. It was
superseded same-day by a soft RBF-weighted variant (see "Follow-up" section below) --
`models/motion_bet.py`'s `MotionBeTPolicy` now implements the soft version; the classifier
head, `use_predicted_codeword_for_residual` config field and oracle-teacher-forcing behavior
described immediately below no longer exist in the current code. Left as-is for the
implementation history/reasoning trail rather than rewritten, since the "why" reasoning
(architecture choice, gradient-flow verification, etc.) is still accurate for how the method
was arrived at.**

Per `REQUEST_MOTION_LEVEL_BET.md`. **This is a partial report, by explicit user instruction** ("just build it for now, we will talk about tests later") — it covers the core implementation, a minimal correctness check, and two real single-seed training runs with codebook diagnostics. It does **not** cover the full spec: no baseline comparison matrix, no K∈{8,16,32} screening (K=16 and K=64 tried instead, ad hoc), no multi-seed runs, no chained-anchor re-run, no literature positioning, no coherence/two-tool-sync metrics, and none of the spec's required unit tests beyond the existing suite passing unmodified. Treat this as a checkpoint, not the deliverable the original request asked for.

## What was built

- **`poses.py`**: added `rotation_vector_from_quaternion` (log map, exact inverse of the existing `quaternion_from_rotation_vector` exp map — verified via round-trip in the realistic <1 rad regime, exact to float32 precision; a known limitation near/beyond pi radians is out of scope for this application's short manipulation trajectories) and `to_relative_delta` (converts an absolute `[...,T,14]` pose trajectory into a `[...,T,12]` delta relative to its own first waypoint — the codebook representation). Verified this composes back exactly through the existing `compose_orientation_delta` (max reconstruction error 1.8e-7).
- **`models/motion_bet.py`** (new file): `fit_motion_codebook` (deterministic from-scratch k-means in pure torch, no new dependency — `sklearn` isn't installed in this environment; fits **training trajectories only**, never validation/test), `nearest_codeword` (oracle label lookup, used for the classification loss target and for diagnostics, never fed back into the model at inference), `MotionBeTPolicy` (encoder -> K-way classifier + residual head conditioned on the selected codeword's embedding -> `codebook[k] + residual`, re-anchored onto the real observed current pose and decoded through the existing `compose_orientation_delta`, exactly like `DirectPolicy`/`RetrievalPolicy` already do).
- **`interfaces.py`**: added `logits`/`codebook` optional fields to `PolicyOutput` (default `None`, no effect on any other method).
- **`training/losses.py`**: added `motion_class: float = 0.0` to `LossWeights`; `trajectory_loss` adds a cross-entropy term against the oracle nearest-codeword label when `output.logits`/`output.codebook` are present, `zero` otherwise (no effect on other methods). The existing generic `residual` weight already applies to `raw_residual` for the L2 magnitude penalty — no new weight field needed there.
- **`training/trainer.py`**: `motion_bet` requires `pose_parameterization: orientation_delta` (raises otherwise); codebook fit inside `make_policy` from the already-normalized `split.train`; new config block `motion_bet: {codebook_k, codebook_seed, use_predicted_codeword_for_residual}`. `train_policy`'s batch loop passes `target=` into `forward()` only when `getattr(model, "wants_target", False)` — a minimal, opt-in change that doesn't touch any other method's call signature. Moved `motion_bet` into `ALL_METHODS` only (not `METHODS`), mirroring exactly how `transformer_direct`/`transformer_residual` are already handled — those also have data requirements (14D poses) the shared `test_all_registered_methods_train_and_evaluate` test's synthetic fixture doesn't satisfy. This kept the existing 104-test suite green with zero modifications to any existing test.

## Design choices worth flagging

- **Architecture is MLP heads, not a transformer** (classifier: `Linear-ReLU-Linear`; residual head: `Linear-ReLU-Linear-ReLU-Linear`), matching `DirectPolicy`'s existing pattern. The spec allowed either; chose the simpler option to get a working, testable core faster. The existing separate `TrajectoryTransformer` class (used by `transformer_direct`/`transformer_residual`) was not touched or reused.
- **No raw absolute tool-pose branch added to the encoder** — reuses whichever encoder the config builds (in production, the already-fixed `HistoryEncoder`), per the explicit warning in the request referencing `RESULT_CHAINED_ANCHOR_BRITTLENESS.md`.
- **Training uses oracle (ground-truth) nearest-codeword for the residual's teacher forcing by default** (`use_predicted_codeword_for_residual: false`), evaluation always uses the predicted codeword (`argmax(logits)`) — this is standard practice for this class of method and avoids training the residual against a moving, frequently-wrong target early in training. This required the small `wants_target` opt-in change to the training loop described above.
- **Classifier gets its own independent gradient signal** (cross-entropy against the oracle label), decoupled from the position/orientation loss path, since `argmax` blocks gradient to the classifier through the residual path entirely.

## Correctness check (smoke test, not the spec's full unit-test list)

Small synthetic dataset (3 motion families, 40 demos, non-temporal proprioception for simplicity). Confirmed: codebook fits with the right shape, training loss decreases monotonically-ish over 15 epochs (2.00 -> 1.11), full eval pipeline runs and produces finite predictions, gradients reach every trainable layer (classifier, residual head, codeword projection) after one optimizer step — a hand-rolled single-backward-pass check initially showed zero gradient in the earlier layers, traced to the deliberate zero-initialized last layer (same convention `DirectPolicy` already uses) blocking backward gradient on that first pass only; confirmed it resolves immediately after one optimizer step. Full existing 104-test suite passes unmodified throughout.

## Real results (production dataset, seed 27, single seed each)

Same data root/architecture conventions as the rest of this session (`Deformapth2_camera1024_interpolated_nodbscan`, horizon=32, points=1024, history_length=8, `HistoryEncoder` with the tool-pose branch already removed per this session's earlier fix).

| method | raw | orientation | note |
|---|---|---|---|
| RBF + residual (bw=0.15) | 21.44mm | 5.57deg | established best |
| vinn_rbf (no residual) | 21.29mm | 5.48deg | established |
| direct + GRU (no retrieval) | 25.87mm | 5.80deg | established |
| no-motion baseline | 25.92mm | 6.22deg | established |
| motion_bet, K=16 | 29.73mm | 6.10deg | 65 epochs, early-stopped at best epoch 35 -- properly converged |
| motion_bet, K=64 | 31.09mm | 6.86deg | 43 epochs, early-stopped at best epoch 13 |

Both motion_bet runs land behind every other method tried this session, including the simple `direct` and no-motion baselines.

## Codebook diagnostics -- this is the useful finding

| K | partition | top-1 acc | top-5 acc | oracle-codeword-only raw/orient (no residual) | empty clusters |
|---|---|---|---|---|---|
| 16 | train (n=437) | 18.3% | 63.2% | 48.47mm / 4.84deg | 0/16 |
| 16 | test (n=125) | 20.0% | 67.2% | 45.88mm / 4.95deg | 2/16 |
| 64 | train (n=437) | 8.9% | 27.7% | 48.50mm / 4.15deg | 0/64 |
| 64 | test (n=125) | 10.4% | 28.8% | 47.13mm / 4.83deg | 21/64 |

Two things stand out:

1. **Codebook reconstruction quality is essentially flat between K=16 and K=64** (45.88mm -> 47.13mm on test, if anything slightly worse) despite 4x more clusters. More clusters aren't carving out finer, more accurate motion templates -- the relative-shape k-means representation appears to have already saturated whatever structure it can find in this motion distribution well below K=16.
2. **Classification accuracy roughly halves going from K=16 to K=64** (top-1: 18-20% -> 9-10%), as expected when adding classes with no corresponding gain in template quality -- the extra classes are distinguishing motions that don't actually cluster apart in this representation. Test-set cluster occupancy at K=64 has 21/64 codewords never used at all, versus 2/16 at K=16 -- more granularity than the held-out motion distribution supports.

The residual head is doing nearly all of the real work in both configurations: oracle-codeword-only error (45-48mm) is worse than every other method tried this session, and the *full* model's improvement over that (down to 29.73mm/31.09mm) comes from the residual correcting both a frequently-wrong codeword pick (only right 1-in-5 times at K=16) and the codeword's own coarseness even when picked correctly. This closely parallels the very first retrieval-mechanism finding from early in this whole investigation (cosine-similarity retrieval weights were close to uniform, contributing little, with the residual network doing nearly all the work) -- a second, independent case of a discrete/hard selection mechanism failing to find real structure in this motion distribution, with the residual compensating either way.

## What this does and doesn't show

- Does show: the core implementation is mechanically correct (composes properly, trains, converges, evaluates).
- Does show: increasing K from 16 to 64 doesn't help and mildly hurts, with a clear diagnosed reason (flat codebook quality, harder classification) -- not noise, not an undertrained artifact (both runs converged with early stopping at a real best epoch).
- Does NOT show: whether motion-level BeT can work well here at all, since only two K values were tried, only one seed, oracle teacher-forcing wasn't compared against predicted-codeword training, and the more fundamental hypothesis (hard k-means clustering may be the wrong mechanism for this motion distribution, versus a soft/weighted combination closer to how RBF outperformed cosine similarity earlier in this investigation) hasn't been tested at all.
- Does NOT show: closed-loop behavior, chained-anchor robustness, coherence/synchronization metrics, or anything from the required baseline comparison matrix -- none of that has been run.

## Follow-up: soft RBF-weighted codeword blend (2026-09-21, same day)

Tested the recommended next step directly: replaced hard `argmax` codeword selection with
a soft RBF-weighted combination of all K codewords, mirroring `RBFTopK`'s exact convention
(`models/retrieval.py`) -- query and codeword-key embeddings L2-normalized, weights =
`softmax(-||q-k||^2 / (2*bandwidth^2))`, `bandwidth=0.15` matching the value that helped in
the original cosine->RBF retrieval change. `soft_codeword = sum_k weights[k] * codebook[k]`
replaces the single selected codeword; the residual head conditions on this blend instead.
This removed the oracle-vs-predicted distinction entirely (no more `argmax`, so the whole
path is differentiable end to end without needing target inside `forward()`) -- simplified
`MotionBeTPolicy` by dropping the `classifier` head, `use_predicted_codeword_for_residual`,
and the `wants_target` teacher-forcing plumbing (that hook is left in `train_policy`,
harmless and unused, in case a future variant needs it again).

Hit and fixed a real bug in the process: populating `PolicyOutput.weights` alone crashes
`evaluate_policy`'s qualitative-export path, which requires `indices`+`weights` together and,
if `indices` is set, a retrieval-style `model.memory` exposing real per-item observations for
an embedding plot -- something synthetic k-means codewords fundamentally don't have. Rather
than fabricate fake observations, backed out `weights`/`indices`/`scores`/`memory` entirely,
keeping only `logits`/`codebook` (load-bearing for the classification loss). Verified the fix
with a dedicated smoke test that exercises the same `output_dir`-set code path
`run_experiment` uses (the original smoke test hadn't caught this since it skipped
`output_dir`). Confirmed with the full 104-test suite passing throughout.

### Result: soft doesn't rescue it

| variant | raw | orientation | convergence |
|---|---|---|---|
| hard argmax, K=16 | 29.73mm | 6.10deg | 65 epochs, best=35 |
| hard argmax, K=64 | 31.09mm | 6.86deg | 43 epochs, best=13 |
| soft RBF, K=16 | 30.89mm | 6.16deg | 72 epochs, best=42 |
| soft RBF, K=32 | 29.65mm | 6.04deg | 75 epochs, best=45 |

All four converged properly (none of these are undertrained artifacts) and land in the same
tight 29.6-31.1mm / 6.0-6.9deg band -- essentially indistinguishable on a single seed. Unlike
the original retrieval investigation, where cosine->RBF was a real, repeatable improvement,
**the soft/hard mechanism change makes no meaningful difference here.** That sharpens the
earlier diagnosis rather than contradicting it: the bottleneck was never really "hard
selection is too rigid" (RBF's fix for cosine retrieval) -- it's that the k-means codebook
over this relative-shape representation doesn't carry much discriminative structure for this
motion distribution in the first place, so blending softly instead of picking hard doesn't
matter much when there isn't a lot of real signal being selected among either way. The
residual head remains the thing doing essentially all of the work in every variant tried.

## Recommended next step

The two most obvious levers (K, hard-vs-soft selection) have both now been tried and both land in the same place. If this method is still worth pursuing, the more fundamental question is the **representation itself** -- whether relative-to-start-waypoint k-means clustering can ever find real structure in this motion distribution, versus e.g. a learned (not k-means) discrete or continuous motion-embedding space, or accepting that this data may just not have strong discrete multi-modal structure to exploit and a pure regression approach (`direct`, already at 25.87mm) is the more natural fit. Multi-seed confirmation of the current numbers would also be needed before treating any of these single-seed results as settled.

Configs and checkpoints: `/workspace/runs/motion_bet/cond_motion_bet_k16_seed27{.yaml,.log,_out/}` (hard, K=16), `cond_motion_bet_k64_seed27{...}` (hard, K=64), `cond_motion_bet_soft_k16_seed27{...}`, `cond_motion_bet_soft_k32_seed27{...}`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
