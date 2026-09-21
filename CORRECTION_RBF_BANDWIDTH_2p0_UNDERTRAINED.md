# Correction: bandwidth=2.0's "worst" result may just be undertrained, not a real bandwidth effect

Checked training curves across all recent runs after being asked whether any model
looked untrained. Found one: `cond_rbf_bw2p0_seed27` from
`RESULT_RBF_BANDWIDTH_SWEEP.md`.

## The tell

| run | epochs to early-stop | val loss: first -> best | improvement |
|---|---|---|---|
| bw=0.15 (seed7) | 83 | 0.759 -> 0.283 | 63% |
| bw=0.25 | 119 | 0.636 -> 0.428 | 33% |
| bw=1.0 | 134 | 0.631 -> 0.327 | 48% |
| **bw=2.0** | **32** | **0.634 -> 0.583** | **8%** |
| vinn_rbf / direct_gru (all seeds, for reference) | 70-270 | — | 40-60%+ |

Every other run in this whole session's sweeps trained 70-270 epochs with 33-63%
validation loss reduction. `bw=2.0` hit early-stop patience (30 epochs) after only 32
total epochs, having improved essentially only in its first ~2 epochs (8% total
reduction). That's a model that got stuck almost immediately, not one that converged to
a genuinely worse optimum.

## What this means for the earlier finding

`RESULT_RBF_BANDWIDTH_SWEEP.md` reported bw=2.0 as clearly worst (30.08mm/6.53deg) and
used it as evidence that "too-wide a kernel washes toward uniform averaging and hurts
both metrics." That directional story is still plausible (an RBF kernel with a very
large bandwidth relative to the embedding scale does approach uniform weighting, which
independently explains poor optimization: if every candidate gets near-identical weight
regardless of embedding distance, the loss landscape w.r.t. the encoder is very flat,
which would exactly produce this "stuck after 2 epochs" symptom) — but I hadn't
separated "genuinely poor performance at convergence" from "never actually converged"
before reporting it. Only ran 1 seed at bw=2.0 to begin with (it was the wide end of a
single-seed screen, never seed-confirmed like bw=0.15 was), so this was already the
weakest-evidence cell in that table.

**Not retracting the bw=0.15 result** (properly converged, 3-seed confirmed, tight
variance) — just flagging that bw=2.0's specific number shouldn't be read as a clean
"wide bandwidth is bad" data point without a proper multi-seed rerun, since this one run
may simply not have trained.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
