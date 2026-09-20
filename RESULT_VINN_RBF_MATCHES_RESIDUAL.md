# VINN-RBF (zero residual) matches the full residual-corrected RBF model

Pulled and tried the new `vinn_rbf` method (`6bc4a49`). Trained 3 seeds, bandwidth 0.15
(the confirmed-good value from `RESULT_RBF_BANDWIDTH_SWEEP.md`), everything else matched
to the rest of today's RBF work (new dataset, k=100, early stopping patience 30).

## Result

| condition | start | shape | raw | orientation |
|---|---|---|---|---|
| RBF + residual, bw=0.15 | 15.76+/-1.55mm | 20.74+/-0.98mm | 21.44+/-1.09mm | 5.57+/-0.17deg |
| **vinn_rbf, bw=0.15 (zero residual)** | 15.24+/-1.95mm | 20.32+/-1.09mm | **21.29+/-1.53mm** | **5.48+/-0.14deg** |

Per-seed vinn_rbf: seed7 22.22mm/5.61deg, seed17 22.13mm/5.50deg, seed27 19.53mm/5.33deg.

**Pure RBF-kernel-weighted retrieval, with the learned residual correction network
removed entirely, matches or very slightly beats the full residual-corrected model on
both position and orientation.** This is at the same bandwidth as the best confirmed RBF
result — not a coincidence of a different hyperparameter, an apples-to-apples ablation of
just the residual network's contribution.

## Reading this

Consistent with something this investigation already flagged early on: retrieval weight
histograms showed weights barely differing from uniform, and `retrieval_entropy_normalized`
sat at 0.996-0.999 across every well-performing variant, i.e. the residual network was
doing "almost all the work" against near-uniform retrieval weighting *under cosine
similarity*. RBF's sharper kernel changes that story — apparently RBF's kernel-weighted
average is now good enough on its own that the residual network has nothing left to
correct. Worth checking whether RBF's retrieval_entropy_normalized is meaningfully lower
than cosine's was, as a mechanistic explanation, if useful context for later sweeps.

## Practical implication

If holding up further, `vinn_rbf` is a strictly simpler model (no residual network
parameters, no residual loss term) with equal accuracy to the more complex architecture
this whole investigation has been building on. Worth considering as the new default
architecture rather than an ablation curiosity, pending: a check with the deep selection
scorer (does it still help vinn_rbf the way it helped observation-decoupled residual?),
and an MPM replay to confirm real-trajectory behavior isn't degraded despite the matching
aggregate numbers.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
