# RBF + pre-Top-K deep MLP scorer: clean loss, all 3 seeds

Follow-up to `RESULT_RBF_KERNEL_ORIENTATION_WIN.md`, which flagged stacking the two as
"not yet tried." Now tried: `retrieval_mode: rbf` + `deep_selection_scorer: true` (the
small MLP that scores query x key pairs and adds to the RBF score *before* Top-K
selection, so it changes which candidates get retrieved, not just their weights).
Same recipe otherwise (bandwidth 0.5, Top-100, new dataset, early stopping patience 30).

| seed | start | shape | raw | orientation |
|---|---|---|---|---|
| 17 | 24.47mm | 23.46mm | 27.97mm | 5.77deg |
| 27 | 23.55mm | 24.80mm | 29.13mm | 8.44deg |
| 7 | 17.22mm | 23.23mm | 23.38mm | 5.51deg |
| **mean +/- std** | 21.75+/-3.95mm | 23.83+/-0.92mm | **26.83+/-3.05mm** | **6.57+/-1.62deg** |

Compared to plain RBF (no scorer), same 3 seeds:

| condition | start | shape | raw | orientation |
|---|---|---|---|---|
| RBF plain | 17.91+/-1.24mm | 21.73+/-0.83mm | 22.24+/-0.94mm | 5.58+/-0.20deg |
| RBF + deep selection scorer | 21.75+/-3.95mm | 23.83+/-0.92mm | 26.83+/-3.05mm | 6.57+/-1.62deg |

**Worse on every single metric, and worse variance on every metric too.** Position raw
error is up ~20% (26.83mm vs 22.24mm), orientation mean is up almost a full degree and
its std alone (1.62deg) is 8x plain RBF's std (0.20deg) — driven by seed27 regressing
to 8.44deg, the worst orientation number seen with RBF retrieval at all. This isn't a
borderline or seed-dependent result anymore now that all 3 seeds are in: the pre-Top-K
deep scorer helps nothing here and actively hurts both accuracy and stability when
stacked on RBF.

For contrast, the same deep-selection-scorer mechanism paired with the
observation-decoupled residual (see the architecture sweep report) was one of the
better-performing variants — so the scorer isn't uniformly bad, it just doesn't combine
well with RBF's kernel-shaped retrieval specifically. Plausible mechanism: RBF's
Gaussian kernel is already doing implicit "soft selection" based on embedding distance,
and the deep scorer's job of re-deciding *which* candidates matter conflicts with that
rather than complementing it, whereas the observation-decoupled residual has no such
retrieval-shaping mechanism of its own to conflict with.

**Conclusion: plain RBF (no scorer) remains the best orientation result overall
(5.58+/-0.20deg) and the recommended retrieval mechanism when orientation matters.**
Not recommending the RBF+scorer combination for further use.

Visualized plain RBF's best seed (27, 21.24mm raw / 5.68deg) end-to-end (open-loop
rollout + MPM sim replay with the validated transform) on the hosted page:
https://claude.ai/artifact/MmxhTp5Gxscb4MGajC2sk6

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
