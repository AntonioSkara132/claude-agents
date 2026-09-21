# Zero-reference robustness holds for RBF+residual too, all 3 seeds

Follow-up to `RESULT_ZERO_REFERENCE_INTERVENTION.md`, which tested the zero-reference
intervention on cosine-retrieval `cartesian_residual` (single seed) and `vinn_rbf`
(single seed). Filled in the missing cell: RBF retrieval **with** the residual network
(`cond_rbf_bw0p15_seed{7,17,27}`, the checkpoints behind
`RESULT_RBF_BANDWIDTH_SWEEP.md`), all 3 seeds.

| condition | raw | orientation |
|---|---|---|
| RBF + residual, bw=0.15, normal | 21.44+/-1.09mm | 5.57+/-0.17deg |
| **RBF + residual, bw=0.15, zero-reference** | **44.36+/-0.98mm** | **7.13+/-0.38deg** |
| vinn_rbf (no residual), zero-reference (seed27) | ~44mm | 143.0deg |

Per-seed zero-reference: seed7 43.49mm/6.71deg, seed17 45.43mm/7.24deg, seed27
44.17mm/7.45deg — tight across seeds (raw std 0.98mm, orientation std 0.38deg), so this
isn't a one-off.

## Conclusion

The robustness pattern from the cosine-retrieval test holds identically for RBF
retrieval: having a residual network at all is what matters for graceful degradation
under a broken/zeroed reference, not which retrieval mechanism feeds it. Position
roughly doubles either way (residual networks lean on the reference for position more
than orientation), orientation degrades by ~15-30% but stays coherent, and the
no-residual (`vinn_rbf`) case collapses completely regardless of retrieval mechanism
quality. Combined with the earlier accuracy-matching result, this is a clean argument
for keeping the residual network in production even though it doesn't help (and RBF
without it slightly helps) under clean-retrieval conditions — it's the only thing
providing any fallback behavior at all.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
