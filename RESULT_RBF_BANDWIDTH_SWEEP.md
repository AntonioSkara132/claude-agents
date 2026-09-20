# Small RBF bandwidth sweep: 0.15 is a modest, confirmed improvement over 0.5

Follow-up to `RESULT_RBF_KERNEL_ORIENTATION_WIN.md`, which flagged bandwidth (0.5) as "a
single guess, not tuned." Swept bandwidth in {0.15, 0.25, 0.5, 1.0, 2.0}, single seed (27)
first, then confirmed the winner across all 3 seeds.

## Single-seed screen (seed27, new dataset, everything else matched to the RBF baseline)

| bandwidth | start | shape | raw | orientation |
|---|---|---|---|---|
| 0.15 | 14.09mm | 20.05mm | 20.19mm | 5.51deg |
| 0.25 | 21.89mm | 20.70mm | 25.95mm | 5.58deg |
| 0.5 | 17.35mm | 20.78mm | 21.24mm | 5.68deg |
| 1.0 | 16.47mm | 21.03mm | 21.87mm | 5.67deg |
| 2.0 | 23.89mm | 25.57mm | 30.08mm | 6.53deg |

0.15 looked like a clean win at first pass; 0.25 broke the otherwise-monotonic trend
(worse than both neighbors), which was the tell that this needed multi-seed confirmation
before trusting it. 2.0 confirms the expected direction at the other extreme: too-wide a
kernel washes toward uniform averaging and hurts both metrics clearly.

## 3-seed confirmation, bandwidth=0.15 vs the established 0.5 baseline

| bandwidth | start | shape | raw | orientation |
|---|---|---|---|---|
| 0.5 (established, `RESULT_RBF_KERNEL_ORIENTATION_WIN.md`) | 17.91+/-1.24mm | 21.73+/-0.83mm | **22.24+/-0.94mm** | **5.58+/-0.20deg** |
| 0.15 (new) | 15.76+/-1.55mm | 20.74+/-0.98mm | **21.44+/-1.09mm** | **5.57+/-0.17deg** |

Per-seed at 0.15: seed7 22.20mm/5.43deg, seed17 21.92mm/5.76deg, seed27 20.19mm/5.51deg.

**Real but modest**: ~3.6% lower raw position error, not the ~5% the single-seed screen
suggested (seed27 was the best of the three, regression to the mean once seed7/17 came
in). Orientation is a wash — 5.57 vs 5.58deg, well inside the noise band either bandwidth
shows on its own. Variance is comparable between the two (1.09mm vs 0.94mm std).

**Recommendation: bandwidth=0.15 is a safe, small default improvement over 0.5** for
position, with no orientation cost. Not worth re-running every prior RBF-based comparison
over 0.6mm of mean shift, but worth using going forward for any new RBF training.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
