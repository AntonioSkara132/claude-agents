# Multi-episode chained-anchor evaluation: confirms the single-pair finding, reveals the failure's actual shape

Follow-up to `RESULT_CHAINED_ANCHOR_BRITTLENESS.md`, per explicit request to not rely on one
segment pair. Ran **true compounding chains** (each segment's anchor is the *previous chained
step's own prediction*, not the ground truth) through the **full ordered segment sequence of
3 held-out test episodes** (26, 30, 34 — 15, 20, 18 segments respectively), for two
checkpoints: the accuracy-optimized baseline (`robustness_sweep2/cond_bw0p3_seed27`, raw
tool-pose branch intact) and the mitigation (`no_tool_branch/cond_bw0p3_notoolbranch_seed27`,
branch removed). Script: `/workspace/runs/reference_dropout/test_chained_anchor_multi.py`.

## Per-episode pooled result

| episode | segments | baseline (raw branch): normal -> chained | ratio | mitigation (no branch): normal -> chained | ratio |
|---|---|---|---|---|---|
| episode26 | 15 | 23.0mm -> 83.2mm | **3.6x** | 31.7mm -> 31.7mm | **1.0x** |
| episode30 | 20 | 22.3mm -> 88.7mm | **4.0x** | 34.4mm -> 34.4mm | **1.0x** |
| episode34 | 18 | 16.5mm -> 84.1mm | **5.1x** | 25.4mm -> 25.4mm | **1.0x** |

Not a one-pair artifact: the baseline's brittleness (3.6-5.1x) and the mitigation's complete
elimination of it (exactly 1.0x, every episode) both hold consistently across three
independent held-out episodes.

## Error by segment index -- the failure has a specific shape, not unbounded compounding

Baseline checkpoint's chained raw error (mm), by position in the chain, one column per episode:

```
seg  0:  39  13  10   (unchained -- this IS the normal/ground-truth-anchored number)
seg  1:  61  59  71   <- jumps immediately at the FIRST chained transition
seg  2:  78 101  91
seg  3:  90 105  82
...                    <- then plateaus around 80-100mm for the rest of the chain
seg 13:  92  96  94
seg 14:  56  88  93
```

Mitigation checkpoint, same layout:

```
seg  0:  43  41  26
seg  1:  32  42  31
seg  2:  24  36  18
...                    <- stays low and flat across the whole 15-20 segment chain
seg 13:  35  40  24
seg 14:  92  30  27    <- occasional non-systematic outlier (not a trend)
```

**The baseline doesn't compound indefinitely -- it breaks once, at the very first chained
transition, then plateaus near a ceiling (~80-100mm) for the rest of the episode.** This is
consistent with the retrieval-discontinuity mechanism already identified: the first anchor
drift knocks the retrieval query into a different embedding-space neighborhood, and it stays
roughly equally "wrong" from then on rather than getting progressively worse with each
additional link in the chain. Still a severe failure (one bad transition ruins the entire rest
of a long-horizon rollout), just not a runaway explosion.

**The mitigation holds up over the full chain length tested** (up to 20 segments), not just a
single hop -- no jump at the first transition, no drift over the following segments. A few
non-systematic outliers remain (e.g. 92mm at one segment of episode26, 72mm at one segment of
episode30) that don't follow a chain-position trend, most likely ordinary per-segment
difficulty rather than anything chaining-related.

## What this does and doesn't establish

- Does establish: both the original brittleness and the branch-removal fix generalize across
  episodes, and the failure shape (single-step break + plateau, not runaway compounding) is
  consistent across episodes too.
- Does NOT yet establish: behavior on the full 9-episode held-out set (3 of 9 tested), multi-
  seed confirmation (seed 27 only), or results for `reference_dropout` as the "mitigation"
  arm (only branch-removal was tested here as the mitigation; dropout's chained-anchor
  behavior on multiple episodes hasn't been re-checked, only the single-pair number from the
  earlier report).

Raw per-segment data: `/workspace/runs/reference_dropout/multi_chain_A_withbranch.json`,
`/workspace/runs/reference_dropout/multi_chain_B_nobranch.json`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
