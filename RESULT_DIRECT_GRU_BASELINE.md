# Direct + GRU history baseline: nearly matches the whole retrieval pipeline

Trained the `direct` method (no retrieval, no residual — just `encoder + GRU history ->
trajectory` via `DirectPolicy`) on the current dataset/architecture for the first time.
The only prior `direct` checkpoint was from the old architecture sweep on an old dataset
with history disabled entirely (`history: None`), so not comparable. This one uses the
same nodbscan dataset, orientation_delta parameterization, and GRU history config
(`history: {enabled: true, mode: full}`) as every other checkpoint in this investigation.
3 seeds.

## Result

| seed | start | shape | raw | orientation |
|---|---|---|---|---|
| 7 | 19.91mm | 25.46mm | 26.94mm | 5.90deg |
| 17 | 23.82mm | 23.74mm | 28.24mm | 5.91deg |
| 27 | 17.22mm | 21.50mm | 22.44mm | 5.59deg |
| **mean +/- std** | 20.32+/-2.75mm | 23.57+/-1.62mm | **25.87+/-3.05mm** | **5.80+/-0.18deg** |

## In context

| condition | raw | orientation |
|---|---|---|
| direct + GRU (no retrieval at all) | 25.87+/-3.05mm | 5.80+/-0.18deg |
| no-motion baseline | 25.92mm | 6.22deg |
| cartesian_residual, cosine (original best-of-day) | 25.25mm | 6.51deg |
| RBF + residual, bw=0.15 | 21.44+/-1.09mm | 5.57+/-0.17deg |
| vinn_rbf (zero-residual RBF) | 21.29+/-1.53mm | 5.48+/-0.14deg |

A model with **no retrieval mechanism whatsoever** essentially ties the no-motion
baseline and the original cosine-retrieval+residual pipeline, and trails the best
RBF-family result by only ~4mm raw / ~0.2deg orientation. All the retrieval machinery
this investigation has spent most of its time on (cosine vs RBF, bandwidth, deep
selection scorer, learned reranker, residual-mode variants) buys roughly 4mm of position
improvement and a fraction of a degree of orientation over just regressing straight from
the point cloud + recent pose history with a GRU, no memory bank involved at all.

Retrieval clearly still helps at the margin (RBF+residual and vinn_rbf are both
genuinely better, consistently, across seeds) — but the size of that margin relative to
the complexity retrieval adds (memory bank, kernel/bandwidth tuning, selection
mechanisms) is worth weighing against a much simpler direct+history model, especially
given the zero-reference robustness results already suggest retrieval-dependent
predictions are fragile to bad retrieval in ways direct regression structurally can't
be.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
