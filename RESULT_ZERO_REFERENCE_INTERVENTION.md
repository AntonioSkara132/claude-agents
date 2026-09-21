# Zero-reference intervention: cartesian_residual keeps orientation, vinn_rbf collapses

User asked whether we'd tested a "zero starting point" ablation. Turns out it already
exists in dom_retrieval as `intervention="zero"` (`scripts/evaluate.py --intervention
zero`, alongside `"replace"`) — zeroes the retrieved trajectories entirely before the
residual network ever sees them, evaluation-only. Ran it on two checkpoints to compare
architectures directly.

## What the intervention actually changes

Confirmed by reading the code, not assumed: `intervention="zero"` sets
`reference = 0` (via `weighted_pose_mean` over zeroed retrieved trajectories). The
residual network's own forward call, `self.residual(embedding, reference)`, therefore
sees the point-cloud embedding and a zero reference — no current pose, no retrieval
information at all reaches the network's decision. The **final** absolute prediction is
still anchored to the real current pose via
`compose_orientation_delta(current_tool_pose(observations), delta, positions)` — that
composition is unconditional, not part of the intervention. So this tests "can the
residual network produce a good *delta* with zero information from retrieval or its own
current state," not "does the whole pipeline lose access to current pose."

## Result

| condition | raw | orientation |
|---|---|---|
| `cartesian_residual`, normal (single-seed baseline) | 30.31mm | 6.02deg |
| `cartesian_residual`, zero-reference | 47.52mm | 6.90deg |
| `vinn_rbf` (zero-residual RBF), normal (3-seed mean) | 21.29mm | 5.48deg |
| `vinn_rbf`, zero-reference (seed27) | ~44mm (per-tool waypoint error) | **143.0deg** |

`cartesian_residual` (has a learned residual network) barely loses orientation accuracy
under zero-reference (6.02 -> 6.90deg, +15%) while position degrades substantially
(30.31 -> 47.52mm, +57%). It learned to predict reasonable rotation deltas essentially
from the point cloud alone, independent of what retrieval hands it.

`vinn_rbf` (no residual network at all — this is exactly the checkpoint from
`RESULT_VINN_RBF_MATCHES_RESIDUAL.md`) collapses completely on orientation
(143deg, i.e. essentially random/opposite) under the same intervention. Makes sense
architecturally: with no residual net, `prediction = reference` directly for
orientation_delta parameterization (see the early-return in `RetrievalPolicy.forward`
when `self.residual is None`), so zeroing the reference means the model has nothing
left to predict from at all.

## Reading this

Together with the earlier finding that vinn_rbf matches cartesian_residual+RBF's normal
performance, this shows those two are NOT equivalent in what they've learned — vinn_rbf
achieves its accuracy entirely through retrieval quality, with zero fallback capability,
while cartesian_residual has learned real generalizable signal in its residual network
(at least for orientation) that survives even total removal of retrieval. If robustness
to noisy/degraded retrieval matters (not just clean-retrieval accuracy), that's a point
in favor of keeping the residual network despite the equal-accuracy result under normal
conditions.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
