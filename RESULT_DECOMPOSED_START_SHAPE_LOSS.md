# Implemented and tested the decomposed start/shape loss

Note on process: implemented this myself this time (the request in
`REQUEST_DECOMPOSED_START_SHAPE_LOSS.md` was addressed to you, but you'd gone quiet, and
the user asked me to make the change directly). Documenting exactly what changed and
what I found, same as if this were a normal handoff.

## What changed

- `poses.py`: added `start_discrepancy` (mean squared XYZ error at t=0 only) and
  `shape_discrepancy` (mean squared XYZ error after aligning both trajectories to their
  own t=0) — the same decomposition already used for reporting, now available as
  differentiable loss terms. Both handle the 6D (xyz-only) and 14D (pose) trajectory
  shapes the rest of the codebase already distinguishes.
- `training/losses.py`: added `start: float = 0.0` and `shape: float = 0.0` to
  `LossWeights` (default zero, so every existing config is unaffected), wired both into
  `trajectory_loss`'s component dict.
- `tests/test_poses.py`: two new tests — `start_discrepancy`/`shape_discrepancy`
  independence (shifted-start-same-shape gives shape=0, same-start-different-shape gives
  start=0), and that `trajectory_loss` actually respects the new weights end-to-end.
- Ran the full suite before and after: 99 -> 101 tests, all passing, no regressions.

Did not touch `trajectory`'s existing behavior or default weights — this is purely
additive. A config that doesn't set `start`/`shape` trains exactly as before.

## Trained it: RBF (bw=0.15) + residual, `trajectory: 0, start: 1, shape: 1`

Same architecture as the established `RESULT_RBF_BANDWIDTH_SWEEP.md` baseline, only the
position loss composition changed (raw position loss fully replaced by the start+shape
split; orientation/residual/acceleration/contrastive weights unchanged). 3 seeds.

| condition | start | shape | raw | orientation |
|---|---|---|---|---|
| original loss (raw position term) | 15.76+/-1.55mm | 20.74+/-0.98mm | 21.44+/-1.09mm | 5.57+/-0.17deg |
| **decomposed start+shape loss** | **11.98+/-2.12mm** | **19.32+/-0.92mm** | **21.07+/-1.67mm** | **5.35+/-0.21deg** |

Real, consistent improvement: start error down ~24%, shape down ~7%, orientation down
slightly, raw about the same (raw isn't directly optimized anymore, it's just what
falls out of start+shape combined, and it holds up fine). Exactly the effect the
hypothesis predicted for the start term specifically.

## But: doesn't change zero-reference robustness

Ran the same `intervention=zero` protocol from `RESULT_ZERO_REFERENCE_RBF_RESIDUAL_CONFIRMED.md`
on these checkpoints:

| condition | raw (zero-ref) | orientation (zero-ref) |
|---|---|---|
| original loss | 44.36+/-0.98mm | 7.13+/-0.38deg |
| decomposed start+shape loss | 44.41+/-1.47mm | 7.15+/-0.16deg |

Essentially identical. The motivating idea was that decomposing the loss might make the
retrieval/shape pathway more independently robust to a broken reference (since it's now
explicitly supervised on shape rather than raw position). That didn't happen — the
residual network's dependence on the reference for position doesn't seem to be a loss-
formulation artifact, it's more likely structural (the network genuinely needs the
reference's information to do its job, decomposing the target doesn't change what
information reaches it).

## Recommendation

Adopt the decomposed loss for accuracy under normal conditions (clear win, no cost, drop-in
compatible with existing configs via the new zero-default weights) but don't expect it to
help with retrieval robustness — that's a separate problem needing a different
intervention (e.g. training with the reference already perturbed/noised, not just changing
what the target loss measures).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
