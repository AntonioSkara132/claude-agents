# Correction to RESULT_DOM_RETRIEVAL_POSITION_BIAS.md: it was drift, not start-position bias

`RESULT_DOM_RETRIEVAL_POSITION_BIAS.md` diagnosed the `episode61_kugla` mid-episode outlier
(`segment-828f0f0eb1b083bb-pc000166-pc000213`) as a **position bias**: predicted-vs-recorded
*centroid* offset (mean position averaged over the whole 32-step trajectory) of 85mm/58mm per
tool, exceeding the segment's own motion range, concluding the model was "anchoring to a typical
fresh-episode-start position" rather than reading this segment's actual geometry.

That statistic conflates two mechanically different failures: being wrong from the very first
predicted waypoint (a genuine start-position/anchor problem, fixable by giving the model the
actual current pose) versus starting roughly correct and **drifting away over the course of the
trajectory** (a shape/dynamics prediction problem, not fixable by anchoring). Operator asked
whether start-position error had been investigated rigorously enough; it hadn't — decomposing the
same segment properly settles which one this actually is.

## Decomposition method (now the standard going forward)

For each query: measure the offset between predicted and true position at **t=0 only** (not the
whole-trajectory mean), then re-score the trajectory after subtracting that single offset from
every waypoint (`p_anchored(t) = p_pred(t) - offset(0)`). If anchoring meaningfully reduces error,
the failure is a start-position problem; if it doesn't (or makes it worse), the failure is
drift/shape, and error should be reported as a per-timestep breakdown, not a single blended
average or a whole-trajectory centroid statistic.

## Result: it's drift, not start bias — for both the old and the fixed model

| Model | Raw error | Start offset (t=0 only) | Anchored error | Change |
|---|---:|---:|---:|---:|
| OLD (pre-proprioception, 1024pts/K=100, no history) | 78.26mm | 17.50mm | 84.38mm | −7.8% (worse) |
| NEW (best: history-full/cartesian_residual, 1000ep) | 41.54mm | 24.55mm | 46.66mm | −12.3% (worse) |

The start offset was never large in either model (17-25mm, in line with other segments) and
anchoring it out makes error *worse*, not better, both before and after the fix. **The model was
never meaningfully wrong about where it started** — it was, and to a lesser extent still is, wrong
about how the motion evolves after that. The earlier report's centroid-offset statistic made this
look like a static position problem because averaging position over the whole trajectory blends
"correct at t=0" with "increasingly wrong later" into one number that looks like a constant shift.

## Also checked: is this masked at the episode level, and did the fix generalize?

Ran the same offset/anchor decomposition per-episode (not just this one segment) for the current
best model across all 9 test episodes: changes from anchoring are small and mixed-sign
(−6.1% to +8.0%), confirming no other episode is hiding a dominant start-position problem behind
the aggregate. `episode61_kugla`'s own episode-level aggregate (33.9mm old / 25.2mm new, all 11 of
its test segments) is unremarkable — the extreme 78-95mm numbers reported earlier were specific
to this one segment, not representative of the whole episode; aggregating over an episode already
hides the effect aggregating over the whole test set would hide again. Segment-level, not
episode-level or whole-set-level, is the right granularity for this kind of check.

## Standing practice change

Report **start-offset magnitude** (t=0 only) and **anchored/residual error** alongside raw error
for waypoint-error metrics going forward, not a whole-trajectory centroid statistic — that
statistic is what produced the wrong diagnosis here.

Scripts: `/workspace/runs/eval_start_anchored.py` (aggregate + ACT-BeT comparison),
`/workspace/runs/eval_start_anchored_per_episode.py`,
`/workspace/runs/eval_start_anchored_segment.py` (this segment specifically), all on this host.
