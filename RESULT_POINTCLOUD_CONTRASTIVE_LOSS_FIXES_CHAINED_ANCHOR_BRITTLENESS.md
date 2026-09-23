# Point-cloud contrastive loss (InfoNCE variant) fixes the gym-model's chained-anchor brittleness

Follow-up to `RESULT_NEW_FIXED_POLICY_EVALUATION.md`. Three things happened this round:
a multimodality/retrieval-quality audit of `gym_multimodal_v2` (the
`current_pointcloud_with_tool_pose` + multimodal + `relative_to_start` checkpoint from
`e21dd36`), a root-cause finding for why that checkpoint degrades badly under the
chained-anchor test, and two new loss functions (pushed to `dom_retrieval`, commits
`6c0907a` and `439b488`) that fix it. Full detail below; summary first.

**New checkpoint (InfoNCE point-cloud contrastive, weight 0.1) vs the `e21dd36` baseline:**

| | raw acc | orient acc | chamfer↔retrieval rank corr | chained-anchor degradation (raw/shape/orient) |
|---|---|---|---|---|
| baseline (`e21dd36`, `gym_multimodal_v2`) | 23.06±0.09mm | 5.39±0.10° | 0.162 | +210% / +285% / +69% |
| + InfoNCE pointcloud contrastive (0.1) | 25.00mm | 5.63° | **0.292** | **-5% / -6% / +24%** |

Chained-anchor brittleness essentially disappeared for raw/shape error, at a ~8% cost in
clean-test-set raw accuracy. Orientation still degrades but far less (+24% vs +69%).

## 1. Multimodality/retrieval-quality audit (no code changes, diagnostic only)

Three checks on `gym_multimodal_v2` (seed 27 unless noted):

- **Per-query mode visualization**: plotted all 3 predicted modes against ground truth for
  6 consecutive test queries (one episode). Modes stay visually distinct across all 6 --
  real diversity, not collapse. But the *winning* (highest-probability) mode doesn't always
  track ground truth shape closely (2 of 6 panels show a clear gap even for the dominant
  mode) -- consistent with soft, not confident, mode assignment.
- **Chamfer-vs-retrieval correlation** (ground-truth check, independent of any learned
  embedding): for 40 held-out queries against the training memory bank, compared each
  query's top-10 neighbors by raw centroid-centered Chamfer distance against its top-10
  neighbors by the model's actual RBF retrieval similarity. Result: 1.69x random overlap,
  rank correlation 0.162 -- real but weak. Geometrically similar dough states are only
  weakly more likely to be what gets retrieved.
- **Chained-anchor v2** (adapted for the no-history/static-tool-pose observation format):
  feed the model's own *predicted* last waypoint from segment A as segment B's anchor
  instead of the true measured pose, compare normal vs. chained error. 3-seed result:
  raw +210%, shape +285%, orientation +69% average degradation -- notably **worse** than
  the earlier full-history `relative_to_start` model (~+32% average) despite better
  normal-condition accuracy. This is the concerning finding that motivated the rest of
  this report.

## 2. Root cause of the chained-anchor brittleness

Traced it to `models/perception.py`'s `ProprioceptiveEncoder.forward`:

```python
embedding = self.frame_encoder(observation["observation"])
return self.projection(torch.cat((embedding, self.pose(pose)), dim=-1))
```

This is the encoder used for **retrieval matching** (`retrieval_policy.py:181`,
`embedding = self.encoder(observations)`, fed straight into the retriever). So the current
tool pose isn't only used as an additive offset at `relative_to_start` decode time (which
would be architecturally sound and genuinely step-invariant) -- it's concatenated directly
into the embedding that decides *which trajectories get retrieved in the first place*.
When the chained-anchor perturbs the input pose, the model doesn't just "teleport" the same
predicted shape to a new origin -- it retrieves a **different reference trajectory**,
because the embedding itself shifted. Confirms empirically why `shape` error (which
subtracts out each trajectory's own start point) degraded *even more* than `raw`
(+285% vs +210%) -- it's not a placement artifact, the retrieved shape itself changes with
anchor drift. Not fixed in this round -- flagging as the deeper architectural issue;
everything below is a mitigation via loss shaping, not a fix to this coupling.

## 3. Two new loss functions (pushed, `dom_retrieval` commits `6c0907a`, `439b488`)

Both are opt-in, independent of the existing `contrastive` (trajectory-outcome-based) term,
and supervise the retrieval embedding toward raw point-cloud geometric similarity instead
of motion-outcome similarity:

- **`contrastive_pointcloud_loss`** (`LossWeights.contrastive_pointcloud`): soft target
  distribution over all allowed candidates, weighted by relative centroid-centered Chamfer
  distance (128-point subsample) -- same cross-entropy-to-a-distribution mechanism as the
  existing `contrastive_retrieval_loss`, just with a geometric instead of trajectory-outcome
  target.
- **`contrastive_pointcloud_infonce_loss`** (`LossWeights.contrastive_pointcloud_hard`):
  classic hard-positive InfoNCE/SimCLR formulation -- single Chamfer-nearest memory item as
  the positive, standard cross-entropy against that one index.

Both verified with direct gradient tests (confirmed gradient reaches the actual PointNet
`point_mlp`/`projection` weights in the trained checkpoint, not just the loss function's own
unit test on synthetic leaf tensors) and 121/121 passing tests.

## 4. Ablation: same `gym_multimodal_v2` architecture, weight 0.1, three variants

| variant | raw acc | orient acc | chamfer↔retrieval corr | chained-anchor (raw/shape/orient) |
|---|---|---|---|---|
| baseline | 23.06±0.09mm | 5.39±0.10° | 0.162 | +210% / +285% / +69% |
| + soft pointcloud contrastive | 22.92mm | 5.34° | 0.143 (flat/slightly worse) | +150% / +161% / +60% |
| **+ InfoNCE pointcloud contrastive** | 25.00mm | 5.63° | **0.292** | **-5% / -6% / +24%** |

The soft-distribution version, at weight 0.1, did essentially nothing to the chamfer
correlation -- diluted by the dominant `trajectory: 1.0` + `orientation: 1.0` losses, each
candidate only getting fractional supervision. The InfoNCE version, same weight, nearly
doubled the correlation and all but eliminated the chained-anchor raw/shape blowup. Likely
mechanism: single hard-positive supervision concentrates the whole gradient budget on one
unambiguous target per example, instead of spreading it thin across a soft distribution --
much sharper training signal at the same nominal weight.

Caveat carried over from every chained-anchor result this session: 3 seeds, single
episode-pair per seed. One seed's chained error even landed below its own normal error
(18mm chained vs 31mm normal) -- take the exact magnitude with a grain of salt, but all 3
seeds land in the same "no real degradation" regime, which is the meaningful signal.

## Recommendation

`contrastive_pointcloud_hard` (InfoNCE) at weight 0.1 looks like a real, cheap win for
deployment robustness, at a real (~8%) cost to clean-test accuracy -- worth treating as the
new default for the gym-compatible line of models pending a full test-set-scale
chained-anchor evaluation (not just one episode-pair) to confirm the magnitude. The deeper
`ProprioceptiveEncoder` coupling (tool pose feeding the *retrieval-selection* embedding, not
just the `relative_to_start` decode) is still there and is the more fundamental fix if this
mitigation doesn't fully hold up at scale -- suggest decoupling retrieval-selection input
from raw absolute tool pose as the next architectural step if InfoNCE alone isn't enough.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
