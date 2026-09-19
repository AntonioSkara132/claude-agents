# Request: matched retrieval ablation matrix

The latest Episode 18 policy report shows that the learned retrieval weights remain almost uniform:

- `K=100` uniform baseline weight: `0.01`;
- observed top weights: `0.011–0.023`;
- normalized retrieval entropy: approximately `0.996–0.999`;
- this held for the baseline, reranker, and deeper projection variants.

Do not run another post-Top-K reranker experiment yet. First separate candidate selection, candidate weighting, and residual prediction with the following matched ablation.

## Main configuration

Use the successful orientation-delta Cartesian-residual setup:

- same train/validation/test split and split identity;
- training retrieval memory containing training trajectories only;
- test trajectories never added to memory;
- same observation and pose-history inputs;
- same loss terms, normalization, point count, `K` where applicable, optimizer, checkpoint selection, and maximum budget;
- up to 1000 epochs;
- validation-only checkpoint selection;
- three reproducible seeds for each condition.

Keep same-recording exclusions and all existing leakage checks unchanged.

## Conditions

Run these five conditions:

1. **Cosine Top-100 with cosine weights**
   - Existing matched baseline.

2. **Same cosine Top-100 with uniform weights**
   - Keep exactly the same selected candidates as condition 1.
   - Replace cosine-derived weights with `1/100` for every selected candidate.
   - Answers whether weighting matters after candidate membership is fixed.

3. **Random 100 eligible training candidates with uniform weights**
   - Sample candidates only from the eligible training memory.
   - Use a deterministic seed per run.
   - Apply the same candidate-count and exclusion rules as the cosine condition.
   - Answers whether cosine candidate selection matters.

4. **No-motion reference with the identical residual model**
   - Repeat the measured current two-tool XYZ and orientation across the prediction horizon.
   - Do not use all-zero pose values; zero quaternions are invalid.
   - Keep the residual architecture and observation inputs unchanged.
   - Answers whether any retrieved trajectory helps compared with a physically valid stationary reference.

5. **Cosine Top-1 with residual**
   - Use the nearest eligible cosine candidate without averaging multiple candidates.
   - Answers whether averaging 100 candidates removes useful motion information.

## Required metrics

For every seed and condition, report:

- position error in millimetres;
- orientation error in degrees;
- validation-selected epoch;
- test error only from the selected checkpoint;
- retrieval entropy where retrieval is used;
- error of the reference trajectory before residual correction;
- final prediction error after residual correction;
- per-recording paired differences against condition 1;
- mean and standard deviation across the three seeds.

The reference error must be computed before the residual network changes the reference. For the no-motion condition, compare the repeated current pose against the future target using the same position and sign-invariant quaternion metrics.

## Interpretation rules

- If cosine Top-100 and uniform Top-100 are equivalent, stop tuning learned weights.
- If either retrieved Top-100 condition beats random candidates, candidate selection contributes and trajectory-supervised embeddings are the next retrieval experiment.
- If retrieved references match the no-motion reference, retrieval has not shown a measurable benefit under this model and budget; prioritize the prediction architecture.
- If Top-1 beats Top-100, test smaller fixed values such as `K=5, 10, 25` and candidate-conditioned residual prediction. Averaging may be removing motion information.
- If position and orientation prefer different conditions, report the trade-off separately; do not combine them into one unsupported ranking.
- Do not claim that retrieval is useless solely because the Top-100 weights are flat. Candidate membership and candidate weighting answer different questions.

## Leakage and reproducibility requirements

- Training memory must contain training trajectories only.
- Test trajectories must never enter retrieval memory or candidate sampling.
- Future target trajectories may supervise training losses but must not be retrieval inputs at validation or test time.
- Save the candidate-selection mode, random seed, eligible-memory identifiers, split identity, and checkpoint path in each run manifest.
- Verify checkpoint reload before evaluating each selected checkpoint.

## Follow-up after this matrix

Only after this matrix is complete:

- if candidate selection helps, train trajectory-supervised retrieval embeddings and compare `K=25, 50, 100`;
- if Top-100 averaging is harmful, test smaller K and candidate-conditioned residuals;
- if retrieval and no-motion references are equivalent, stop adding retrieval modules until the direct/residual model is improved.

This request concerns the `dom_retrieval` training/evaluation campaign. It is separate from the CUDA MPM control-condition runs in `REQUEST_MPM_POLICY_ABLATION_CUDA.md`.
