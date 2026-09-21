# Request: implement and evaluate motion-level BeT for complete bimanual trajectories

Implement a motion-level extension of the BeT codebook-plus-residual idea in the local DeformPath retrieval code, then evaluate whether one discrete motion mode for a complete future is more coherent and more stable in closed loop than per-waypoint behavior tokens.

This is a research implementation request. Do not claim that the method is wholly new: original BeT uses per-action K-means bins and per-timestep residuals. Describe this method as a trajectory-level or motion-level BeT extension. Check recent literature, including BeT, VQ-BeT, action chunking, AMPLIFY, and Moto, before writing the report and state the closest prior methods and the remaining distinction.

## Scope and repositories

1. Inspect the current local `dom_retrieval` implementation and its tests before editing. Reuse existing dataset loading, recording-level splits, pose conversions, normalization, checkpoint selection, and metric code.
2. Do not modify the separate `/workspace/DeformPath` ACT--BeT implementation for this request. It remains the existing per-waypoint transformer baseline.
3. Do not discard unrelated local changes. If the implementation does not fit the current revision, report the revision and the conflict instead of forcing a patch.
4. Do not use validation or test trajectories to fit the motion codebook. The codebook, normalization statistics, and any learned trajectory encoder must use training demonstrations only.
5. Do not start the full three-seed CUDA campaign until the one-epoch smoke test, tiny-set overfit test, codebook checks, and model tests pass.

## Research question

Test this specific hypothesis:

> Assigning one codeword to a complete synchronized two-tool future trajectory can preserve temporal and bimanual coherence better than selecting an independent behavior bin at each waypoint, while a residual trajectory decoder adapts the selected motion to the observation.

The primary comparison is not only mean waypoint error. It must include coherence and closed-loop behavior.

## Proposed representation

Use the existing final horizon and output representation from the matched DeformPath configuration. Do not silently change the horizon, pose parameterization, units, or target convention. The current matched campaign uses a 32-waypoint future, two tools, and orientation-delta targets; verify the actual tensor dimensions in code rather than assuming them.

Do not initially run K-means on raw absolute flattened poses. Raw absolute positions can make clusters describe workspace location rather than motion shape. Implement and document one checked representation, preferably:

- positions expressed relative to each trajectory's first waypoint;
- orientations expressed as relative rotations or the repository's existing orientation-delta representation;
- both tools retained jointly in one trajectory;
- time order retained across the full horizon.

Keep the start/current pose as a separately defined component. The final prediction must be transformed back to the repository's expected pose convention without changing quaternion sign handling or current-pose orientation composition.

The report must state whether the codebook clusters (a) relative motion shape only, (b) complete trajectories including start, or (c) a factored start plus shape representation. Do not mix these interpretations.

## Codebook construction

Fit a K-means codebook from training trajectories only. Each codeword represents one complete future:

```text
codebook[k] : [horizon, trajectory_dim]
```

Use a deterministic seed and save:

- codeword tensor;
- training trajectory IDs assigned to each codeword;
- normalization and representation metadata;
- K-means seed, K, iterations, and convergence information.

Start with `K in {8, 16, 32}`. Do not choose K from test performance. Report cluster sizes, empty/small clusters, nearest-codeword error, and train/validation/test occupancy after assignment.

Use a distance appropriate to the representation. If a weighted start/shape/orientation distance is used, expose the weights in the config and report them. A plain flattened Euclidean distance is allowed only as a named ablation, not as an unexamined default.

## Model

Add a new method without changing existing `direct`, `vinn_rbf`, `cartesian_residual`, or other baseline behavior. The proposed method has:

```text
point cloud + tool history
          -> existing observation encoder / temporal history path
          -> transformer or transformer-style motion decoder
          -> one K-way motion-class distribution
          -> one full-horizon residual trajectory
          -> selected codeword + residual trajectory
```

The primary model predicts:

```text
logits: [batch, K]
residual: [batch, horizon, trajectory_dim]
```

At inference:

```text
k_hat = argmax(logits)
trajectory_hat = decode(codebook[k_hat] + residual)
```

The selected codeword must be shared across all waypoints. Do not independently select a class for each waypoint in the motion-level method; that is the per-waypoint BeT control baseline.

The residual must be conditioned on the observation and the selected codeword or its embedding. Keep a no-residual motion-level variant as an ablation. If training uses the ground-truth codeword for residual teacher forcing, also evaluate with the predicted codeword; predicted-codeword evaluation is primary. Clearly label oracle-codeword numbers as diagnostics only.

Use the existing loss conventions for position, orientation, acceleration, and any contrastive term. Add cross-entropy for the motion class and a full-trajectory residual loss. Ensure quaternion or rotation-vector losses follow existing repository utilities. Avoid adding a raw absolute tool-pose branch to the retrieval/classification embedding without documenting its closed-loop implications; the recent chained-anchor report found that branch to be the source of a severe retrieval discontinuity.

Add config fields rather than hard-coding values, for example:

```yaml
methods: [motion_bet]
motion_bet:
  codebook_k: 16
  codebook_seed: 7
  codebook_representation: relative_shape
  class_loss_weight: 1.0
  residual_loss_weight: 1.0
  use_predicted_codeword_for_residual: true
```

Use names that match the actual repository configuration conventions if they differ. Document every field.

## Required baselines and ablations

Run or reuse matched results for:

1. direct + full GRU history, no retrieval;
2. VINN-RBF reference-only complete-trajectory retrieval;
3. RBF-plus-residual;
4. existing PointMAE-conditioned ACT--BeT per-waypoint transformer, clearly marked as a separate repository and diagnostic if its old evaluation is not matched;
5. motion-level BeT without a residual;
6. motion-level BeT with a full-trajectory residual.

For the new method, include:

- `K = 8, 16, 32` screening;
- predicted-codeword versus oracle-codeword diagnostic;
- relative-shape codebook versus raw-absolute codebook only if the raw version is useful as a clearly labelled ablation;
- class-only versus class-plus-residual;
- one comparison of independent per-waypoint code selection against one shared motion code, using the same encoder and training budget.

Do not call close means equivalent. Report paired recording-level differences where the recording splits match.

## Evaluation beyond average error

For every final seed and method report:

- train/validation/test recording and motion counts;
- codebook construction provenance and training-only checks;
- motion-class top-1 and top-5 accuracy;
- nearest-codeword/oracle trajectory error;
- final position error in mm, including start and shape decomposition;
- orientation error in degrees;
- final waypoint error;
- velocity and acceleration discontinuity or another explicitly defined temporal-coherence metric;
- two-tool synchronization error if available;
- inference latency and number of parameters if practical;
- checkpoint and evaluation paths.

Run the closed-loop chained-anchor test already used in `RESULT_CHAINED_ANCHOR_BRITTLENESS.md`:

1. predict segment A;
2. use its predicted endpoint as the history anchor for segment B;
3. continue for at least two consecutive segments, and preferably every eligible segment in selected held-out episodes;
4. report normal ground-truth-anchored and chained errors separately.

Also run realistic reference/input interventions where the method supports them: endpoint noise, anchor noise, wrong/lagged references, and reference shuffling. Keep the zero-reference intervention but state its quaternion/identity convention and do not present it as a realistic corruption model by itself.

For a closed-loop model, report error by segment index. Do not claim deployment robustness from a single two-segment pair.

## Validation before full training

Add unit tests for:

- codebook fitting excludes validation and test trajectories;
- deterministic codebook fitting with a fixed seed;
- codeword assignment and decode/inverse-decode round trips;
- quaternion sign and orientation conversion handling;
- one shared class is used for every waypoint;
- residual output has the expected horizon and trajectory dimension;
- oracle and predicted-codeword paths are distinct and correctly labelled;
- no accidental use of ground-truth target trajectories during inference;
- model save/reload preserves codebook and config metadata.

Run the focused tests and the full existing test suite. Run a tiny-set overfit test that verifies the motion class and residual losses decrease. Run a one-epoch real-data CPU smoke test that verifies the training memory, codebook, split fingerprints, checkpoint, and evaluator.

## Required report

Create a report named `RESULT_MOTION_LEVEL_BET.md` with:

1. exact implementation files and commit/revision;
2. exact codebook representation and distance;
3. K and codebook statistics;
4. model architecture and loss equations;
5. tests and smoke results;
6. all seed-level metrics and mean/sample standard deviation;
7. paired recording-level comparisons when available;
8. coherence and chained-anchor results;
9. comparison with per-waypoint ACT--BeT and literature;
10. failure cases and what they do not prove;
11. recommended next experiment.

The conclusion must distinguish:

- an architectural change from original BeT;
- a new method contribution from an application-specific adaptation;
- offline imitation error from closed-loop control behavior;
- codeword selection accuracy from final trajectory accuracy.

Do not claim that the method is novel merely because it uses K-means, a transformer, or a residual head. The strongest possible claim is conditional on measured improvements in complete-trajectory coherence and chained-anchor behavior under matched evaluation.
