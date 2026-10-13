# Plan: full calibration study for the paper (2026-10-13)

Scope: calibrate the TaichiDough differentiable MPM dough model from partial top-view point clouds on 5-6 recorded episodes
(candidates with prepared data: 3, 11, 14, 18, 20, 26, 30), and make the claims a paper needs: which parameters the data
identify, whether optimisation converges to one solution, how well parameters transfer across episodes and motion types,
and what the calibrated simulator buys downstream. Everything below is designed to produce a table or figure.

## 0. Gates before any experiment counts (what the last two days taught us)
G1 **Tool poses.** Every episode re-exported with the v3 pipeline (per-frame ICP, manual mount). Acceptance: 0 dough
   particles inside a tool at frame 0; forward simulation at default parameters within 2x the noise floor on the end frame.
   (v2 poses on Episode 14 cost: 1/9 finite adjoints -> 7/9, chunk 11 chamfer 64 -> 7 mm.)
G2 **Contact model.** Adhesion term off or replaced by normal-only adhesion with a pull-off limit; floor as Coulomb
   friction with a fitted mu_table (dt-independent). Until then stickiness = 0 and floor_retention reported as a nuisance.
G3 **Adjoint health.** Gradient preflight per chunk; report pass rate per episode; f64 + atomic P2G; log the failing
   step for the solver owner. Chunks that fail stay in as forward-only validation.
G4 **Loss fixed in advance.** Frame-referenced depth (height map) + coverage + distance on the first and last frame of each
   chunk (occluded intermediate frames excluded), occluder-masked. One loss for all experiments; variants only in E3.
G5 **Noise floor.** Reconstruction repeatability per episode (two subsamples / consecutive static frames): chamfer, pixel
   MAE, footprint area, height p95. Every result is reported relative to this floor.

## 1. Parameterisation
Fitted: E, nu, viscosity, von Mises yield stress (currently fixed at 1000 Pa; schema v4 allows it), mu_tool, mu_table
(after G2). Density from measured mass (fixed). Optimise E, yield, viscosity in log space; bounds from dough literature
(E 5-100 kPa, yield 0.3-5 kPa, viscosity 0-100 Pa s, mu 0.1-1.5). Report both raw and normalised sensitivities dL/dlog p.

## 2. Experiments
E1 **Identifiability.** For every chunk of every episode: adjoint gradient and 1-D profile curves (7 points per parameter
   in log space) of the loss at the default parameters; finite-difference check on 3 chunks. Fisher-information proxy from
   the gradients per motion class (transport / press / knead, classified from travel and height change as done for
   Episodes 11 and 14). Output: sensitivity heat-map parameter x motion class; which parameters are observable where.
E2 **Multi-start convergence (local vs global).** Per episode, 8 starts from a Latin hypercube in log space (plus the
   default). Three optimisers on the same starts: Adam on the adjoint (ours), L-BFGS on the adjoint, CMA-ES forward-only
   (no adjoint needed, parallel, uses the chunks that fail G3). 100 updates / equivalent forward budget. Output: final loss
   and parameter spread per start, clustering of solutions, fraction reaching the best basin, wall-clock per optimiser.
   This is the "do we converge to a local or global optimum" result; also seed and batch-order sensitivity (3 shuffles).
E3 **Loss ablation (bias and variance).** Initial- vs frame-referenced depth; all frames vs first+last; height map vs
   Chamfer vs EMD (DPSI's choice); occluder mask on/off. Each with 3 starts; report the solution and its multi-start spread.
E4 **Cross-episode generalisation (central table).** Per-episode fits and a joint fit over all episodes; leave-one-episode-
   out: fit on N-1, evaluate forward-only on the held-out episode (chamfer, pixel MAE, footprint, height). Consistency of
   the fitted parameters across episodes of the same dough recipe. Output: parameter table with per-episode confidence from E2.
E5 **Motion-class transfer.** Fit on transport chunks only -> evaluate on compression chunks, and the reverse; fit on
   one tool -> evaluate on the other. Systematises the Episode 11 (contact only) vs Episode 14 (compression) finding.
E6 **Contact-model ablation.** Adhesion on / off / normal-only-with-limit; floor damping vs Coulomb; each refitted (3 starts)
   and evaluated on the push-and-return synthetic test and on the recorded chunks. Output: the "which contact model" table.
E7 **Numerics.** f32 vs f64, P2G mode, segment length 32/64/128: adjoint pass rate, gradient agreement with finite
   differences, consistency-check failures, cost per update. Supports the method section's choices.
E8 **Downstream value.** Default vs calibrated simulator in the simulator-in-the-loop stroke search (CMA-ES over the CVAE
   prior, 13 Episode 18 chunks) and the progress-reward transfer test: does calibration change which strokes are selected,
   and do simulated outcomes agree better with the recorded outcomes? Output: before/after table on the existing pipeline.
E9 **Perception robustness.** Repeat E4's evaluation with subsampled, occluded and noisy clouds (the same perturbations as
   the encoder robustness table) to show how calibration quality depends on the sensing.

## 3. Budget (this GPU, f64 adjoint fits ~4-8 min per update per chunk)
A 100-update adjoint fit over ~6 chunks: 6-10 h. E2 as written (6 episodes x 9 starts x 2 adjoint optimisers) is ~900 GPU
hours: not feasible. Priority order and cuts: (1) E1 and E5 on all episodes (gradients only, ~1 h per episode); (2) E2 with
CMA-ES forward-only for all starts (parallel, ~2-3 h per episode) and adjoint Adam/L-BFGS from the 3 best CMA-ES basins
only; (3) E4 joint fit once (~1 day) plus leave-one-out with CMA-ES; (4) E3, E6, E7 on two episodes (one transport, one
compression); (5) E8, E9 last. Total on the order of two to three GPU-weeks; a second GPU halves it.

## 4. Deliverables
Tables: sensitivity heat-map (E1); multi-start convergence per optimiser (E2); loss ablation (E3); cross-episode parameter
table with spreads and leave-one-out errors (E4); transfer matrix motion class x motion class (E5); contact-model table
(E6); downstream before/after (E8). Figures: profile curves, solution clusters in parameter space, per-episode forward
snapshots (top-down + 3-D with tools) at default / per-episode / joint parameters, push-and-return animations.

## 5. Risks
Tool-pose quality on the older episodes (3, 26 had transform problems); adjoint failures concentrated in the compression
chunks (then CMA-ES carries E2/E4 and the paper says so); material parameters may remain unidentifiable at the current
reconstruction noise (then E1 and E9 are the result, and the paper is about what partial-view data can calibrate).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
