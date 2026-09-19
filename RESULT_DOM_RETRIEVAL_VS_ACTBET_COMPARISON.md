# Result: dom_retrieval vs. ACT-BeT (DeformPath), measured head-to-head

Operator asked to compare `dom_retrieval`'s best model against a DeformPath ACT-BeT checkpoint
(`act_bet_chunk50`, action-chunking behavior-transformer, chunk_size=50, predicts 50-step chunks
of full 7D pose per tool via BeT bin-classification + offset, frozen PointMAE encoder). No
existing metric was directly comparable (ACT-BeT reports a mixed classification+offset+rotation
loss in unclear units), so this required writing a small standalone eval script
(`/workspace/runs/eval_act_bet_chunk50.py`) that runs the actual model forward pass and computes
a position-only L2 waypoint error in millimeters, matching `dom_retrieval`'s
`average_waypoint_error` definition exactly. Did not modify `DeformPath` source.

## Two checkpoint-management bugs found in DeformPath along the way

1. **Checkpoint directory collision.** Two separate training runs (`act_bet_chunk50` and a later
   "_new" run) wrote to the same `ckpt_path` (`act_bet_transformers_checkpoints/act_bet_chunk50/`).
   The later run's `best_model.pth` (epoch 1, clearly undertrained — val loss only got worse after
   that) overwrote the original run's true best-epoch (9) checkpoint. That weight file is gone;
   confirmed by comparing embedded `val_loss`/`epoch` values against both runs' full metrics
   histories. Only the original run's *final*-epoch (40) weights survived, in the separately-named
   `act_bet_chunk50.pth`.
2. **`use_current_state_token: True` never receives real data.** `training_act_chunked_pos_quat_bet.py`
   imports `ACTChunkEpisodeDataset` from `training_act_chunked.py`, whose `__getitem__` never sets a
   `prev_pose_abs` key. `collate_act_chunks` only populates `batch["prev_pose_abs"]` when that key is
   present in the sample. Verified directly during evaluation: `current_state` was `None` on every
   batch. So despite the architecture having a proprioceptive input slot, it always sees zeros —
   the same category of gap `dom_retrieval` had before adding real `tool_positions`, but never fixed
   here.

## Comparison 1: each model on its own genuinely-held-out set

| Model | Test set | Position error (avg waypoint, mm) |
|---|---|---:|
| dom_retrieval (history full / cartesian_residual) | own 119-motion test set (9 episodes) | 24.38 |
| ACT-BeT chunk50 (final-epoch weights) | own val set (12 episodes, 2121 samples) | 58.52 (tool0 55.9 / tool1 61.2) |

## Comparison 2: rigorously matched, single held-out episode

Checked episode overlap between the two models' own splits before trusting comparison 1 further.
**6 of `dom_retrieval`'s 9 test episodes (25, 33, 36, 39, 43, 53) are in ACT-BeT's *training* set**
— evaluating ACT-BeT on `dom_retrieval`'s test set directly would have given ACT-BeT an unfair
advantage on those. Only `episode32_kugla` is genuinely held out from both models' training
(2 others, 22 and 61, aren't in ACT-BeT's dataset at all). Evaluated both models on just that
episode:

| Model | episode32 position error (mm) |
|---|---:|
| dom_retrieval | 21.82 (20 segments) |
| ACT-BeT chunk50 | 44.89 (208 overlapping chunk-starts, 18448 waypoints) |

**~2.06x advantage for dom_retrieval**, consistent with the broader (2.4x) comparison — confirms
comparison 1 wasn't an artifact of mismatched test sets.

## Caveats, stated plainly

- ACT-BeT's true best-epoch weights are unrecoverable (bug 1 above); evaluated final-epoch weights
  instead. Their own tracked `val_action_l1` was flat/slightly better at the final epoch (0.029 vs
  0.032 at best-epoch), so this is a reasonable stand-in, not a worst-case cherry-pick, but it isn't
  the literal best checkpoint that run produced.
- ACT-BeT's evaluation uses dense, overlapping 50-step chunks starting at every timestep (208
  chunk-starts for one episode); `dom_retrieval` uses 20 non-overlapping segments for the same
  episode. Shouldn't bias the mean, but the two aren't equally statistically independent.
- Different task formulations throughout: ACT-BeT predicts full 7D pose (position+quaternion) in
  overlapping 50-step chunks; `dom_retrieval` predicts 6D position-only single-shot trajectories.
  This comparison isolates just the position component for both.

Eval script and full stdout logs at `/workspace/runs/eval_act_bet_chunk50.py` and this
conversation's history on this host.
