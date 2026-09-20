# Request: run the VINN-RBF baseline on CUDA

Run the reference-only VINN-style complete-trajectory baseline for seeds 7, 17, and 27. This baseline retrieves the Top-100 training trajectories with an RBF kernel (bandwidth 0.15), returns their weighted pose-trajectory mean, and has no learned residual network.

## Code

The implementation is local commit `6bc4a49` from `dom_retrieval`. The direct push to `origin` failed because GitHub authentication was unavailable, so apply the attached patch from this repository:

```bash
cd /workspace/dom_retrieval
git status --short
git apply --check /workspace/claude-agents/dom_retrieval-6bc4a49-vinn-rbf.patch
git apply /workspace/claude-agents/dom_retrieval-6bc4a49-vinn-rbf.patch
```

If either repository lives elsewhere, adjust only the paths. Do not discard existing changes. If the patch conflicts, report the current `dom_retrieval` revision and the conflict instead of forcing it.

The added configuration is:

```text
configs/deformpath_vinn_rbf.yaml
```

Important settings:

```yaml
horizon: 32
points: 1024
history_length: 8
history_window_seconds: 0.7
pose_parameterization: orientation_delta
k: 100
retrieval_mode: rbf
retrieval_bandwidth: 0.15
retrieval_all: false
freeze_encoder: false
pretrain_epochs: 20
epochs: 1000
batch_size: 32
methods: [vinn_rbf]
```

## Verify before the full runs

```bash
cd /workspace/dom_retrieval
export PYTHONPATH=/workspace
python3 -m unittest \
  dom_retrieval.tests.test_models \
  dom_retrieval.tests.test_training -v
```

The focused local check passed all 32 tests. The local machine had no CUDA. A real-data one-epoch CPU smoke run loaded 611 motions (451 train, 41 validation, 119 test), wrote and reloaded a checkpoint, and confirmed:

- policy method: `vinn_rbf`;
- retriever: `RBFTopK(k=100, bandwidth=0.15)`;
- `model.residual is None`;
- memory size: 451 training demonstrations;
- validation and test demonstrations are absent from memory;
- reloaded one-epoch smoke result: 31.48 mm average waypoint error and 6.32 degrees average orientation error.

These smoke numbers are not final results.

## Train three seeds

Use a separate empty output directory for every seed:

```bash
cd /workspace/dom_retrieval
export PYTHONPATH=/workspace

for seed in 7 17 27; do
  python3 scripts/train.py \
    --config configs/deformpath_vinn_rbf.yaml \
    --seed "$seed" \
    --output "/workspace/checkpoints/vinn_rbf_seed${seed}"
done
```

Do not overwrite an existing checkpoint directory. If those names exist, choose new empty directories and report the paths used.

Each run must retain the split and dataset checks. Do not place validation or test trajectories in retrieval memory, bypass memory validation, or reuse a checkpoint with a different split or dataset fingerprint.

## Evaluate the selected checkpoints

Training already evaluates the best-validation model and writes metrics under `vinn_rbf/`. Also run the explicit checkpoint evaluator into empty directories:

```bash
cd /workspace/dom_retrieval
export PYTHONPATH=/workspace

for seed in 7 17 27; do
  python3 scripts/evaluate.py \
    --checkpoint "/workspace/checkpoints/vinn_rbf_seed${seed}/vinn_rbf/checkpoint.pt" \
    --output "/workspace/checkpoints/vinn_rbf_seed${seed}/evaluation" \
    --device cuda
done
```

## Required report

For every seed, report:

1. whether training completed or failed;
2. best epoch and early-stopping epoch;
3. train/validation/test recording and motion counts;
4. average and future average waypoint error in millimetres;
5. final waypoint error in millimetres;
6. average, future average, and final orientation error in degrees;
7. retrieval entropy and normalized retrieval entropy;
8. checkpoint and evaluation-output paths;
9. confirmation that `residual is None`, retrieval uses Top-100 RBF with bandwidth 0.15, and memory contains only training demonstrations.

Then calculate the three-seed mean and sample standard deviation and compare against the existing RBF-plus-residual result:

- position: **21.44 ± 1.09 mm**;
- orientation: **5.57 ± 0.17 degrees**.

When the recording splits match, also report paired per-recording VINN-RBF minus RBF-plus-residual differences. Positive error differences mean the residual model performed better. Do not claim superiority or equivalence without the completed held-out evaluations.
