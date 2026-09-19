# Available Episode 18 sequence archive

Archive: `episode18_kugla_available_sequence.tar.gz`

- Source directory: `data/deformpath_training/preprocessed_dataset/episode18_kugla`
- Compressed size: 55 MB
- SHA-256: `1adbcff74a2c714e084476dc3faab3285924bc25e4cf989474798ad57c7d1ad8`

The archive contains the available Episode 18 paths, point clouds, calibration and manifests. It is the local counterpart of CUDA's `/workspace/data/preprocessed_dataset/episode18_kugla` copy.

**Important:** CUDA has already established that this sequence does not match the expected Episode 18 sequence fingerprint in the reconstructed-MPM request, despite matching individual calibration/geometry file hashes. Use it to inspect or reproduce the mismatch, not as evidence that it is the exact calibration-validated sequence. Keep the fingerprint check active for the policy-replay experiment.

Extract with:

```bash
tar -xzf episode18_kugla_available_sequence.tar.gz
```
