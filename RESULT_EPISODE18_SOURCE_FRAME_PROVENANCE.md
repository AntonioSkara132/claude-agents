# Episode 18 source-frame provenance

The latest cross-session inspection resolves the frame identities enough to explain why the policy adapter cannot proceed, but does not provide a valid conversion.

## Training export

The `dom_retrieval` configuration uses:

```text
dom_retrieval/configs/history_deformpath.yaml
  data.root: Deformapth2_predownsampled1024_interpolated
  data.source_root: DeformPath2_release/DeformPath/DeformPath2Bags
```

Archived DeformPath2 metadata shows:

- `pose_reference_frame_mode: pointcloud`
- `pose_reference_frame: pointcloud_header_frame`
- archived `/camera/camera/depth/color/points` headers use `camera_depth_optical_frame`
- stored XYZ values are consumed unchanged by `dom_retrieval/data/dataset.py`
- the exporter only reverses model canonical coordinates and normalization; it does not convert sensor frames to mocap

Therefore the checkpoint training data is in `camera_depth_optical_frame` coordinates.

Useful hashes:

```text
training paths_interpolated.pt: 787dabdf3510ac0e7d4a8d8a97a9511635e6c66533fa5786082753aea6ec0962
training sequence_metadata.json: 57f3896fe6f527e76b5b46d7892e89366dd105d3d470dff640987edfcfe9e3bd
```

## Verified MPM source

The DeformPath3 / `relocated_episode18` metadata declares mocap output. Its point-cloud inputs are `camera_depth_optical_frame`, but the documented output transform is for the color optical frame:

```text
mocap_from_camera_color_optical_frame
```

No numeric `mocap_from_camera_depth_optical_frame` or verified depth-to-color/static camera-link transform has been found. Do not treat the color and depth optical frames as identical.

Useful hashes:

```text
relocated paths_interpolated.pt: a8600ddff186d6318e07afb37ba580d0ae2cc12fb1389c58afe21eb1b6e21e53
relocated sequence_metadata.json: 9b1598b2d50d622125ddf8c02add4aa673dd1d7d781ad43ee9dbd94d363f2ea9
```

## Decision

The two exports have matching timestamps but different coordinate frames and approximately 13 cm position differences. The existing metadata is insufficient to convert the policy output safely. The empirical rigid fit is diagnostic only and must not become the simulator calibration. `tool-friction-sweep` also tested it against point clouds: nearest-neighbor median residual was 50.4 mm and median cloud-center residual was 92.9 mm. The documented DeformPath3 color-optical transform performed much better (3.88 mm and 17.2 mm respectively), but it still cannot be applied to depth optical without the missing static depth/color/link transforms. The Kabsch result therefore likely recovers a legacy tool-label conversion rather than a physical camera calibration.

Before paired MPM policy conditions run, obtain one of:

1. the original bag's timestamped/static transform for `camera_depth_optical_frame` to mocap;
2. matching mocap-frame training data and a checkpoint trained from it;
3. a verified depth-to-color transform plus the documented mocap/color transform, with validation over all 388 tool-pose frames.

Record the exact transform provenance, both input hashes, quaternion composition order, and residual statistics. Otherwise keep the paired run blocked.
