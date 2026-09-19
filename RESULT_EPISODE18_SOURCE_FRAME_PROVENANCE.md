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

## Branch-level root cause

`mpm-adapter-scan` traced the exporter branch logic in `export_deformpath2_offline.py:949-1017` and confirmed that the two files were produced by different output-frame paths. The DeformPath2 training export predates the later output-frame option and transforms poses into the point cloud's camera optical frame. The verified DeformPath3 export explicitly selects `output_frame: mocap` and interpolates poses directly in mocap coordinates. The shared `pose_reference_frame_mode: pointcloud` field is unused by the mocap branch and must not be used to infer that both exports share a frame.

The approximately 13 cm difference is therefore a camera-optical-frame versus mocap-frame mismatch, not evidence that two mocap calibrations disagree. The empirical rigid fit remains diagnostic only. Its poor point-cloud alignment (nearest-neighbor median 50.4 mm; cloud-center median 92.9 mm) confirms it is not a physical camera calibration.

## Decision

The frame conversion is now clear, but its exact static camera extrinsic is still required. Because the DeformPath2 target frame is `camera_depth_optical_frame`, recover the original bag's exact `camera_link -> camera_depth_frame -> camera_depth_optical_frame` chain, compose it with the archived DeformPath2 camera-link-to-mocap calibration, invert the result to map depth-optical predictions into mocap, and validate all 388 tool poses and point clouds against `relocated_episode18`. The color-optical chain may be used only as an independent check. Do not substitute the color transform, nominal optical rotation, or an empirical Kabsch fit.

Before paired MPM policy conditions run, obtain one of:

1. the original bag's exact static camera transforms;
2. matching mocap-frame training data and a checkpoint trained from it;
3. a verified camera calibration bundle containing the required static extrinsic.

Record the exact transform provenance, both input hashes, quaternion composition order, and residual statistics. Otherwise keep the paired run blocked.

Before paired MPM policy conditions run, obtain one of:

1. the original bag's timestamped/static transform for `camera_depth_optical_frame` to mocap;
2. matching mocap-frame training data and a checkpoint trained from it;
3. a verified depth-to-color transform plus the documented mocap/color transform, with validation over all 388 tool-pose frames.

Record the exact transform provenance, both input hashes, quaternion composition order, and residual statistics. Otherwise keep the paired run blocked.

## Diagnostic rigid fit (not a physical calibration)

`tool-friction-sweep` fitted the training tool labels in `camera_depth_optical_frame` to the relocated tool labels in `mocap`, excluding anomalous frame 272. Use the convention `T_A_B` maps coordinates from frame B into frame A.

```text
T_M_DO =
[ [ 0.003680170,  0.997599284, -0.069152912, 0.260031328 ],
  [ 0.046511497, -0.069249300, -0.996514533, 0.525863257 ],
  [-0.998910975,  0.000450937, -0.046654685, 0.853676561 ],
  [ 0.000000000,  0.000000000,  0.000000000, 1.000000000 ] ]
```

Equivalent XYZW quaternion: `[0.52905202, 0.49338758, -0.50470646, 0.47110938]`.

The fit retained 774/776 pose pairs, with 0.569 mm position RMS, 1.221 mm position p95, 0.154 degrees orientation RMS, and 0.319 degrees orientation p95. It is diagnostic only: applying it to point clouds gives 50.4 mm nearest-neighbor median and 92.9 mm cloud-center median residuals. Do not use it as the simulator transform.
