# Evaluation

This directory contains the evaluation code used for the paper. Generated
reports, reconstructed point clouds, and ground-truth assets are intentionally
kept outside the repository.

## Trajectory evaluation

`evaluate_da3_against_gt.py` collapses the four yaw views that share a capture
timestamp into one camera center, associates those centers with the public
benchmark trajectory, and reports rigidly aligned trajectory errors. Exact
evaluation requires the rendered image names because they carry the ROS
timestamps.

```bash
python tools/hilti_workflow/evaluation/evaluate_da3_against_gt.py \
  --camera-poses /path/to/reconstruction/camera_poses.txt \
  --image-dir /path/to/pinhole_yaw4_imu \
  --ground-truth /path/to/groundtruth.txt \
  --run-name floor_1_2025-05-05_run_1 \
  --output-dir /path/to/evaluation \
  --modes rigid,sim3 \
  --rpe-horizon 10.0
```

The timestamp approximation flags are intended only for diagnostics and must
not be used in place of timestamp-exact paper evaluation. The paper reports
the translational RPE at the 10-second horizon shown explicitly above.

## Geometry evaluation

The paper geometry numbers use `register_trajectory_sim3_icp.py`. For each
method output, it:

1. recovers physical capture centres from the exact timestamps in the four
   rendered-view filenames;
2. fits a trajectory-to-ground-truth Sim(3), requiring at least 99% temporal
   coverage after the benchmark's initial five-second exclusion;
3. applies that initializer to the reconstruction; and
4. proposes a scale-adjusting ICP update against the ROI LiDAR cloud.

The ICP proposal uses deterministic samples and the archived thresholds. It is
selected only if symmetric trimmed RMSE improves by at least 0.5%, source
fitness at 0.5 m is at least 5%, residual translation and rotation are no more
than 2 m and 12 degrees, and relative scale lies in [0.5, 2]. Otherwise the
trajectory initializer is retained. The script records both candidates, the
selected transform, input hashes, timestamp coverage, and every gate value.
Ground truth is used only in this evaluation step.

```bash
python tools/hilti_workflow/evaluation/register_trajectory_sim3_icp.py \
  --source-cloud /path/to/reconstruction.ply \
  --camera-poses /path/to/camera_poses.txt \
  --image-dir /path/to/pinhole_yaw4_imu \
  --ground-truth-trajectory /path/to/groundtruth.txt \
  --ground-truth-cloud /path/to/roi_gt.ply \
  --output-dir /path/to/registered_run \
  --run-name example_run
```

If the trajectory and dense LiDAR use different reference frames, provide the
known 4x4 mapping with `--gt-trajectory-to-geometry`. A retained tar archive of
timestamped images can replace `--image-dir` via `--timestamp-archive`. Run the
entrypoint separately for every independently evaluated system. The optional
`--icp-reference-cloud` exists only to reproduce an explicitly declared
shared-gauge controlled comparison; it must not be used silently.

After paper-protocol registration:

1. `crop_aligned_regions.py` partitions each selected aligned cloud into
   inside-ROI and outside-ROI points using locally supplied ROI metadata.
2. `run_geometry_metrics.py` computes symmetric Chamfer distance,
   bidirectional distance statistics, RMSE, Hausdorff distance, and precision,
   recall, and F-score at the requested thresholds. Full-cloud evaluation is
   the default; deterministic subsampling is opt-in.
3. `aggregate_geometry_metrics.py` combines per-run JSON reports without
   discarding their individual provenance.

## Retrieval evaluation

`evaluate_salad_retrieval_ablation.py` compares single-view maximum similarity
and cyclic four-view scoring from the same cached SALAD descriptors. Public
ground-truth trajectories are used only after retrieval to label eligible pairs;
they do not affect descriptor extraction or ranking.

## Data policy

The benchmark trajectories are available from the public challenge release.
The dense LiDAR clouds and ROI annotations used in the paper are private and
are not redistributed. Supply equivalent assets through local paths. Do not
commit data, registrations, metrics, logs, or machine-specific paths.

Run the maintained regression tests with:

```bash
python -m pytest -q tools/hilti_workflow/evaluation/tests
```
