# Output layout

Every method uses:

```text
<output_root>/<method>/<floor>/<date>/run_N/reconstruction/
```

For `run_method.py`, `--output-root` is the shared root and the runner appends
the method name. For the method-specific batch entrypoints
`run_hilti_batch.py` and `run_panovggt_batch.py`, `--output-root` already
denotes the corresponding `da3` or `panovggt` directory.

`method` is `da3` or `panovggt`. Use distinct output roots for different
configurations or ablations.

## Reconstruction

| File | Content |
|---|---|
| `reconstruction.ply` | Final predicted point cloud |
| `camera_poses.txt` | Row-major 4×4 camera-to-world matrices |
| `camera_poses.ply` | Predicted camera centers |
| `workflow_manifest.json` | Inputs, settings, and method provenance |
| `logs/` | Retained stage logs and decision diagnostics |
| `logs/run_checkpoint.json` | Latest workflow checkpoint, when using the ROS-bag wrapper |
| `logs/run_status.jsonl` | Append-only stage history, when using the ROS-bag wrapper |

The point cloud and poses share the same reconstruction frame. Timestamped
input names are required for trajectory evaluation. DA3 loop and rig reports
remain with the work logs; they describe why a proposed measurement or local
repair was accepted or rejected.

Input frames, masks, and model chunks belong in the configured work directory.
Keep them for parameter replay and interrupted-run recovery. Before reusing a
completed result, verify its input and effective-configuration provenance.

## Evaluation

Evaluation has a separate root, preserving the unregistered reconstruction:

```text
<evaluation_root>/<method>/<floor>/<date>/run_N/reconstruction/
  aligned.ply
  aligned_roi.ply
  aligned_outside_roi.ply
  registration_report.json
  transform_selected.txt
  roi_crop_report.json
  metrics/
```

Each method variant is registered independently. The selected evaluation
transform, initialization, scale adjustment, ICP decision, and ROI partition
must remain traceable in the reports. Failed or uncertain alignments require
review; the existence of an aligned file alone is not proof of correct
registration. GT registration must never overwrite `reconstruction.ply`.

Generated data, metrics, logs, and machine-local settings stay outside Git.
