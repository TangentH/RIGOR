# HILTI workflow tools

Stable user-facing entrypoints:

```text
run_method.py                         Shared DA3/PanoVGGT single-run launcher
run_hilti_rosbag_to_reconstruction.py DA3 single ROS-bag workflow
run_hilti_batch.py                    DA3 manifest scheduler
run_hilti_best_recon_workflow.py      DA3 reconstruction/postprocess backend
run_panovggt_rosbag_to_reconstruction.py PanoVGGT restart-safe ROS-bag workflow
run_panovggt_batch.py                 PanoVGGT one-process/one-GPU scheduler
```

Production modules:

```text
configs/                Curated portable YAML configurations
extraction/             ROS bag to equirectangular panoramas
view_generation/        IMU levelling and perspective-view generation
postprocess/            Mask and point-cloud cleanup
panovggt/               Native ERP inference, Sim(3) stitching, mask utilities
evaluation/             GT registration and metrics (outputs stay external)
manifests/              Public run identity manifests only
tests/                  Workflow policy/unit tests
```

The production tree must not import code from `docs/`, generated artifacts, or
machine-local paths. One-off scripts belong outside the repository or in an
ignored local `tools/hilti_workflow/experimental/` directory.

Both maintained methods write:

```text
<output_root>/<method>/<floor>/<date>/run_N/reconstruction/
```

A single 24 GB GPU is sufficient for one run; the selected card is bound
outside the process and is seen as device 0:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n da3 python \
  tools/hilti_workflow/run_method.py da3 \
  --floor floor_1 --date 2025-05-05 --run 1 \
  --rosbag /path/to/rosbag.db3 --output-root /path/to/outputs \
  --no-delete-rosbag-on-success
```

The shipped 30-run manifest can be scheduled by one worker:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n da3 python \
  tools/hilti_workflow/run_hilti_batch.py \
  --source-data-root /path/to/local/data --output-root /path/to/outputs/da3 \
  --worker-index 0 --num-workers 1 --worker-id gpu0
```

For multiple GPUs, start one independent command per card with distinct
`--worker-index` values and the same `--num-workers`; no DDP or fixed GPU count
is assumed. The scheduler reads local bags only and never downloads or deletes
them. The exact output contract is in `docs/OUTPUT_LAYOUT.md`.

The paper configuration uses 768x512 yaw-four views at 95-degree horizontal
FoV, spaced by 90 degrees, with complementary-filter levelling (`tau=2`).
Sequential DA3 inference uses 32 images with 16-image overlap. The complete
system is defined by `configs/rigor_paper_free_scale.yaml`; the controlled
repair ablation is `configs/rigor_ablation_no_repair.yaml`.

Public trajectory evaluation uses the trajectories distributed by the
challenge. Dense LiDAR geometry and ROI annotations are not redistributed;
users running geometry evaluation must supply compatible assets locally.
`DA3-Sequential` in the paper is the upstream DA3 streaming control without a
non-local graph update. `DA3-Legacy` is a historical same-backbone predecessor,
not a one-component ablation. See [BASELINES.md](BASELINES.md) for the DA3-Legacy preset and the
cache-based DA3-Sequential construction; neither is approximated by an
arbitrary one-flag change to RIGOR.
