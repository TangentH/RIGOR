# Third-party components

This repository contains project-specific workflow code plus selected public
upstream components needed to run it. Copyright notices, licenses, and URLs
inside third-party paths identify their respective upstream projects.

- `src/depth_anything_3/`, `da3_streaming/`, and the root `LICENSE`: Depth
  Anything 3 and adapted DA3-Streaming/VGGT-Long components.
- `Grounded-SAM-2/`: the SAM 2 runtime and Grounding DINO integration. Large
  upstream notebooks, demos, training utilities, and showcase media are not
  included.
- `da3_streaming/loop_utils/salad/`: the minimal inference subset of the public
  [SALAD repository](https://github.com/serizba/salad) at commit
  `6aede13a3f6c25750bf7fde10209c06cb73060bb`, used by loop retrieval. Its GPL-3.0
  license is retained in that directory; training and benchmark assets are omitted.
- `hilti-trimble-slam-challenge-2026/`: public benchmark calibration,
  floor-plan, and trajectory resources. Unused showcase media and the Stella
  vocabulary are not included.
- PanoVGGT is fetched at the fixed public revision recorded in
  `setup_external_repos.sh` and remains an external dependency.

Model weights and datasets are intentionally excluded and are obtained from
their public upstream sources where available.
