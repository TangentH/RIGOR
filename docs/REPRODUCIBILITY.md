# Reproducing RIGOR

## Environment and hardware

The tested reconstruction environment is Linux with an NVIDIA RTX 4090
(24 GB), Python 3.11, PyTorch 2.7.1, and CUDA 12.8 wheels. Mask inference uses
a separate Python 3.10 environment. Installation requires no administrator
access:

```bash
./setup_external_repos.sh
PYTORCH_CUDA=12.8 ./setup_hilti_conda_envs.sh
./download_hilti_weights.sh --prefetch-hf
```

The NVIDIA driver must support the selected PyTorch CUDA runtime. The setup
uses binary wheels; a local CUDA compiler is not required with the default
masking settings. Other NVIDIA GPUs can use the same process-level workflow
when the installed PyTorch build supports their architecture and they have
sufficient memory. Their runtime and numerical equivalence have not been
benchmark-validated. CPU-only reconstruction is not a supported reproduction
path; CPU evaluation tools can be used separately.

One GPU is enough: assign `CUDA_VISIBLE_DEVICES=0` to the run process. With
additional GPUs, launch independent runs under `CUDA_VISIBLE_DEVICES=1`, etc.
Inside each process the selected card is device 0. A single run is not DDP;
memory from multiple cards is not combined. Under Slurm, keep the scheduler's
GPU visibility and allocation rather than selecting unallocated physical cards.

Changing GPU count must not change block size, overlap, sampling, masks, or
model parameters. If a device runs out of memory, use a larger-memory device
for the reference protocol; changing these parameters defines a new experiment.

## Assets and local configuration

Copy `hilti_paths.template.yaml` to the ignored `hilti_paths.local.yaml` and
set the local data, work, and output paths. Public sequences and trajectories
are distributed by the official benchmark. The private dense LiDAR clouds and
ROI annotations are not included and are needed for the reported geometry
evaluation. Public model downloads are pinned and checksum-verified.

```bash
conda run -n da3 python tools/reproducibility/preflight.py --hash-weights
conda run -n da3 python -m pip check
conda run -n gsam2 python -m pip check
```

Keep enough storage for extracted panoramas, masks, chunk predictions, and
the final point cloud. Required space depends on sequence length. Preserve the
work directory until the output and its provenance are validated.

## Configurations and experiment identity

The complete method uses
`tools/hilti_workflow/configs/rigor_paper_free_scale.yaml`. Its numerical
settings are explicit; the workflow writes the effective configuration and
input provenance with each result. The 30-run manifest contains public run
identities only. Output paths follow [OUTPUT_LAYOUT.md](OUTPUT_LAYOUT.md).

The repair comparison uses
`tools/hilti_workflow/configs/rigor_ablation_no_repair.yaml`. It retains rig
detection and validation, while disabling accepted pose/pointmap writes.
Removing the verifier's rig evidence as well would change loop acceptance
and would not isolate repair. Use a separate output root for every variant.
Interpret results according to the actual configuration recorded in their
manifest, rather than a directory name alone.

The current executable and archived experiments can differ in release tooling.
For exact scientific provenance, retain the source revision, resolved YAML,
checkpoint hashes, input ordering/timestamps, and per-run reports together.
Cross-device floating-point behavior can differ even with the same parameters;
bitwise equivalence across GPU architectures is not promised.

## Reconstruction and evaluation

See the root [README](../README.md) for single-run commands and the
[workflow guide](../tools/hilti_workflow/README.md) for preprocessing and
batch execution. Reconstruction is independent of GT.

The [evaluation guide](../tools/hilti_workflow/evaluation/README.md) describes
trajectory association, geometry registration/cropping, and retrieval ranking.
Each geometry variant must be independently registered, and uncertain
registrations reviewed before interpreting its metrics. Trajectory evaluation
and point-cloud registration are separate protocols.

## Tests

```bash
conda run -n da3 python -m pytest -q tools/hilti_workflow
conda run -n da3 python -m pytest -q da3_streaming/loop_utils/tests
```

These tests cover geometry, rig decisions, retrieval, configuration, output
provenance, and workflow recovery. Passing unit tests does not substitute for
running the benchmark or inspecting registration quality.
