#!/usr/bin/env python3
"""Check whether a checkout has the assets needed for the HILTI pipeline."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
EXPECTED_WEIGHTS = {
    "da3_streaming/weights/config.json": "09adf89474017e717bc05aa86fd3a378708ba8914b036d61874eced328069468",
    "da3_streaming/weights/model.safetensors": "8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c",
    "da3_streaming/weights/dino_salad.ckpt": "6b3f1720954293e83da6966c5cfcfc6713200d7fefadcca76fc51aeb80b3cada",
    "Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt": "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
    ".external/PanoVGGT/checkpoints/model.pt": "4adab888064ef206c20ab42c08c5f973e3bd98a813fa2c3d849b4d445e278670",
}
PANOVGGT_COMMIT = "556bb7d2ec2d02bd3ee4ed535542e74290ba22cf"
REQUIRED_OUTPUTS = (
    "reconstruction.ply",
    "camera_poses.txt",
    "camera_poses.ply",
    "workflow_manifest.json",
)


def read_paths(path: Path) -> dict[str, Path]:
    values: dict[str, Path] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        text = value.strip().strip("'\"").format(repo=REPO, user=getpass.getuser())
        candidate = Path(text).expanduser()
        values[key.strip()] = candidate if candidate.is_absolute() else (REPO / candidate).resolve()
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_results_root(
    explicit_results_root: Path | None,
    selected_methods: list[str],
    path_settings: dict[str, Path],
) -> Path | None:
    """Return a results root only when output validation was requested."""
    if explicit_results_root is None and not selected_methods:
        return None
    return (
        explicit_results_root
        or path_settings.get("results_root")
        or path_settings.get("output_root")
    )


def missing_result_outputs(
    results_root: Path,
    methods: list[str],
    runs: list[dict[str, object]],
) -> list[str]:
    missing: list[str] = []
    for method in methods:
        for run in runs:
            reconstruction = (
                results_root
                / method
                / str(run["relative_path"])
                / "reconstruction"
            )
            for name in REQUIRED_OUTPUTS:
                if not (reconstruction / name).is_file():
                    missing.append(f"{method}:{run['run_name']}:{name}")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_config = REPO / ("hilti_paths.local.yaml" if (REPO / "hilti_paths.local.yaml").exists() else "hilti_paths.yaml")
    parser.add_argument("--paths-config", type=Path, default=default_config)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--method", action="append", choices=("da3", "panovggt"), default=[])
    parser.add_argument("--hash-weights", action="store_true", help="Read and SHA-256 all model weights.")
    parser.add_argument("--skip-envs", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true", help="Only check assets/imports; do not claim inference readiness.")
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []

    def check(condition: bool, label: str, *, warning: bool = False) -> None:
        status = "PASS" if condition else ("WARN" if warning else "FAIL")
        print(f"[{status}] {label}")
        if not condition:
            (warnings if warning else failures).append(label)

    check((REPO / "README.md").is_file(), "repository root is accessible")
    salad_files = (
        REPO / "da3_streaming/loop_utils/salad/models/helper.py",
        REPO / "da3_streaming/loop_utils/salad/models/backbones/dinov2.py",
        REPO / "da3_streaming/loop_utils/salad/models/aggregators/salad.py",
    )
    check(all(path.is_file() for path in salad_files), "vendored SALAD inference source is present")
    check((REPO / "device_mask_final.png").is_file(), "static device mask is tracked")

    pano_root = REPO / ".external/PanoVGGT"
    pano_revision = subprocess.run(
        ["git", "-C", str(pano_root), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=False,
    )
    check(
        pano_revision.returncode == 0 and pano_revision.stdout.strip() == PANOVGGT_COMMIT,
        "official PanoVGGT checkout is at the pinned commit",
    )

    path_settings = read_paths(args.paths_config.expanduser())

    run_manifest = REPO / "tools/hilti_workflow/manifests/hilti_all_runs.json"
    runs = json.loads(run_manifest.read_text(encoding="utf-8"))["runs"]
    check(len(runs) == 30, f"full-run manifest contains 30 runs (found {len(runs)})")

    for relative, expected in EXPECTED_WEIGHTS.items():
        path = REPO / relative
        check(path.is_file(), f"weight exists: {relative}")
        if path.is_file() and args.hash_weights:
            actual = sha256(path)
            check(actual == expected, f"weight checksum: {relative}")

    if not args.skip_envs:
        env_checks = {
            "da3": "import cv2,faiss,hydra,kornia,numpy,open3d,pypose,rosbags,torch,yacs; import depth_anything_3",
            "gsam2": "import cv2,sam2,torch,transformers",
        }
        for env_name, snippet in env_checks.items():
            result = subprocess.run(
                ["conda", "run", "-n", env_name, "python", "-c", snippet],
                cwd=REPO,
                text=True,
                capture_output=True,
            )
            check(result.returncode == 0, f"Conda environment imports: {env_name}")
            if result.returncode:
                print(result.stderr.strip())

    results_root = select_results_root(args.results_root, args.method, path_settings)
    if results_root is not None and results_root.exists():
        methods = args.method or ["da3", "panovggt"]
        missing = missing_result_outputs(results_root, methods, runs)
        check(not missing, f"selected method results complete ({len(missing)} missing)")
        if missing:
            print("  " + "\n  ".join(missing[:20]))
    elif results_root is not None:
        check(False, f"results root is unavailable: {results_root}", warning=True)

    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        )
        if gpu.returncode == 0:
            print("[INFO] GPU: " + gpu.stdout.strip())
        else:
            print("[INFO] nvidia-smi did not report a GPU")
    except FileNotFoundError:
        print("[INFO] nvidia-smi not found")

    if not args.skip_gpu:
        probe = subprocess.run(
            ["conda", "run", "-n", "da3", "python", "-c",
             "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
             "x=torch.arange(1024,device='cuda',dtype=torch.float32); "
             "assert float((x*x).sum().cpu()) == 357389824.0; "
             "print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),"
             "torch.cuda.device_count())"],
            text=True, capture_output=True, check=False,
        )
        check(probe.returncode == 0, "visible GPU executes a CUDA tensor operation")
        if probe.stdout.strip():
            print("[INFO] CUDA runtime: " + probe.stdout.strip())
        if probe.returncode:
            print(probe.stderr.strip())

    print(f"\nsummary: {len(failures)} failures, {len(warnings)} warnings")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
