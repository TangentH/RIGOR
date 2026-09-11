#!/usr/bin/env python3
"""Run the full HILTI pipeline from one ROS2 bag to final reconstruction assets.

The script intentionally keeps DA3 and Grounded-SAM-2 in separate conda
environments. Intermediate images and the DA3 work directory are retained by
default for restart and parameter replay; explicit negative flags clean them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


REPO = Path(__file__).resolve().parents[2]
HILTI_REPO = REPO / "hilti-trimble-slam-challenge-2026"
GSAM2_REPO = REPO / "Grounded-SAM-2"
BEST_WORKFLOW = REPO / "tools/hilti_workflow/run_hilti_best_recon_workflow.py"
BEST_WORKFLOW_CONFIG = REPO / "tools/hilti_workflow/configs/rigor_paper_free_scale.yaml"


ARG_DEFAULTS = {
    "extract_stride": 10,
    "max_equirect_frames": 0,
    "extract_progress_interval_s": 5.0,
    "pinhole_progress_interval_s": 5.0,
    "pinhole_width": 768,
    "pinhole_height": 512,
    "pinhole_fov_deg": 95.0,
    "yaws": "0,90,180,270",
    "imu_tau": 2.0,
    "imu_method": "complementary",
    "accel_gate_sigma": 0.2,
    "time_offset_ns": 0,
    "mask_prompt": "person.helmet.",
    "box_threshold": 0.3,
    "text_threshold": 0.3,
    "mask_close_pixels": 3,
    "mask_dilate_pixels": 5,
    "grounding_batch_size": 2,
    "hf_grounding_model": "IDEA-Research/grounding-dino-base",
    "hf_grounding_revision": "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
}

STATE_FORMAT_VERSION = 1
CHECKPOINT_NAME = "run_checkpoint.json"
STATUS_JSONL_NAME = "run_status.jsonl"
FRONTEND_STAGE_NAMES = (
    "extract_equirect",
    "generate_pinhole_yaw4_imu",
    "gsam2_masks",
)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON state file without exposing partial content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        with temporary.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def required_file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    return {
        "size_bytes": int(stat.st_size),
        "sha256": file_sha256(resolved),
    }


def _listed_output_files(directory: Path, suffixes: set[str]) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def filename_set_signature(
    directory: Path,
    suffixes: set[str],
    *,
    require_nonempty: bool,
) -> dict[str, Any]:
    paths = _listed_output_files(directory, suffixes)
    empty = [path.name for path in paths if path.stat().st_size <= 0]
    if empty:
        raise RuntimeError(
            f"Empty stage outputs in {directory}: {empty[:5]}"
        )
    if require_nonempty and not paths:
        raise RuntimeError(f"No stage outputs in {directory}")
    entries: list[dict[str, Any]] = []
    for path in paths:
        before = path.stat()
        digest = file_sha256(path)
        after = path.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ino != after.st_ino
        ):
            raise RuntimeError(
                f"Stage output changed while hashing: {path}"
            )
        entries.append(
            {
                "name": path.name,
                "size_bytes": int(after.st_size),
                "sha256": digest,
            }
        )
    names = [entry["name"] for entry in entries]
    return {
        "count": len(paths),
        "filename_set_sha256": canonical_json_sha256(names),
        "content_tree_sha256": canonical_json_sha256(entries),
        "content_tree_format": "sorted-name-size-sha256-v1",
        "total_size_bytes": sum(entry["size_bytes"] for entry in entries),
        "all_nonempty": bool(paths) and not empty,
        "first_name": names[0] if names else None,
        "last_name": names[-1] if names else None,
    }


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def validate_output_signature(
    payload: Any, *, require_nonempty: bool
) -> list[str]:
    problems: list[str] = []
    if not isinstance(payload, dict):
        return ["outputs are not an object"]
    try:
        count = int(payload.get("count", -1))
        total_size = int(payload.get("total_size_bytes", -1))
    except (TypeError, ValueError):
        return ["output count or total size is invalid"]
    if count < 0 or (require_nonempty and count <= 0):
        problems.append("output count is not valid")
    if total_size < 0 or (require_nonempty and total_size <= 0):
        problems.append("output total size is not valid")
    if payload.get("all_nonempty") is not (count > 0):
        problems.append("all_nonempty does not match output count")
    if not _is_sha256(payload.get("filename_set_sha256")):
        problems.append("filename-set SHA-256 is invalid")
    if not _is_sha256(payload.get("content_tree_sha256")):
        problems.append("content-tree SHA-256 is invalid")
    if payload.get("content_tree_format") != "sorted-name-size-sha256-v1":
        problems.append("content-tree format is invalid")
    if count:
        if not isinstance(payload.get("first_name"), str):
            problems.append("first output name is invalid")
        if not isinstance(payload.get("last_name"), str):
            problems.append("last output name is invalid")
    return problems


def validate_stage_record(
    payload: Any,
    *,
    expected_stage: str,
    expected_request: dict[str, Any] | None = None,
    expected_outputs: dict[str, Any] | None = None,
) -> list[str]:
    """Validate one complete or resumable stage record without trusting its hash."""
    if not isinstance(payload, dict):
        return [f"{expected_stage}: stage record is not an object"]
    problems: list[str] = []
    if payload.get("schema_version") != 1:
        problems.append(f"{expected_stage}: schema_version is not 1")
    if payload.get("stage") != expected_stage:
        problems.append(f"{expected_stage}: stage name mismatch")
    status = payload.get("status")
    if status not in {"in_progress", "complete", "cli_skipped"}:
        problems.append(f"{expected_stage}: invalid status")

    if status == "cli_skipped":
        problems.extend(
            f"{expected_stage}: {problem}"
            for problem in validate_output_signature(
                payload.get("outputs"), require_nonempty=False
            )
        )
        return problems

    request = payload.get("request")
    if not isinstance(request, dict):
        problems.append(f"{expected_stage}: request is not an object")
    else:
        request_sha256 = payload.get("request_sha256")
        if not _is_sha256(request_sha256):
            problems.append(f"{expected_stage}: request SHA-256 is invalid")
        elif request_sha256 != canonical_json_sha256(request):
            problems.append(f"{expected_stage}: request SHA-256 mismatch")
        if expected_request is not None and request != expected_request:
            problems.append(f"{expected_stage}: request does not match invocation")

    if status == "complete":
        outputs = payload.get("outputs")
        problems.extend(
            f"{expected_stage}: {problem}"
            for problem in validate_output_signature(
                outputs, require_nonempty=True
            )
        )
        if expected_outputs is not None and outputs != expected_outputs:
            problems.append(f"{expected_stage}: outputs do not match current files")
    elif expected_outputs is not None:
        problems.append(f"{expected_stage}: stage is not complete")
    return problems


def validate_frontend_stage_chain(
    stages: dict[str, dict[str, Any]],
) -> list[str]:
    """Validate embedded records and connect every available upstream digest."""
    problems: list[str] = []
    for stage in FRONTEND_STAGE_NAMES:
        record = stages.get(stage)
        problems.extend(
            validate_stage_record(record, expected_stage=stage)
        )

    extraction = stages.get("extract_equirect", {})
    pinhole = stages.get("generate_pinhole_yaw4_imu", {})
    masks = stages.get("gsam2_masks", {})
    extraction_outputs = extraction.get("outputs")
    pinhole_outputs = pinhole.get("outputs")
    pinhole_request = pinhole.get("request")
    masks_request = masks.get("request")
    if isinstance(pinhole_request, dict):
        if pinhole_request.get("equirect_outputs") != extraction_outputs:
            problems.append(
                "generate_pinhole_yaw4_imu: extraction output digest is not chained"
            )
    elif pinhole.get("status") == "complete":
        problems.append(
            "generate_pinhole_yaw4_imu: request is unavailable for upstream validation"
        )
    if isinstance(masks_request, dict):
        if masks_request.get("pinhole_outputs") != pinhole_outputs:
            problems.append("gsam2_masks: pinhole output digest is not chained")
    elif masks.get("status") == "complete":
        problems.append(
            "gsam2_masks: request is unavailable for upstream validation"
        )
    return problems


def _stage_manifest_path(logs_dir: Path, stage: str) -> Path:
    return logs_dir / f"stage_{stage}_provenance.json"


def _load_stage_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _expected_names_match(
    directory: Path,
    suffixes: set[str],
    expected_names: set[str] | None,
    *,
    allow_subset: bool,
) -> bool:
    if expected_names is None:
        return True
    actual = {path.name for path in _listed_output_files(directory, suffixes)}
    return actual <= expected_names if allow_subset else actual == expected_names


def prepare_stage_provenance(
    *,
    logs_dir: Path,
    stage: str,
    request: dict[str, Any],
    output_dir: Path,
    suffixes: set[str],
    expected_names: set[str] | None = None,
    force: bool,
) -> tuple[bool, dict[str, Any] | None]:
    """Return whether a complete, exactly matching stage can be reused.

    A matching in-progress record permits restart of the same request. Existing
    files without that record, or files from a different request, are rejected
    unless the caller explicitly selected ``--force``.
    """
    manifest_path = _stage_manifest_path(logs_dir, stage)
    request_sha256 = canonical_json_sha256(request)
    current = filename_set_signature(
        output_dir, suffixes, require_nonempty=False
    )
    manifest = _load_stage_manifest(manifest_path)
    manifest_problems = validate_stage_record(
        manifest,
        expected_stage=stage,
        expected_request=request,
    )
    request_matches = bool(
        manifest
        and not manifest_problems
        and manifest.get("request_sha256") == request_sha256
    )
    if current["count"]:
        complete_match = bool(
            request_matches
            and manifest.get("status") == "complete"
            and manifest.get("outputs") == current
            and _expected_names_match(
                output_dir,
                suffixes,
                expected_names,
                allow_subset=False,
            )
        )
        if complete_match:
            return True, manifest
        resumable_partial = bool(
            request_matches
            and manifest.get("status") == "in_progress"
            and _expected_names_match(
                output_dir,
                suffixes,
                expected_names,
                allow_subset=True,
            )
        )
        if not force and not resumable_partial:
            raise RuntimeError(
                f"Refusing to relabel stale {stage} artifacts in {output_dir}; "
                "the requested input/protocol or exact output set differs. "
                "Use --force only after confirming this generated directory."
            )
    intent = {
        "schema_version": 1,
        "stage": stage,
        "status": "in_progress",
        "request_sha256": request_sha256,
        "request": request,
        "started_at_unix": time.time(),
    }
    atomic_write_json(manifest_path, intent)
    return False, intent


def complete_stage_provenance(
    *,
    logs_dir: Path,
    stage: str,
    request: dict[str, Any],
    output_dir: Path,
    suffixes: set[str],
    expected_names: set[str] | None = None,
) -> dict[str, Any]:
    if not _expected_names_match(
        output_dir, suffixes, expected_names, allow_subset=False
    ):
        actual = {path.name for path in _listed_output_files(output_dir, suffixes)}
        missing = sorted((expected_names or set()) - actual)
        extra = sorted(actual - (expected_names or set()))
        raise RuntimeError(
            f"{stage} output filename set mismatch: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    outputs = filename_set_signature(
        output_dir, suffixes, require_nonempty=True
    )
    payload = {
        "schema_version": 1,
        "stage": stage,
        "status": "complete",
        "request_sha256": canonical_json_sha256(request),
        "request": request,
        "outputs": outputs,
        "completed_at_unix": time.time(),
    }
    problems = validate_stage_record(
        payload,
        expected_stage=stage,
        expected_request=request,
        expected_outputs=outputs,
    )
    if problems:
        raise RuntimeError(
            f"Refusing to publish invalid {stage} provenance: "
            + "; ".join(problems)
        )
    atomic_write_json(_stage_manifest_path(logs_dir, stage), payload)
    return payload


def expected_pinhole_names(
    equirect_dir: Path, yaws_text: str
) -> set[str]:
    yaws = [float(value.strip()) for value in yaws_text.split(",") if value.strip()]
    if not yaws:
        raise ValueError("At least one yaw is required")
    equirect_paths = _listed_output_files(
        equirect_dir, {".jpg", ".jpeg", ".png", ".webp"}
    )
    return {
        f"{frame_idx:05d}_v{view_idx:02d}_{path.stem}_yaw"
        f"{round(yaw_deg) % 360:03d}.jpg"
        for frame_idx, path in enumerate(equirect_paths)
        for view_idx, yaw_deg in enumerate(yaws)
    }


def frontend_protocol_manifest(
    args: argparse.Namespace,
    bag: Path,
    stages: dict[str, dict[str, Any]] | None = None,
    bag_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = args.workflow_config.expanduser().resolve()
    resolved_bag_identity = bag_identity or required_file_identity(bag)
    resolved_stages = stages or {}
    if stages is not None:
        problems = validate_frontend_stage_chain(resolved_stages)
        if problems:
            raise RuntimeError(
                "Refusing to publish invalid frontend stage chain: "
                + "; ".join(problems)
            )
    return {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "workflow_config": str(config),
        "workflow_config_sha256": file_sha256(config),
        "rosbag": str(bag.resolve()),
        "rosbag_sha256": resolved_bag_identity["sha256"],
        "settings": {
            "extract_stride": args.extract_stride,
            "max_equirect_frames": args.max_equirect_frames,
            "extract_progress_interval_s": args.extract_progress_interval_s,
            "pinhole_progress_interval_s": args.pinhole_progress_interval_s,
            "pinhole_width": args.pinhole_width,
            "pinhole_height": args.pinhole_height,
            "pinhole_fov_deg": args.pinhole_fov_deg,
            "yaws": args.yaws,
            "imu_tau": args.imu_tau,
            "imu_method": args.imu_method,
            "accel_gate_sigma": args.accel_gate_sigma,
            "time_offset_ns": args.time_offset_ns,
            "use_yaml_timeshift": args.use_yaml_timeshift,
            "rotate180": args.rotate180,
            "mask_prompt": args.mask_prompt,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "mask_close_pixels": args.mask_close_pixels,
            "mask_dilate_pixels": args.mask_dilate_pixels,
            "grounding_batch_size": args.grounding_batch_size,
            "hf_grounding_model": args.hf_grounding_model,
            "hf_grounding_revision": args.hf_grounding_revision,
        },
        "stages": resolved_stages,
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Durably append one complete JSON object as a single JSONL record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o644,
    )
    with os.fdopen(descriptor, "ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def load_run_checkpoint(path: Path) -> dict[str, Any]:
    """Load a compatible checkpoint; malformed state is ignored safely."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    if payload.get("format_version") != STATE_FORMAT_VERSION:
        return {}
    return payload


class RunProgress:
    """Persistent single-run stage state with an append-only audit trail."""

    def __init__(
        self,
        logs_dir: Path,
        *,
        bag: Path,
        started_at_unix: float,
        force: bool,
    ) -> None:
        self.checkpoint_path = logs_dir / CHECKPOINT_NAME
        self.status_path = logs_dir / STATUS_JSONL_NAME
        previous = load_run_checkpoint(self.checkpoint_path)
        try:
            self.attempt = int(previous.get("attempt", 0)) + 1
        except (TypeError, ValueError):
            self.attempt = 1
        self.started_at_unix = started_at_unix
        self.bag = bag
        self.current_stage: str | None = None
        if force:
            self.completed_stages: list[str] = []
            self.stage_outcomes: dict[str, str] = {}
            self.stage_timings_s: dict[str, float] = {}
        else:
            recovered = previous.get("completed_stages", [])
            self.completed_stages = (
                [str(value) for value in recovered]
                if isinstance(recovered, list)
                else []
            )
            outcomes = previous.get("stage_outcomes", {})
            self.stage_outcomes = (
                {str(key): str(value) for key, value in outcomes.items()}
                if isinstance(outcomes, dict)
                else {}
            )
            timings = previous.get("stage_timings_s", {})
            self.stage_timings_s = {}
            if isinstance(timings, dict):
                for key, value in timings.items():
                    try:
                        self.stage_timings_s[str(key)] = float(value)
                    except (TypeError, ValueError):
                        continue
        self._write_checkpoint("running")
        self._append_event(
            "started",
            recovered_completed_stages=list(self.completed_stages),
            force=bool(force),
        )

    def _checkpoint_payload(
        self,
        status: str,
        error: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format_version": STATE_FORMAT_VERSION,
            "updated_at_unix": time.time(),
            "started_at_unix": self.started_at_unix,
            "attempt": self.attempt,
            "status": status,
            "current_stage": self.current_stage,
            "completed_stages": list(self.completed_stages),
            "stage_outcomes": dict(self.stage_outcomes),
            "stage_timings_s": dict(self.stage_timings_s),
            "rosbag": str(self.bag),
        }
        if error is not None:
            payload["error"] = error
        return payload

    def _write_checkpoint(
        self,
        status: str,
        error: dict[str, str] | None = None,
    ) -> None:
        atomic_write_json(
            self.checkpoint_path,
            self._checkpoint_payload(status, error),
        )

    def _append_event(self, event: str, **values: Any) -> None:
        payload = {
            "format_version": STATE_FORMAT_VERSION,
            "time_unix": time.time(),
            "attempt": self.attempt,
            "event": event,
            "status": event,
            **values,
        }
        append_jsonl(self.status_path, payload)

    def start_stage(self, name: str) -> None:
        self.current_stage = name
        self._write_checkpoint("running")
        self._append_event(
            "stage_started",
            stage=name,
            completed_stages=list(self.completed_stages),
        )

    def complete_stage(
        self,
        name: str,
        *,
        outcome: str,
        elapsed_s: float = 0.0,
    ) -> None:
        if name not in self.completed_stages:
            self.completed_stages.append(name)
        self.stage_outcomes[name] = outcome
        self.stage_timings_s[name] = float(elapsed_s)
        self.current_stage = None
        self._write_checkpoint("running")
        self._append_event(
            "stage_complete",
            stage=name,
            outcome=outcome,
            elapsed_s=float(elapsed_s),
            completed_stages=list(self.completed_stages),
        )

    def finish(self) -> None:
        self.current_stage = None
        self._write_checkpoint("complete")
        self._append_event(
            "complete",
            completed_stages=list(self.completed_stages),
            elapsed_s=time.time() - self.started_at_unix,
        )

    def fail(self, exc: BaseException) -> None:
        error = {"type": type(exc).__name__, "message": str(exc)}
        try:
            self._write_checkpoint("failed", error)
        except Exception as checkpoint_error:
            print(
                f"[checkpoint-write-failed] {checkpoint_error}",
                file=sys.stderr,
                flush=True,
            )
        try:
            self._append_event(
                "failed",
                stage=self.current_stage,
                completed_stages=list(self.completed_stages),
                error=error,
                elapsed_s=time.time() - self.started_at_unix,
            )
        except Exception as status_error:
            print(
                f"[status-append-failed] {status_error}",
                file=sys.stderr,
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("rosbag_pos", nargs="?", type=Path, help="Path to rosbag.db3 or a directory containing it")
    p.add_argument("--rosbag", dest="rosbag_opt", type=Path, help="Path to rosbag.db3 or a directory containing it")
    p.add_argument("--output-dir", type=Path, help="Default: rosbag run dir/reconstruction")
    p.add_argument(
        "--relative-path",
        default="",
        help="Optional floor/date/run_N identity recorded for batch verification",
    )
    p.add_argument("--da3-env", default="da3")
    p.add_argument("--gsam2-env", default="gsam2")
    p.add_argument("--workflow-config", type=Path, default=BEST_WORKFLOW_CONFIG)
    p.add_argument("--kalibr-yaml", type=Path, default=HILTI_REPO / "config/hilti_openvins/kalibr_imucam_chain.yaml")
    p.add_argument("--mask0", type=Path, default=HILTI_REPO / "config/hilti_openvins/mask_cam0.png")
    p.add_argument("--mask1", type=Path, default=HILTI_REPO / "config/hilti_openvins/mask_cam1.png")
    p.add_argument("--extract-stride", type=int, default=10)
    p.add_argument("--max-equirect-frames", type=int, default=0)
    p.add_argument("--extract-progress-interval-s", type=float, default=5.0)
    p.add_argument("--pinhole-progress-interval-s", type=float, default=5.0)
    p.add_argument("--pinhole-width", type=int, default=768)
    p.add_argument("--pinhole-height", type=int, default=512)
    p.add_argument("--pinhole-fov-deg", type=float, default=ARG_DEFAULTS["pinhole_fov_deg"])
    p.add_argument("--yaws", default="0,90,180,270")
    p.add_argument("--imu-tau", type=float, default=ARG_DEFAULTS["imu_tau"])
    p.add_argument("--imu-method", choices=("causal_accel", "complementary"), default=ARG_DEFAULTS["imu_method"])
    p.add_argument("--accel-gate-sigma", type=float, default=0.2)
    p.add_argument("--time-offset-ns", type=int, default=0)
    p.add_argument("--use-yaml-timeshift", action="store_true")
    p.set_defaults(rotate180=True)
    p.add_argument("--rotate180", dest="rotate180", action="store_true", help="Rotate pinhole views by 180 degrees after projection; default matches the old HILTI workflow")
    p.add_argument("--no-rotate180", dest="rotate180", action="store_false", help="Disable the default 180 degree pinhole rotation")
    p.add_argument("--mask-prompt", default="person.helmet.")
    p.add_argument("--box-threshold", type=float, default=0.3)
    p.add_argument("--text-threshold", type=float, default=0.3)
    p.add_argument("--mask-close-pixels", type=int, default=3)
    p.add_argument("--mask-dilate-pixels", type=int, default=5)
    p.add_argument("--grounding-batch-size", type=int, default=ARG_DEFAULTS["grounding_batch_size"])
    p.add_argument("--hf-grounding-model", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--hf-grounding-revision", default=ARG_DEFAULTS["hf_grounding_revision"])
    p.add_argument("--skip-equirect", action="store_true")
    p.add_argument("--skip-pinhole", action="store_true")
    p.add_argument("--skip-masks", action="store_true")
    p.add_argument("--skip-da3", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--keep-intermediate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retain extracted frames and masks after success (default: true).",
    )
    p.add_argument(
        "--keep-work",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retain DA3 work products after success (default: true).",
    )
    p.add_argument(
        "--no-delete-rosbag-on-success",
        dest="delete_rosbag_on_success",
        action="store_false",
        default=False,
        help=(
            "Explicitly record that the input ROS bag must be retained. "
            "The standalone runner never deletes ROS bags."
        ),
    )
    args = p.parse_args()
    args.rosbag = args.rosbag_opt or args.rosbag_pos
    if args.rosbag is None:
        p.error("rosbag is required, either as a positional argument or with --rosbag")
    del args.rosbag_opt
    del args.rosbag_pos
    return args


def _parse_scalar(value: str):
    value = value.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_simple_yaml(path: Path) -> dict:
    data = {}
    stack = [(-1, data)]
    for raw in path.expanduser().read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, value = line.strip().split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip():
            parent[key] = _parse_scalar(value)
        else:
            child = {}
            parent[key] = child
            stack.append((indent, child))
    return data


def load_yaml(path: Path) -> dict:
    if yaml is not None:
        with path.expanduser().open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    return _load_simple_yaml(path)


def apply_workflow_frontend_config(args: argparse.Namespace, argv: list[str]) -> None:
    """Apply extraction/pinhole/mask settings from the workflow YAML."""

    cfg = load_yaml(args.workflow_config)
    sections = {
        "extraction": {
            "stride": "extract_stride",
            "max_equirect_frames": "max_equirect_frames",
            "progress_interval_s": "extract_progress_interval_s",
        },
        "pinhole": {
            "width": "pinhole_width",
            "height": "pinhole_height",
            "fov_deg": "pinhole_fov_deg",
            "yaws": "yaws",
            "imu_tau": "imu_tau",
            "imu_method": "imu_method",
            "accel_gate_sigma": "accel_gate_sigma",
            "time_offset_ns": "time_offset_ns",
            "progress_interval_s": "pinhole_progress_interval_s",
            "rotate180": "rotate180",
            "use_yaml_timeshift": "use_yaml_timeshift",
        },
        "confidence_zero_masks": {
            "prompt": "mask_prompt",
            "box_threshold": "box_threshold",
            "text_threshold": "text_threshold",
            "mask_close_pixels": "mask_close_pixels",
            "mask_dilate_pixels": "mask_dilate_pixels",
            "grounding_batch_size": "grounding_batch_size",
            "hf_grounding_model": "hf_grounding_model",
            "hf_grounding_revision": "hf_grounding_revision",
        },
    }
    explicit_flags = {
        token.split("=", 1)[0].replace("-", "_").lstrip("_")
        for token in argv
        if token.startswith("--")
    }
    for section_name, mapping in sections.items():
        section = cfg.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for key, attr in mapping.items():
            if key not in section or attr in explicit_flags:
                continue
            if attr in ARG_DEFAULTS and getattr(args, attr) != ARG_DEFAULTS[attr]:
                continue
            setattr(args, attr, section[key])


def resolve_rosbag(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    candidates = [path / "rosbag.db3", path / "rosbag" / "rosbag.db3"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"No rosbag.db3 found from {path}")


def default_output_dir(bag: Path) -> Path:
    if bag.name == "rosbag.db3" and bag.parent.name == "rosbag":
        return bag.parent.parent / "reconstruction"
    return bag.parent / "reconstruction"


def run(cmd: list[str], log: Path, cwd: Path | None = None) -> float:
    started_at = time.perf_counter()
    log.parent.mkdir(parents=True, exist_ok=True)
    text = " ".join(str(x) for x in cmd)
    print(f"\n[run] {text}", flush=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n\n$ {text}\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)
    return time.perf_counter() - started_at


def conda_python(env: str, script: Path, *args: str | Path) -> list[str]:
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        env,
        "python",
        str(script),
        *[str(a) for a in args],
    ]


def has_files(path: Path, pattern: str) -> bool:
    return path.exists() and any(path.glob(pattern))


def ensure_clean_dir(path: Path, force: bool) -> None:
    if path.exists() and force:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def files_are_identical(first: Path, second: Path) -> bool:
    return bool(
        first.is_file()
        and second.is_file()
        and first.stat().st_size == second.stat().st_size
        and file_sha256(first) == file_sha256(second)
    )


def copy_file(
    src: Path,
    dst: Path,
    *,
    required: bool = True,
    preserve_identical_existing: bool = False,
) -> bool:
    """Copy through a fsynced sibling and optionally protect an existing target."""
    if not src.is_file():
        if required:
            raise FileNotFoundError(src)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if preserve_identical_existing and dst.exists():
        if not files_are_identical(src, dst):
            raise RuntimeError(
                f"Refusing to overwrite non-identical existing output: {dst}"
            )
        return True
    temporary = dst.with_name(
        f".{dst.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        with src.open("rb") as source, temporary.open("xb") as target:
            before = os.fstat(source.fileno())
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
            after = os.fstat(source.fileno())
        current = src.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ino != after.st_ino
            or after.st_size != current.st_size
            or after.st_mtime_ns != current.st_mtime_ns
            or after.st_ino != current.st_ino
        ):
            raise RuntimeError(f"Source changed while copying: {src}")
        shutil.copystat(src, temporary)
        temporary.replace(dst)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def copy_light_logs(work_dir: Path, logs_dir: Path) -> list[Path]:
    suffixes = {".txt", ".csv", ".json", ".yaml", ".yml", ".log"}
    scientific_npz = {"salad_descriptors.npz", "salad_capture_matrix.npz"}
    copied: list[Path] = []
    for path in work_dir.rglob("*"):
        if not path.is_file() or (
            path.suffix.lower() not in suffixes and path.name not in scientific_npz
        ):
            continue
        if any(part in {"pcd", "tmp_render_mp4v"} for part in path.parts):
            continue
        rel = path.relative_to(work_dir)
        destination = logs_dir / "da3_work" / rel
        if copy_file(path, destination, required=False):
            copied.append(destination)
    return copied


def write_camera_centers_ply(poses_txt: Path, output_ply: Path) -> None:
    centers = []
    for line in poses_txt.read_text(encoding="utf-8").splitlines():
        vals = [float(x) for x in line.split()]
        if len(vals) != 16:
            continue
        centers.append((vals[3], vals[7], vals[11]))
    if not centers:
        return
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(centers)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for idx, (x, y, z) in enumerate(centers):
        # A blue-to-green ramp makes trajectory direction visible in viewers.
        t = idx / max(1, len(centers) - 1)
        red = 30
        green = int(80 + 175 * t)
        blue = int(255 - 175 * t)
        lines.append(f"{x:.8f} {y:.8f} {z:.8f} {red} {green} {blue}")
    encoded = ("\n".join(lines) + "\n").encode("utf-8")
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_ply.with_name(
        f".{output_ply.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output_ply)
    finally:
        if temporary.exists():
            temporary.unlink()


def select_first(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of the expected outputs exist:\n" + "\n".join(str(p) for p in paths))


def finalize_outputs(
    output_dir: Path,
    work_dir: Path,
    masks_dir: Path,
    started_at: float,
    args: argparse.Namespace,
    bag: Path,
    stage_timings_s: dict[str, float],
) -> None:
    """Publish reconstruction-space outputs without evaluation alignment."""
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    final_pcd = select_first(
        [
            work_dir / "pcd" / "combined_pcd_clean_voxel005.ply",
            work_dir / "pcd" / "combined_pcd.ply",
        ]
    )
    final_poses = work_dir / "camera_poses.txt"
    if not final_poses.is_file():
        raise FileNotFoundError(final_poses)

    copy_file(
        final_pcd,
        output_dir / "reconstruction.ply",
        preserve_identical_existing=True,
    )
    copy_file(final_poses, output_dir / "camera_poses.txt")
    write_camera_centers_ply(
        final_poses,
        output_dir / "camera_poses.ply",
    )
    copied_logs = copy_light_logs(work_dir, logs_dir)
    copy_file(
        masks_dir / "summary.json",
        logs_dir / "gsam2_mask_summary.json",
        required=False,
    )

    manifest = {
        "schema_version": 2,
        "created_at_unix": time.time(),
        "elapsed_s": time.time() - started_at,
        "stage_timings_s": stage_timings_s,
        "rosbag": str(bag),
        "output_dir": str(output_dir),
        "da3_env": args.da3_env,
        "gsam2_env": args.gsam2_env,
        "relative_path": args.relative_path.strip("/") or None,
        "workflow_config_provenance": required_file_identity(
            args.workflow_config
        ),
        "final_outputs": {
            "pointcloud": str(output_dir / "reconstruction.ply"),
            "camera_poses": str(output_dir / "camera_poses.txt"),
            "camera_pose_ply": str(output_dir / "camera_poses.ply"),
        },
        "final_output_provenance": {
            "reconstruction.ply": required_file_identity(
                output_dir / "reconstruction.ply"
            ),
            "camera_poses.txt": required_file_identity(
                output_dir / "camera_poses.txt"
            ),
            "camera_poses.ply": required_file_identity(
                output_dir / "camera_poses.ply"
            ),
        },
        "scientific_artifacts": {
            str(path.relative_to(output_dir)): required_file_identity(path)
            for path in copied_logs
            if path.name in {
                "camera_poses_pre_loop.txt",
                "yaw4_rig_repair_frames.csv",
                "yaw4_rig_repair_groups.csv",
                "loop_geometry_verification.csv",
                "salad_descriptors.npz",
                "salad_capture_matrix.npz",
            }
        },
        "source_outputs": {
            "pointcloud": str(final_pcd),
            "camera_poses": str(final_poses),
        },
        "settings": {
            "extract_stride": args.extract_stride,
            "max_equirect_frames": args.max_equirect_frames,
            "extract_progress_interval_s": args.extract_progress_interval_s,
            "pinhole_progress_interval_s": args.pinhole_progress_interval_s,
            "pinhole_width": args.pinhole_width,
            "pinhole_height": args.pinhole_height,
            "pinhole_fov_deg": args.pinhole_fov_deg,
            "imu_method": args.imu_method,
            "imu_tau": args.imu_tau,
            "accel_gate_sigma": args.accel_gate_sigma,
            "yaws": args.yaws,
            "rotate180": args.rotate180,
            "mask_prompt": args.mask_prompt,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "grounding_batch_size": args.grounding_batch_size,
            "hf_grounding_model": args.hf_grounding_model,
            "hf_grounding_revision": args.hf_grounding_revision,
            "workflow_config": str(args.workflow_config),
            "keep_intermediate": args.keep_intermediate,
            "keep_work": args.keep_work,
            "rosbag_retention_policy": "retain",
        },
    }
    atomic_write_json(output_dir / "workflow_manifest.json", manifest)


def run_pipeline(
    args: argparse.Namespace,
    bag: Path,
    output_dir: Path,
    started_at: float,
    progress: RunProgress,
) -> int:
    # Preserve the manifest's historical meaning: timings include only work
    # executed by this invocation.  The checkpoint separately records every
    # stage outcome, including zero-cost reuse and explicit skips.
    stage_timings_s: dict[str, float] = {}
    logs_dir = output_dir / "logs"
    intermediate = output_dir / "_intermediate"
    equirect_dir = intermediate / "equirect"
    pinhole_dir = intermediate / "pinhole_yaw4_imu"
    masks_dir = intermediate / "confidence_zero_masks"
    work_dir = output_dir / "_da3_work"
    stage_records: dict[str, dict[str, Any]] = {}
    bag_identity = required_file_identity(bag)

    progress.start_stage("prepare_workdirs")
    if args.force:
        for path in (intermediate, work_dir):
            if path.exists():
                shutil.rmtree(path)

    ensure_clean_dir(equirect_dir, False)
    ensure_clean_dir(pinhole_dir, False)
    ensure_clean_dir(masks_dir, False)
    ensure_clean_dir(work_dir, False)
    progress.complete_stage(
        "prepare_workdirs",
        outcome="force_reset" if args.force else "ready",
    )

    progress.start_stage("extract_equirect")
    if not args.skip_equirect:
        extraction_request = {
            "bag": bag_identity,
            "kalibr_yaml": required_file_identity(args.kalibr_yaml),
            "mask0": required_file_identity(args.mask0),
            "mask1": required_file_identity(args.mask1),
            "settings": {
                "stride": int(args.extract_stride),
                "max_equirect_frames": int(args.max_equirect_frames),
                "progress_interval_s": float(args.extract_progress_interval_s),
            },
        }
        extraction_reusable, extraction_record = prepare_stage_provenance(
            logs_dir=logs_dir,
            stage="extract_equirect",
            request=extraction_request,
            output_dir=equirect_dir,
            suffixes={".jpg", ".jpeg", ".png", ".webp"},
            force=args.force,
        )
        if extraction_reusable:
            print(f"[skip-equirect] existing frames: {equirect_dir}", flush=True)
            extract_outcome = "reused_existing"
            extract_elapsed = 0.0
        else:
            extract_elapsed = run(
                conda_python(
                    args.da3_env,
                    REPO / "tools/hilti_workflow/extraction/extract_hilti_frames.py",
                    "--bag",
                    bag,
                    "--yaml",
                    args.kalibr_yaml,
                    "--out_dir",
                    equirect_dir,
                    "--mask0",
                    args.mask0,
                    "--mask1",
                    args.mask1,
                    "--stride",
                    args.extract_stride,
                    "--max_frames",
                    args.max_equirect_frames,
                    "--progress_interval_s",
                    args.extract_progress_interval_s,
                ),
                logs_dir / "01_extract_equirect.log",
            )
            stage_timings_s["extract_equirect"] = extract_elapsed
            extract_outcome = "executed"
            extraction_record = complete_stage_provenance(
                logs_dir=logs_dir,
                stage="extract_equirect",
                request=extraction_request,
                output_dir=equirect_dir,
                suffixes={".jpg", ".jpeg", ".png", ".webp"},
            )
        assert extraction_record is not None
        equirect_signature = extraction_record["outputs"]
        stage_records["extract_equirect"] = {
            **extraction_record,
            "invocation_outcome": extract_outcome,
        }
    else:
        extract_outcome = "cli_skipped"
        extract_elapsed = 0.0
        equirect_signature = filename_set_signature(
            equirect_dir,
            {".jpg", ".jpeg", ".png", ".webp"},
            require_nonempty=False,
        )
        stage_records["extract_equirect"] = {
            "schema_version": 1,
            "stage": "extract_equirect",
            "status": "cli_skipped",
            "invocation_outcome": extract_outcome,
            "outputs": equirect_signature,
        }
    progress.complete_stage(
        "extract_equirect",
        outcome=extract_outcome,
        elapsed_s=extract_elapsed,
    )

    progress.start_stage("generate_pinhole_yaw4_imu")
    if not args.skip_pinhole:
        pinhole_names = expected_pinhole_names(equirect_dir, args.yaws)
        if not pinhole_names:
            raise RuntimeError(
                "Pinhole generation requires nonempty equirectangular inputs"
            )
        pinhole_request = {
            "bag": bag_identity,
            "kalibr_yaml": required_file_identity(args.kalibr_yaml),
            "equirect_outputs": equirect_signature,
            "settings": {
                "width": int(args.pinhole_width),
                "height": int(args.pinhole_height),
                "fov_deg": float(args.pinhole_fov_deg),
                "yaws": str(args.yaws),
                "imu_tau": float(args.imu_tau),
                "imu_method": str(args.imu_method),
                "accel_gate_sigma": float(args.accel_gate_sigma),
                "time_offset_ns": int(args.time_offset_ns),
                "use_yaml_timeshift": bool(args.use_yaml_timeshift),
                "rotate180": bool(args.rotate180),
                "progress_interval_s": float(
                    args.pinhole_progress_interval_s
                ),
            },
        }
        pinhole_reusable, pinhole_record = prepare_stage_provenance(
            logs_dir=logs_dir,
            stage="generate_pinhole_yaw4_imu",
            request=pinhole_request,
            output_dir=pinhole_dir,
            suffixes={".jpg"},
            expected_names=pinhole_names,
            force=args.force,
        )
        if pinhole_reusable:
            print(f"[skip-pinhole] existing yaw views: {pinhole_dir}", flush=True)
            pinhole_outcome = "reused_existing"
            pinhole_elapsed = 0.0
        else:
            cmd = conda_python(
                args.da3_env,
                REPO / "tools/hilti_workflow/view_generation/imu_level_equirect_pose_sequence.py",
                "--bag",
                bag,
                "--input_dir",
                equirect_dir,
                "--yaml",
                args.kalibr_yaml,
                "--output_dir",
                pinhole_dir,
                "--width",
                args.pinhole_width,
                "--height",
                args.pinhole_height,
                "--fov_deg",
                args.pinhole_fov_deg,
                "--yaws",
                args.yaws,
                "--imu_tau",
                args.imu_tau,
                "--imu-method",
                args.imu_method,
                "--accel-gate-sigma",
                args.accel_gate_sigma,
                "--time_offset_ns",
                args.time_offset_ns,
                "--progress_interval_s",
                args.pinhole_progress_interval_s,
            )
            if args.use_yaml_timeshift:
                cmd.append("--use_yaml_timeshift")
            if args.rotate180:
                cmd.append("--rotate180")
            pinhole_elapsed = run(
                cmd, logs_dir / "02_generate_pinhole_yaw4_imu.log"
            )
            stage_timings_s["generate_pinhole_yaw4_imu"] = pinhole_elapsed
            pinhole_outcome = "executed"
            pinhole_record = complete_stage_provenance(
                logs_dir=logs_dir,
                stage="generate_pinhole_yaw4_imu",
                request=pinhole_request,
                output_dir=pinhole_dir,
                suffixes={".jpg"},
                expected_names=pinhole_names,
            )
        assert pinhole_record is not None
        pinhole_signature = pinhole_record["outputs"]
        stage_records["generate_pinhole_yaw4_imu"] = {
            **pinhole_record,
            "invocation_outcome": pinhole_outcome,
        }
    else:
        pinhole_outcome = "cli_skipped"
        pinhole_elapsed = 0.0
        pinhole_signature = filename_set_signature(
            pinhole_dir, {".jpg"}, require_nonempty=False
        )
        stage_records["generate_pinhole_yaw4_imu"] = {
            "schema_version": 1,
            "stage": "generate_pinhole_yaw4_imu",
            "status": "cli_skipped",
            "invocation_outcome": pinhole_outcome,
            "outputs": pinhole_signature,
        }
    progress.complete_stage(
        "generate_pinhole_yaw4_imu",
        outcome=pinhole_outcome,
        elapsed_s=pinhole_elapsed,
    )

    progress.start_stage("gsam2_masks")
    if not args.skip_masks:
        mask_output_dir = masks_dir / "masks_npy"
        expected_mask_names = {
            f"{path.stem}.npy"
            for path in _listed_output_files(pinhole_dir, {".jpg"})
        }
        if not expected_mask_names:
            raise RuntimeError("GSAM mask generation requires nonempty pinhole views")
        masks_request = {
            "pinhole_outputs": pinhole_signature,
            "settings": {
                "prompt": str(args.mask_prompt),
                "box_threshold": float(args.box_threshold),
                "text_threshold": float(args.text_threshold),
                "mask_close_pixels": int(args.mask_close_pixels),
                "mask_dilate_pixels": int(args.mask_dilate_pixels),
                "grounding_batch_size": int(args.grounding_batch_size),
                "hf_grounding_model": str(args.hf_grounding_model),
                "hf_grounding_revision": str(args.hf_grounding_revision),
            },
        }
        masks_reusable, masks_record = prepare_stage_provenance(
            logs_dir=logs_dir,
            stage="gsam2_masks",
            request=masks_request,
            output_dir=mask_output_dir,
            suffixes={".npy"},
            expected_names=expected_mask_names,
            force=args.force,
        )
        if masks_reusable:
            print(
                f"[skip-masks] complete existing npy masks: "
                f"{len(expected_mask_names)}/{len(expected_mask_names)}",
                flush=True,
            )
            masks_outcome = "reused_existing"
            masks_elapsed = 0.0
        else:
            existing_masks = len(
                _listed_output_files(mask_output_dir, {".npy"})
            )
            if existing_masks and not args.force:
                print(
                    f"[resume-masks] existing={existing_masks} "
                    f"expected={len(expected_mask_names)}; filling missing files",
                    flush=True,
                )
            masks_elapsed = run(
                conda_python(
                    args.gsam2_env,
                    GSAM2_REPO / "tools/hilti_person_mask_batch.py",
                    "--input-dir",
                    pinhole_dir,
                    "--output-dir",
                    masks_dir,
                    "--prompt",
                    args.mask_prompt,
                    "--grounding-backend",
                    "hf",
                    "--hf-grounding-model",
                    args.hf_grounding_model,
                    "--hf-grounding-revision",
                    args.hf_grounding_revision,
                    "--box-threshold",
                    args.box_threshold,
                    "--text-threshold",
                    args.text_threshold,
                    "--mask-close-pixels",
                    args.mask_close_pixels,
                    "--mask-dilate-pixels",
                    args.mask_dilate_pixels,
                    "--grounding-batch-size",
                    args.grounding_batch_size,
                    "--export-mode",
                    "npy",
                    "--skip-existing",
                ),
                logs_dir / "03_extract_confidence_zero_masks.log",
                cwd=GSAM2_REPO,
            )
            stage_timings_s["gsam2_masks"] = masks_elapsed
            masks_outcome = "executed"
            masks_record = complete_stage_provenance(
                logs_dir=logs_dir,
                stage="gsam2_masks",
                request=masks_request,
                output_dir=mask_output_dir,
                suffixes={".npy"},
                expected_names=expected_mask_names,
            )
        assert masks_record is not None
        stage_records["gsam2_masks"] = {
            **masks_record,
            "invocation_outcome": masks_outcome,
        }
    else:
        masks_outcome = "cli_skipped"
        masks_elapsed = 0.0
        stage_records["gsam2_masks"] = {
            "schema_version": 1,
            "stage": "gsam2_masks",
            "status": "cli_skipped",
            "invocation_outcome": masks_outcome,
            "outputs": filename_set_signature(
                masks_dir / "masks_npy",
                {".npy"},
                require_nonempty=False,
            ),
        }
    progress.complete_stage(
        "gsam2_masks",
        outcome=masks_outcome,
        elapsed_s=masks_elapsed,
    )
    atomic_write_json(
        logs_dir / "frontend_protocol_manifest.json",
        frontend_protocol_manifest(args, bag, stage_records, bag_identity),
    )

    if args.skip_da3:
        progress.start_stage("best_reconstruction_workflow")
        skip_log = logs_dir / "04_da3_streaming_postprocess_visualize.log"
        skip_log.write_text(
            "[skip-da3] skipped DA3 streaming, point-cloud cleanup, and final "
            "output collection.\n",
            encoding="utf-8",
        )
        progress.complete_stage(
            "best_reconstruction_workflow",
            outcome="cli_skipped",
        )
        progress.start_stage("finalize_outputs")
        progress.complete_stage("finalize_outputs", outcome="cli_skipped")
    else:
        progress.start_stage("best_reconstruction_workflow")
        best_cmd = conda_python(
            args.da3_env,
            BEST_WORKFLOW,
            "--image-dir",
            pinhole_dir,
            "--confidence-zero-mask-dir",
            masks_dir / "masks_npy",
            "--output-dir",
            work_dir,
            "--workflow-config",
            args.workflow_config,
        )
        if args.force:
            best_cmd.append("--force")
        best_elapsed = run(
            best_cmd, logs_dir / "04_da3_streaming_postprocess_visualize.log"
        )
        stage_timings_s["best_reconstruction_workflow"] = best_elapsed
        progress.complete_stage(
            "best_reconstruction_workflow",
            outcome="executed",
            elapsed_s=best_elapsed,
        )

        progress.start_stage("finalize_outputs")
        finalize_started = time.perf_counter()
        finalize_outputs(
            output_dir,
            work_dir,
            masks_dir,
            started_at,
            args,
            bag,
            stage_timings_s,
        )
        progress.complete_stage(
            "finalize_outputs",
            outcome="executed",
            elapsed_s=time.perf_counter() - finalize_started,
        )

    progress.start_stage("retention_cleanup")
    if not args.keep_intermediate:
        shutil.rmtree(intermediate, ignore_errors=True)
    if not args.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)
    retention_outcome = (
        "retained"
        if args.keep_intermediate and args.keep_work
        else "cleanup_applied"
    )
    progress.complete_stage(
        "retention_cleanup",
        outcome=retention_outcome,
    )

    print(f"\nDone: {output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    apply_workflow_frontend_config(args, sys.argv[1:])
    started_at = time.time()
    bag = resolve_rosbag(args.rosbag)
    output_dir = (args.output_dir or default_output_dir(bag)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = RunProgress(
        output_dir / "logs",
        bag=bag,
        started_at_unix=started_at,
        force=args.force,
    )
    try:
        result = run_pipeline(args, bag, output_dir, started_at, progress)
        progress.finish()
        return result
    except BaseException as exc:
        progress.fail(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
