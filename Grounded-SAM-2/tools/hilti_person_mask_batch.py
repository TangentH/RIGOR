from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def sorted_images(input_dir: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in suffixes)


def ensure_prompt(prompt: str) -> str:
    prompt = prompt.strip().lower()
    if not prompt.endswith("."):
        prompt += "."
    return prompt


def load_hf_grounding_model(
    model_id: str,
    revision: str | None,
    device: str,
    disable_custom_kernels: bool = True,
):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    if disable_custom_kernels:
        from transformers.models.grounding_dino import modeling_grounding_dino

        # This transformers version tries to compile the extension in each layer's
        # constructor even when the config disables it for forward passes.
        previous_kernel = modeling_grounding_dino.MultiScaleDeformableAttention
        modeling_grounding_dino.MultiScaleDeformableAttention = object()
        try:
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id,
                revision=revision,
                disable_custom_kernels=True,
            )
        finally:
            modeling_grounding_dino.MultiScaleDeformableAttention = previous_kernel
    else:
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            revision=revision,
            disable_custom_kernels=False,
        )
    model = model.to(device)
    model.eval()
    return processor, model


def load_local_grounding_model(config_path: str, checkpoint_path: str, device: str):
    try:
        from grounding_dino.groundingdino.util.inference import load_model
    except ModuleNotFoundError:
        from groundingdino.util.inference import load_model

    model = load_model(
        model_config_path=config_path,
        model_checkpoint_path=checkpoint_path,
        device=device,
    )
    model.eval()
    return model


def detect_hf_batch(
    processor,
    model,
    image_paths: list[Path],
    prompt: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
):
    images = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    image_sources = [np.asarray(image).copy() for image in images]
    prompts = [prompt] * len(images)
    inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = [image.size[::-1] for image in images]
    try:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )

    detections = []
    for image_source, result in zip(image_sources, results):
        boxes_xyxy = result["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = result["scores"].detach().cpu().numpy().astype(np.float32)
        labels = [str(label) for label in result["labels"]]
        detections.append((image_source, boxes_xyxy, scores, labels))
    return detections


def detect_hf(processor, model, image_path: Path, prompt: str, box_threshold: float, text_threshold: float, device: str):
    return detect_hf_batch(
        processor,
        model,
        [image_path],
        prompt,
        box_threshold,
        text_threshold,
        device,
    )[0]


def detect_local(model, image_path: Path, prompt: str, box_threshold: float, text_threshold: float, device: str):
    try:
        from grounding_dino.groundingdino.util.inference import load_image, predict
    except ModuleNotFoundError:
        from groundingdino.util.inference import load_image, predict

    image_source, image_tensor = load_image(str(image_path))
    height, width = image_source.shape[:2]
    boxes, confidences, labels = predict(
        model=model,
        image=image_tensor,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )
    if len(boxes) == 0:
        return image_source, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), []

    boxes = boxes * torch.tensor([width, height, width, height], device=boxes.device)
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
    boxes_xyxy = input_boxes.detach().cpu().numpy().astype(np.float32)
    scores = confidences.detach().cpu().numpy().astype(np.float32)
    labels = [str(label) for label in labels]
    return image_source, boxes_xyxy, scores, labels


def ellipse_kernel(radius: int) -> np.ndarray | None:
    if radius <= 0:
        return None
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def fill_holes_binary(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask

    padded = np.pad(mask.astype(np.uint8), pad_width=1, mode="constant", constant_values=0)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 2)
    holes = flood[1:-1, 1:-1] == 0
    return mask | holes


def refine_binary_mask(mask: np.ndarray, close_pixels: int, fill_holes: bool, dilate_pixels: int) -> np.ndarray:
    refined = mask.astype(bool, copy=True)

    close_kernel = ellipse_kernel(close_pixels)
    if close_kernel is not None and refined.any():
        refined = cv2.morphologyEx(refined.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel).astype(bool)

    if fill_holes:
        refined = fill_holes_binary(refined)

    dilate_kernel = ellipse_kernel(dilate_pixels)
    if dilate_kernel is not None and refined.any():
        refined = cv2.dilate(refined.astype(np.uint8), dilate_kernel, iterations=1).astype(bool)

    return refined


def refine_instance_masks(
    masks: np.ndarray,
    close_pixels: int,
    fill_holes: bool,
    dilate_pixels: int,
) -> np.ndarray:
    if len(masks) == 0:
        return masks.astype(bool)
    return np.stack(
        [refine_binary_mask(mask, close_pixels, fill_holes, dilate_pixels) for mask in masks],
        axis=0,
    )


def draw_overlay(
    image_path: Path,
    union_mask: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    output_path: Path,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image for overlay: {image_path}")

    if union_mask.any():
        color = np.zeros_like(image)
        color[:, :, 2] = 255
        image = np.where(union_mask[:, :, None], (0.55 * image + 0.45 * color).astype(np.uint8), image)

    for box, score, label in zip(boxes_xyxy, scores, labels):
        x1, y1, x2, y2 = np.round(box).astype(int).tolist()
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        text = f"{label} {score:.2f}"
        cv2.putText(
            image,
            text,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), image)


def save_empty_outputs(
    image_path: Path,
    height: int,
    width: int,
    out_npy: Path,
    out_npz: Path | None = None,
    out_png: Path | None = None,
    out_overlay: Path | None = None,
) -> dict:
    union_mask = np.zeros((height, width), dtype=bool)
    instance_masks = np.zeros((0, height, width), dtype=bool)
    boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
    scores = np.zeros((0,), dtype=np.float32)
    labels = np.asarray([], dtype=str)

    np.save(out_npy, union_mask)
    if out_npz is not None:
        np.savez_compressed(
            out_npz,
            person_mask=union_mask,
            raw_person_mask=union_mask,
            instance_masks=instance_masks,
            boxes_xyxy=boxes_xyxy,
            scores=scores,
            labels=labels,
            image_path=str(image_path),
            img_height=height,
            img_width=width,
        )
    if out_png is not None:
        cv2.imwrite(str(out_png), union_mask.astype(np.uint8) * 255)
    if out_overlay is not None:
        draw_overlay(image_path, union_mask, boxes_xyxy, scores, [], out_overlay)

    item = {
        "image_path": str(image_path),
        "mask_npy": str(out_npy),
        "num_instances": 0,
        "raw_mask_pixels": 0,
        "mask_pixels": 0,
        "img_height": height,
        "img_width": width,
    }
    if out_npz is not None:
        item["mask_npz"] = str(out_npz)
    if out_png is not None:
        item["binary_png"] = str(out_png)
    if out_overlay is not None:
        item["overlay"] = str(out_overlay)
    return item


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    npy_dir = output_dir / "masks_npy"
    npy_dir.mkdir(parents=True, exist_ok=True)
    save_debug_outputs = args.export_mode == "full"
    if save_debug_outputs:
        npz_dir = output_dir / "masks_npz"
        png_dir = output_dir / "binary_png"
        overlay_dir = output_dir / "overlays"
        for directory in (npz_dir, png_dir, overlay_dir):
            directory.mkdir(parents=True, exist_ok=True)
    else:
        npz_dir = None
        png_dir = None
        overlay_dir = None

    image_paths = sorted_images(input_dir)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    device = args.device
    prompt = ensure_prompt(args.prompt)

    if torch.cuda.is_available() and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    if args.grounding_backend == "hf":
        grounding = load_hf_grounding_model(
            args.hf_grounding_model,
            args.hf_grounding_revision,
            device,
            disable_custom_kernels=not args.enable_hf_custom_kernels,
        )
    else:
        grounding = load_local_grounding_model(
            args.grounding_dino_config,
            args.grounding_dino_checkpoint,
            device,
        )

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "grounding_backend": args.grounding_backend,
        "hf_grounding_model": args.hf_grounding_model if args.grounding_backend == "hf" else None,
        "hf_grounding_revision": args.hf_grounding_revision if args.grounding_backend == "hf" else None,
        "hf_custom_kernels_enabled": (
            args.enable_hf_custom_kernels if args.grounding_backend == "hf" else None
        ),
        "local_grounding_config": args.grounding_dino_config if args.grounding_backend == "local" else None,
        "local_grounding_checkpoint": args.grounding_dino_checkpoint if args.grounding_backend == "local" else None,
        "prompt": prompt,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "mask_close_pixels": args.mask_close_pixels,
        "mask_fill_holes": args.mask_fill_holes,
        "mask_dilate_pixels": args.mask_dilate_pixels,
        "export_mode": args.export_mode,
        "skip_existing": args.skip_existing,
        "grounding_batch_size": args.grounding_batch_size,
        "limit": args.limit,
        "num_images": len(image_paths),
        "images": [],
    }

    def process_detection(idx, image_path, detection):
        out_npy = npy_dir / f"{image_path.stem}.npy"
        out_npz = npz_dir / f"{image_path.stem}.npz" if npz_dir is not None else None
        out_png = png_dir / f"{image_path.stem}.png" if png_dir is not None else None
        out_overlay = overlay_dir / f"{image_path.stem}.jpg" if overlay_dir is not None else None
        image_source, input_boxes_np, scores_np, labels = detection
        height, width = image_source.shape[:2]

        if len(input_boxes_np) == 0:
            summary["images"].append(
                save_empty_outputs(
                    image_path,
                    height,
                    width,
                    out_npy=out_npy,
                    out_npz=out_npz,
                    out_png=out_png,
                    out_overlay=out_overlay,
                )
            )
            return

        sam2_predictor.set_image(image_source)
        with torch.inference_mode():
            if device == "cuda":
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            else:
                autocast_ctx = torch.autocast(device_type="cpu", enabled=False)
            with autocast_ctx:
                masks, mask_scores, _ = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes_np,
                    multimask_output=False,
                )

        if masks.ndim == 4:
            masks = masks.squeeze(1)
        masks = masks.astype(bool)
        raw_union_mask = masks.any(axis=0) if len(masks) else np.zeros((height, width), dtype=bool)
        masks = refine_instance_masks(
            masks,
            close_pixels=args.mask_close_pixels,
            fill_holes=args.mask_fill_holes,
            dilate_pixels=args.mask_dilate_pixels,
        )
        union_mask = masks.any(axis=0) if len(masks) else np.zeros((height, width), dtype=bool)
        labels_np = np.asarray(labels, dtype=str)

        np.save(out_npy, union_mask)
        if save_debug_outputs:
            np.savez_compressed(
                out_npz,
                person_mask=union_mask,
                raw_person_mask=raw_union_mask,
                instance_masks=masks,
                boxes_xyxy=input_boxes_np,
                scores=scores_np,
                labels=labels_np,
                image_path=str(image_path),
                img_height=height,
                img_width=width,
                sam_mask_scores=np.asarray(mask_scores, dtype=np.float32),
                mask_close_pixels=args.mask_close_pixels,
                mask_fill_holes=args.mask_fill_holes,
                mask_dilate_pixels=args.mask_dilate_pixels,
            )
            cv2.imwrite(str(out_png), union_mask.astype(np.uint8) * 255)
            draw_overlay(image_path, union_mask, input_boxes_np, scores_np, list(labels_np), out_overlay)

        item = {
            "image_path": str(image_path),
            "mask_npy": str(out_npy),
            "num_instances": int(len(masks)),
            "raw_mask_pixels": int(raw_union_mask.sum()),
            "mask_pixels": int(union_mask.sum()),
            "img_height": int(height),
            "img_width": int(width),
        }
        if save_debug_outputs:
            item["mask_npz"] = str(out_npz)
            item["binary_png"] = str(out_png)
            item["overlay"] = str(out_overlay)
        summary["images"].append(item)

    batch_size = args.grounding_batch_size if args.grounding_backend == "hf" else 1
    for batch_start in range(0, len(image_paths), batch_size):
        batch_entries = list(
            enumerate(image_paths[batch_start : batch_start + batch_size], start=batch_start + 1)
        )
        pending_entries = []
        for idx, image_path in batch_entries:
            out_npy = npy_dir / f"{image_path.stem}.npy"
            if args.skip_existing and out_npy.exists():
                print(f"[{idx}/{len(image_paths)}] {image_path.name} skip-existing", flush=True)
                summary["images"].append(
                    {
                        "image_path": str(image_path),
                        "mask_npy": str(out_npy),
                        "skipped_existing": True,
                    }
                )
            else:
                print(f"[{idx}/{len(image_paths)}] {image_path.name}", flush=True)
                pending_entries.append((idx, image_path))

        if not pending_entries:
            continue
        if args.grounding_backend == "hf":
            processor, grounding_model = grounding
            detections = detect_hf_batch(
                processor,
                grounding_model,
                [image_path for _, image_path in pending_entries],
                prompt,
                args.box_threshold,
                args.text_threshold,
                device,
            )
        else:
            detections = [
                detect_local(
                    grounding,
                    image_path,
                    prompt,
                    args.box_threshold,
                    args.text_threshold,
                    device,
                )
                for _, image_path in pending_entries
            ]
        for (idx, image_path), detection in zip(pending_entries, detections):
            process_detection(idx, image_path, detection)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"summary={summary_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grounded SAM2 person masks for HILTI images.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prompt", default="person.")
    parser.add_argument("--box-threshold", type=float, default=0.4)
    parser.add_argument("--text-threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--grounding-backend", choices=("hf", "local"), default="hf")
    parser.add_argument("--hf-grounding-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--hf-grounding-revision", default=None)
    parser.add_argument(
        "--enable-hf-custom-kernels",
        action="store_true",
        help="Enable GroundingDINO custom CUDA kernels when the installed extension is compatible.",
    )
    parser.add_argument("--grounding-batch-size", type=int, default=1)
    parser.add_argument("--sam2-checkpoint", default="checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--export-mode", choices=("full", "npy"), default="full")
    parser.add_argument("--skip-existing", action="store_true", help="Skip images whose output .npy mask already exists.")
    parser.add_argument("--mask-close-pixels", type=int, default=0)
    parser.add_argument("--mask-dilate-pixels", type=int, default=0)
    parser.add_argument("--no-fill-holes", action="store_false", dest="mask_fill_holes")
    parser.add_argument(
        "--grounding-dino-config",
        default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument(
        "--grounding-dino-checkpoint",
        default="gdino_checkpoints/groundingdino_swint_ogc.pth",
    )
    parser.set_defaults(mask_fill_holes=True)
    args = parser.parse_args()
    if args.grounding_batch_size < 1:
        parser.error("--grounding-batch-size must be at least 1")
    return args


if __name__ == "__main__":
    run(parse_args())
