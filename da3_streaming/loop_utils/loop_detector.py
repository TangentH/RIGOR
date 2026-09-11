# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long)

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch import nn
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DA3_STREAMING_ROOT = os.path.dirname(CURRENT_DIR)
SALAD_ROOT = os.path.join(CURRENT_DIR, "salad")
DEFAULT_SALAD_CKPT = os.path.join(DA3_STREAMING_ROOT, "weights", "dino_salad.ckpt")
if DA3_STREAMING_ROOT not in sys.path:
    sys.path.insert(0, DA3_STREAMING_ROOT)
if SALAD_ROOT not in sys.path:
    sys.path.insert(0, SALAD_ROOT)
from loop_utils.salad.models import helper


_CAPTURE_RE = re.compile(r"(?:^|_)frame_(?P<capture>\d+)(?:_|$)")
_LEADING_CAPTURE_RE = re.compile(r"^(?P<capture>\d+)(?:_|$)")
_VIEW_RE = re.compile(r"(?:^|_)v(?P<view>\d+)(?:_|$)")
_YAW_RE = re.compile(r"(?:^|_)yaw(?P<yaw>\d+(?:\.\d+)?)(?:_|$)")


def _atomic_write_npz(path, **arrays):
    """Write one complete NPZ and atomically publish it at *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_csv(path, fieldnames, rows):
    """Write a CSV without exposing a partially written final file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


class VPRModel(nn.Module):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(
        self,
        # ---- Backbone
        backbone_arch="resnet50",
        backbone_config={},
        # ---- Aggregator
        agg_arch="ConvAP",
        agg_config={},
    ):
        super().__init__()

        # Backbone
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config

        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, backbone_config)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x


class LoopDetector:
    """Loop detector class for detecting loop closures in image sequences"""

    def __init__(self, image_dir, output="loop_closures.txt", config=None):
        """Initialize the loop detector

        Args:
            image_dir: Directory path containing images
            ckpt_path: Model checkpoint path
            image_size: Image resize dimensions [height width]
            batch_size: Batch size for processing
            similarity_threshold: Similarity threshold for loop closure
            top_k: Number of nearest neighbors to check for each image
            use_nms: Whether to use Non-Maximum Suppression (NMS) filtering
            nms_threshold: NMS threshold for minimum frame difference between loop pairs
            output: Output file path
        """
        self.config = config
        self.image_dir = image_dir
        salad_config = self.config["Loop"]["SALAD"]
        diagnostics = salad_config.get("diagnostics", {})
        self.ckpt_path = self.config["Weights"]["SALAD"]
        self.image_size = salad_config["image_size"]
        self.batch_size = salad_config["batch_size"]
        # This threshold belongs only to the historical single-image candidate
        # mode. Capture-cyclic selection has its own multi-view consensus
        # thresholds below. Keep the old key as a compatibility fallback for
        # archived configs.
        self.single_view_similarity_threshold = float(
            salad_config.get(
                "single_view_similarity_threshold",
                salad_config.get("similarity_threshold", 0.7),
            )
        )
        self.candidate_mode = str(
            salad_config.get("candidate_mode", "image")
        ).lower()
        self.top_k = salad_config["top_k"]
        self.min_frame_gap = salad_config.get("min_frame_gap", 10)
        self.min_gap_unit = str(salad_config.get("min_gap_unit", "image")).lower()
        self.use_nms = salad_config["use_nms"]
        self.nms_threshold = salad_config["nms_threshold"]
        self.nms_unit = str(salad_config.get("nms_unit", "image")).lower()
        self.views_per_capture = int(salad_config.get("views_per_capture", 4))
        consensus = salad_config.get("capture_cyclic_consensus", {})
        self.consensus_threshold = float(
            consensus.get("mean_similarity_threshold", 0.65)
        )
        self.consensus_min_view_similarity = float(
            consensus.get("min_view_similarity", 0.45)
        )
        self.consensus_support_similarity = float(
            consensus.get("support_similarity", 0.50)
        )
        self.consensus_min_support_views = int(
            consensus.get("min_support_views", self.views_per_capture)
        )
        self.descriptors_npz = diagnostics.get("descriptors_npz")
        self.retrieval_csv = diagnostics.get("retrieval_csv")
        self.retrieval_top_k = diagnostics.get("retrieval_top_k")
        self.capture_matrix_npz = diagnostics.get("capture_matrix_npz")
        self.capture_pairs_csv = diagnostics.get("capture_pairs_csv")
        self.output = output

        if self.min_gap_unit not in {"image", "capture"}:
            raise ValueError("Loop.SALAD.min_gap_unit must be 'image' or 'capture'")
        if self.candidate_mode not in {"image", "capture_cyclic"}:
            raise ValueError(
                "Loop.SALAD.candidate_mode must be 'image' or 'capture_cyclic'"
            )
        if self.nms_unit not in {"image", "capture"}:
            raise ValueError("Loop.SALAD.nms_unit must be 'image' or 'capture'")
        if self.views_per_capture <= 0:
            raise ValueError("Loop.SALAD.views_per_capture must be positive")
        if not 1 <= self.consensus_min_support_views <= self.views_per_capture:
            raise ValueError(
                "Loop.SALAD.capture_cyclic_consensus.min_support_views must be "
                "between 1 and views_per_capture"
            )
        if self.retrieval_top_k is not None and int(self.retrieval_top_k) < 0:
            raise ValueError("Loop.SALAD.diagnostics.retrieval_top_k must be non-negative")

        self.model = None
        self.device = None
        self.image_paths = None
        self.image_metadata = None
        self.descriptors = None
        self.loop_closures = None
        self.retrieval_similarities = None
        self.retrieval_indices = None

    def _metadata_for_image(self, image_index, image_path):
        """Return stable capture/view indices without exposing parent paths."""
        name = Path(image_path).name
        capture_match = _CAPTURE_RE.search(name) or _LEADING_CAPTURE_RE.search(name)
        view_match = _VIEW_RE.search(name)
        capture_index = (
            int(capture_match.group("capture"))
            if capture_match is not None
            else image_index // self.views_per_capture
        )
        if view_match is not None:
            view_index = int(view_match.group("view"))
        else:
            yaw_match = _YAW_RE.search(name)
            if yaw_match is None:
                view_index = image_index % self.views_per_capture
            else:
                yaw_step = 360.0 / self.views_per_capture
                view_index = round(float(yaw_match.group("yaw")) / yaw_step)
                view_index %= self.views_per_capture
        return {
            "image_index": image_index,
            "capture_index": capture_index,
            "view_index": view_index,
            "image_name": name,
        }

    def get_image_metadata(self):
        if self.image_paths is None:
            self.get_image_paths()
        if self.image_metadata is None or len(self.image_metadata) != len(self.image_paths):
            self.image_metadata = [
                self._metadata_for_image(index, path)
                for index, path in enumerate(self.image_paths)
            ]
        return self.image_metadata

    def _gap_value(self, first, second, unit):
        if unit == "image":
            return abs(first - second)
        metadata = self.get_image_metadata()
        return abs(
            metadata[first]["capture_index"] - metadata[second]["capture_index"]
        )

    def _passes_min_gap(self, first, second):
        return self._gap_value(first, second, self.min_gap_unit) > self.min_frame_gap

    def _input_transform(self, image_size=None):
        """Create image transformation function"""
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        if image_size:
            return T.Compose(
                [
                    T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ]
            )
        else:
            return T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])

    def load_model(self):
        """Load model"""
        model = VPRModel(
            backbone_arch="dinov2_vitb14",
            backbone_config={
                "num_trainable_blocks": 4,
                "return_token": True,
                "norm_layer": True,
            },
            agg_arch="SALAD",
            agg_config={
                "num_channels": 768,
                "num_clusters": 64,
                "cluster_dim": 128,
                "token_dim": 256,
            },
        )

        model.load_state_dict(torch.load(self.ckpt_path))
        model = model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"Model loaded: {self.ckpt_path}")

        self.model = model
        self.device = device
        return model, device

    def get_image_paths(self):
        """Get paths of all image files in directory"""
        image_extensions = [".jpg", ".jpeg", ".png"]
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(list(Path(self.image_dir).glob(f"*{ext}")))
            image_paths.extend(list(Path(self.image_dir).glob(f"*{ext.upper()}")))

        image_paths = sorted(image_paths)
        self.image_paths = image_paths
        self.image_metadata = None
        return image_paths

    def extract_descriptors(self):
        """Extract image feature descriptors"""
        if self.model is None or self.device is None:
            self.load_model()

        if self.image_paths is None:
            self.get_image_paths()

        transform = self._input_transform(self.image_size)
        descriptors = []

        for i in tqdm(
            range(0, len(self.image_paths), self.batch_size), desc="Extracting features"
        ):
            batch_paths = self.image_paths[i : i + self.batch_size]
            batch_imgs = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    img = transform(img)
                    batch_imgs.append(img)
                except Exception as e:
                    print(f"Error processing image {path}: {e}")
                    img = (
                        torch.zeros(3, 224, 224)
                        if self.image_size is None
                        else torch.zeros(3, self.image_size[0], self.image_size[1])
                    )
                    batch_imgs.append(img)

            batch_tensor = torch.stack(batch_imgs).to(self.device)

            with torch.no_grad():
                with torch.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16
                ):
                    batch_descriptors = self.model(batch_tensor).cpu()

            descriptors.append(batch_descriptors)

        self.descriptors = torch.cat(descriptors)
        return self.descriptors

    def _apply_nms_filter(self, loop_closures, nms_threshold):
        """Apply Non-Maximum Suppression (NMS) filtering to loop pairs"""
        if not loop_closures or nms_threshold <= 0:
            return loop_closures

        if self.nms_unit == "capture":
            metadata = self.get_image_metadata()
            filtered_loops = []
            kept_capture_pairs = []
            for idx1, idx2, sim in sorted(
                loop_closures, key=lambda item: item[2], reverse=True
            ):
                pair = tuple(
                    sorted(
                        (
                            metadata[idx1]["capture_index"],
                            metadata[idx2]["capture_index"],
                        )
                    )
                )
                if any(
                    abs(pair[0] - kept[0]) <= nms_threshold
                    and abs(pair[1] - kept[1]) <= nms_threshold
                    for kept in kept_capture_pairs
                ):
                    continue
                filtered_loops.append((idx1, idx2, sim))
                kept_capture_pairs.append(pair)
            return filtered_loops

        # Preserve the original image-index NMS byte-for-byte as the default.
        sorted_loops = sorted(loop_closures, key=lambda x: x[2], reverse=True)
        filtered_loops = []
        suppressed = set()

        max_frame = max(max(idx1, idx2) for idx1, idx2, _ in loop_closures)

        for idx1, idx2, sim in sorted_loops:
            if idx1 in suppressed or idx2 in suppressed:
                continue

            filtered_loops.append((idx1, idx2, sim))

            suppress_range = set()

            start1 = max(0, idx1 - nms_threshold)
            end1 = min(idx1 + nms_threshold + 1, idx2)
            suppress_range.update(range(start1, end1))

            start2 = max(idx1 + 1, idx2 - nms_threshold)
            end2 = min(idx2 + nms_threshold + 1, max_frame + 1)
            suppress_range.update(range(start2, end2))

            suppressed.update(suppress_range)

        return filtered_loops

    def _ensure_decending_order(self, tuples_list):
        return [(max(a, b), min(a, b), score) for a, b, score in tuples_list]

    def _diagnostic_neighbor_limit(self, image_count):
        if not self.retrieval_csv:
            return 0
        if self.retrieval_top_k is None or int(self.retrieval_top_k) == 0:
            return max(0, image_count - 1)
        return min(int(self.retrieval_top_k), max(0, image_count - 1))

    def find_loop_closures(self):
        """Find loop closures"""
        if self.descriptors is None:
            self.extract_descriptors()

        image_count = len(self.descriptors)
        embed_size = self.descriptors.shape[1]
        faiss_index = faiss.IndexFlatIP(embed_size)

        normalized_descriptors = (
            self.descriptors.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        faiss_index.add(normalized_descriptors)

        diagnostic_limit = self._diagnostic_neighbor_limit(image_count)
        neighbor_limit = max(self.top_k, diagnostic_limit)
        # Keep the historical top_k + 1 FAISS request unchanged unless an
        # explicitly requested diagnostic needs a wider ranking.
        search_count = neighbor_limit + 1
        similarities, indices = faiss_index.search(
            normalized_descriptors, search_count
        )  # +1 because self is most similar
        self.retrieval_similarities = similarities
        self.retrieval_indices = indices

        if self.candidate_mode == "capture_cyclic":
            loop_closures = self._find_capture_cyclic_loop_closures()
            if self.use_nms and self.nms_threshold > 0:
                loop_closures = self._apply_nms_filter(
                    loop_closures, self.nms_threshold
                )
            self.loop_closures = self._ensure_decending_order(loop_closures)
            return self.loop_closures

        loop_closures = []
        for i in range(image_count):
            # Skip first result (self)
            for j in range(1, self.top_k + 1):
                neighbor_idx = int(indices[i, j])
                similarity = similarities[i, j]

                if (
                    similarity > self.single_view_similarity_threshold
                    and self._passes_min_gap(i, neighbor_idx)
                ):
                    if i < neighbor_idx:
                        loop_closures.append((i, neighbor_idx, similarity))
                    else:
                        loop_closures.append((neighbor_idx, i, similarity))

        loop_closures = list(set(loop_closures))
        loop_closures.sort(key=lambda x: x[2], reverse=True)

        if self.use_nms and self.nms_threshold > 0:
            loop_closures = self._apply_nms_filter(loop_closures, self.nms_threshold)

        self.loop_closures = self._ensure_decending_order(loop_closures)
        return self.loop_closures

    @staticmethod
    def _best_cyclic_consensus(matrix, support_similarity):
        """Return the strongest yaw-preserving cyclic diagonal of a view matrix."""
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("capture view score matrix must be square")
        if not np.isfinite(matrix).all():
            return None

        view_count = matrix.shape[0]
        candidates = []
        for shift in range(view_count):
            values = np.asarray(
                [
                    matrix[view, (view + shift) % view_count]
                    for view in range(view_count)
                ],
                dtype=np.float32,
            )
            candidates.append(
                {
                    "shift": shift,
                    "values": values,
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "minimum": float(values.min()),
                    "support_views": int((values >= support_similarity).sum()),
                }
            )
        return max(candidates, key=lambda item: (item["mean"], item["median"]))

    def _find_capture_cyclic_loop_closures(self):
        """Select capture pairs supported by one consistent yaw permutation.

        This mode treats the four simultaneous yaw views as one capture. It can
        recover a revisit observed with a different body heading while rejecting
        candidates supported by only one accidental corridor view.
        """
        if self.min_gap_unit != "capture" or self.nms_unit != "capture":
            raise ValueError(
                "capture_cyclic candidate mode requires capture units for both "
                "min_gap_unit and nms_unit"
            )

        capture_indices, image_indices, _, capture_scores = self._capture_score_data()
        accepted = []
        for first in range(len(capture_indices)):
            for second in range(first + 1, len(capture_indices)):
                if (
                    abs(int(capture_indices[first]) - int(capture_indices[second]))
                    <= self.min_frame_gap
                ):
                    continue
                consensus = self._best_cyclic_consensus(
                    capture_scores[first, second], self.consensus_support_similarity
                )
                if consensus is None:
                    continue
                if consensus["mean"] <= self.consensus_threshold:
                    continue
                if consensus["minimum"] < self.consensus_min_view_similarity:
                    continue
                if consensus["support_views"] < self.consensus_min_support_views:
                    continue

                shift = int(consensus["shift"])
                values = consensus["values"]
                best_view_a = int(np.argmax(values))
                best_view_b = (best_view_a + shift) % self.views_per_capture
                image_a = int(image_indices[first, best_view_a])
                image_b = int(image_indices[second, best_view_b])
                if image_a < 0 or image_b < 0:
                    continue
                accepted.append((image_a, image_b, float(consensus["mean"])))

        accepted.sort(key=lambda item: item[2], reverse=True)
        return accepted

    def _descriptor_array(self):
        if self.descriptors is None:
            raise RuntimeError("Descriptors have not been extracted")
        if torch.is_tensor(self.descriptors):
            descriptors = self.descriptors.detach().cpu().numpy()
        else:
            descriptors = np.asarray(self.descriptors)
        return np.asarray(descriptors, dtype=np.float32)

    def save_descriptor_diagnostics(self):
        if not self.descriptors_npz:
            return
        metadata = self.get_image_metadata()
        _atomic_write_npz(
            self.descriptors_npz,
            format_version=np.asarray(1, dtype=np.int64),
            descriptors=self._descriptor_array(),
            image_indices=np.asarray(
                [item["image_index"] for item in metadata], dtype=np.int64
            ),
            capture_indices=np.asarray(
                [item["capture_index"] for item in metadata], dtype=np.int64
            ),
            view_indices=np.asarray(
                [item["view_index"] for item in metadata], dtype=np.int64
            ),
            image_names=np.asarray(
                [item["image_name"] for item in metadata], dtype=np.str_
            ),
        )
        print(f"SALAD descriptors saved to {self.descriptors_npz}")

    def _retrieval_rows(self, limit):
        if self.retrieval_indices is None or self.retrieval_similarities is None:
            raise RuntimeError("Retrieval results have not been computed")
        metadata = self.get_image_metadata()
        for query_index in range(len(metadata)):
            emitted = 0
            for neighbor_index, similarity in zip(
                self.retrieval_indices[query_index],
                self.retrieval_similarities[query_index],
            ):
                neighbor_index = int(neighbor_index)
                if neighbor_index < 0 or neighbor_index == query_index:
                    continue
                emitted += 1
                if emitted > limit:
                    break
                query = metadata[query_index]
                neighbor = metadata[neighbor_index]
                yield {
                    "query_image_index": query_index,
                    "query_capture_index": query["capture_index"],
                    "query_view_index": query["view_index"],
                    "query_image_name": query["image_name"],
                    "rank": emitted,
                    "neighbor_image_index": neighbor_index,
                    "neighbor_capture_index": neighbor["capture_index"],
                    "neighbor_view_index": neighbor["view_index"],
                    "neighbor_image_name": neighbor["image_name"],
                    "similarity": float(similarity),
                    "image_gap": abs(query_index - neighbor_index),
                    "capture_gap": abs(
                        query["capture_index"] - neighbor["capture_index"]
                    ),
                    "passes_similarity": bool(
                        similarity > self.single_view_similarity_threshold
                    ),
                    "passes_min_gap": self._passes_min_gap(
                        query_index, neighbor_index
                    ),
                }

    def save_retrieval_diagnostics(self):
        if not self.retrieval_csv:
            return
        limit = self._diagnostic_neighbor_limit(len(self.get_image_metadata()))
        fields = [
            "query_image_index",
            "query_capture_index",
            "query_view_index",
            "query_image_name",
            "rank",
            "neighbor_image_index",
            "neighbor_capture_index",
            "neighbor_view_index",
            "neighbor_image_name",
            "similarity",
            "image_gap",
            "capture_gap",
            "passes_similarity",
            "passes_min_gap",
        ]
        _atomic_write_csv(self.retrieval_csv, fields, self._retrieval_rows(limit))
        scope = "full" if self.retrieval_top_k in (None, 0) else f"top-{limit}"
        print(f"SALAD {scope} retrieval table saved to {self.retrieval_csv}")

    def _capture_score_data(self):
        metadata = self.get_image_metadata()
        capture_indices = np.asarray(
            sorted({item["capture_index"] for item in metadata}), dtype=np.int64
        )
        capture_rows = {value: index for index, value in enumerate(capture_indices)}
        image_indices = np.full(
            (len(capture_indices), self.views_per_capture), -1, dtype=np.int64
        )
        max_name_length = max(
            (len(item["image_name"]) for item in metadata), default=1
        )
        image_names = np.full(
            (len(capture_indices), self.views_per_capture),
            "",
            dtype=f"<U{max_name_length}",
        )
        for item in metadata:
            view_index = item["view_index"]
            if not 0 <= view_index < self.views_per_capture:
                raise ValueError(
                    f"View index {view_index} is outside [0, {self.views_per_capture}) "
                    f"for {item['image_name']}"
                )
            row = capture_rows[item["capture_index"]]
            if image_indices[row, view_index] >= 0:
                raise ValueError(
                    "Duplicate capture/view slot for "
                    f"capture={item['capture_index']}, view={view_index}"
                )
            image_indices[row, view_index] = item["image_index"]
            image_names[row, view_index] = item["image_name"]

        descriptors = self._descriptor_array()
        image_scores = descriptors @ descriptors.T
        flat_images = image_indices.reshape(-1)
        valid_slots = np.flatnonzero(flat_images >= 0)
        flat_scores = np.full(
            (len(flat_images), len(flat_images)), np.nan, dtype=np.float32
        )
        valid_images = flat_images[valid_slots]
        flat_scores[np.ix_(valid_slots, valid_slots)] = image_scores[
            np.ix_(valid_images, valid_images)
        ]
        capture_scores = flat_scores.reshape(
            len(capture_indices),
            self.views_per_capture,
            len(capture_indices),
            self.views_per_capture,
        ).transpose(0, 2, 1, 3)
        return capture_indices, image_indices, image_names, capture_scores

    def _capture_pair_rows(
        self, capture_indices, image_indices, image_names, capture_scores
    ):
        for first in range(len(capture_indices)):
            for second in range(first + 1, len(capture_indices)):
                matrix = capture_scores[first, second]
                finite = np.isfinite(matrix)
                row = {
                    "capture_index_a": int(capture_indices[first]),
                    "capture_index_b": int(capture_indices[second]),
                    "capture_gap": int(
                        abs(capture_indices[first] - capture_indices[second])
                    ),
                }
                for view in range(self.views_per_capture):
                    row[f"image_index_a_v{view}"] = int(image_indices[first, view])
                    row[f"image_index_b_v{view}"] = int(image_indices[second, view])
                    row[f"image_name_a_v{view}"] = str(image_names[first, view])
                    row[f"image_name_b_v{view}"] = str(image_names[second, view])
                for view_a in range(self.views_per_capture):
                    for view_b in range(self.views_per_capture):
                        row[f"score_v{view_a}_v{view_b}"] = float(
                            matrix[view_a, view_b]
                        )
                if finite.any():
                    flat_best = int(np.nanargmax(matrix))
                    best_view_a, best_view_b = np.unravel_index(
                        flat_best, matrix.shape
                    )
                    row.update(
                        best_similarity=float(matrix[best_view_a, best_view_b]),
                        best_view_a=int(best_view_a),
                        best_view_b=int(best_view_b),
                        best_image_index_a=int(image_indices[first, best_view_a]),
                        best_image_index_b=int(image_indices[second, best_view_b]),
                    )
                else:
                    row.update(
                        best_similarity=float("nan"),
                        best_view_a=-1,
                        best_view_b=-1,
                        best_image_index_a=-1,
                        best_image_index_b=-1,
                    )
                yield row

    def save_capture_diagnostics(self):
        if not self.capture_matrix_npz and not self.capture_pairs_csv:
            return
        capture_indices, image_indices, image_names, capture_scores = (
            self._capture_score_data()
        )
        if self.capture_matrix_npz:
            _atomic_write_npz(
                self.capture_matrix_npz,
                format_version=np.asarray(1, dtype=np.int64),
                capture_indices=capture_indices,
                view_indices=np.arange(self.views_per_capture, dtype=np.int64),
                image_indices=image_indices,
                image_names=image_names,
                scores=capture_scores,
            )
            print(f"SALAD capture score matrix saved to {self.capture_matrix_npz}")
        if self.capture_pairs_csv:
            fields = ["capture_index_a", "capture_index_b", "capture_gap"]
            fields.extend(
                f"image_index_{side}_v{view}"
                for side in ("a", "b")
                for view in range(self.views_per_capture)
            )
            fields.extend(
                f"image_name_{side}_v{view}"
                for side in ("a", "b")
                for view in range(self.views_per_capture)
            )
            fields.extend(
                f"score_v{view_a}_v{view_b}"
                for view_a in range(self.views_per_capture)
                for view_b in range(self.views_per_capture)
            )
            fields.extend(
                [
                    "best_similarity",
                    "best_view_a",
                    "best_view_b",
                    "best_image_index_a",
                    "best_image_index_b",
                ]
            )
            _atomic_write_csv(
                self.capture_pairs_csv,
                fields,
                self._capture_pair_rows(
                    capture_indices, image_indices, image_names, capture_scores
                ),
            )
            print(f"SALAD capture-pair table saved to {self.capture_pairs_csv}")

    def save_diagnostics(self):
        """Publish only explicitly requested, path-sanitized diagnostics."""
        self.save_descriptor_diagnostics()
        self.save_retrieval_diagnostics()
        self.save_capture_diagnostics()

    def save_results(self):
        """Save loop detection results to file"""
        if self.loop_closures is None:
            self.find_loop_closures()

        Path(self.output).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output, "w") as f:
            f.write("# Loop Detection Results (index1, index2, similarity)\n")
            f.write(f"# Min frame gap: {self.min_frame_gap}\n")
            if self.use_nms:
                f.write(f"# NMS filtering applied, threshold: {self.nms_threshold}\n")
            f.write("\n# Loop pairs:\n")
            for i, j, sim in self.loop_closures:
                f.write(f"{i}, {j}, {sim:.4f}\n")
            f.write("\n# Image path list:\n")
            for i, path in enumerate(self.image_paths):
                f.write(f"# {i}: {path}\n")

        print(f"Found {len(self.loop_closures)} loop pairs, results saved to {self.output}")
        if self.use_nms:
            print(f"NMS filtering applied, threshold: {self.nms_threshold}")

        if self.loop_closures:
            print("\nTop 10 loop pairs:")
            for i, (idx1, idx2, sim) in enumerate(self.loop_closures[:10]):
                print(f"{idx1}, {idx2}, similarity: {sim:.4f}")
                if i >= 9:
                    break

    def get_loop_list(self):
        return [(idx1, idx2) for idx1, idx2, _ in self.loop_closures]

    def run(self):
        """Run complete loop detection pipeline"""
        print("Loading model...")
        if self.model is None:
            self.load_model()

        self.get_image_paths()
        if not self.image_paths:
            print(f"No image files found in {self.image_dir}")
            return

        print(f"Found {len(self.image_paths)} image files")

        self.extract_descriptors()

        self.find_loop_closures()

        self.save_results()
        self.save_diagnostics()

        return self.loop_closures


def main():
    parser = argparse.ArgumentParser(description="Loop detection using SALAD model")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory path containing images",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=DEFAULT_SALAD_CKPT, help="Model checkpoint path"
    )
    parser.add_argument(
        "--image_size",
        nargs=2,
        type=int,
        default=[336, 336],
        help="Image resize dimensions [height width]",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument(
        "--single_view_similarity_threshold",
        "--similarity_threshold",
        dest="single_view_similarity_threshold",
        type=float,
        default=0.7,
        help=(
            "Similarity threshold for the historical single-image candidate "
            "mode; capture-cyclic mode uses its own consensus thresholds"
        ),
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of nearest neighbors to check for each image"
    )
    parser.add_argument(
        "--min_frame_gap",
        type=int,
        default=10,
        help="Minimum index separation for a loop candidate",
    )
    parser.add_argument(
        "--min_gap_unit",
        choices=("image", "capture"),
        default="image",
        help="Unit for --min_frame_gap; image preserves the historical behavior",
    )
    parser.add_argument("--output", type=str, default="loop_closures.txt", help="Output file path")
    parser.add_argument(
        "--use_nms",
        action="store_true",
        default=True,
        help="Whether to use Non-Maximum Suppression (NMS) filtering",
    )
    parser.add_argument(
        "--nms_threshold",
        type=int,
        default=25,
        help="NMS threshold for minimum frame difference between loop pairs",
    )
    parser.add_argument(
        "--nms_unit",
        choices=("image", "capture"),
        default="image",
        help="Unit for NMS; image preserves the historical behavior",
    )
    parser.add_argument(
        "--no_nms",
        action="store_true",
        help="Disable Non-Maximum Suppression filtering",
    )
    parser.add_argument(
        "--views_per_capture",
        type=int,
        default=4,
        help="Number of ordered views used to infer capture/view indices",
    )
    parser.add_argument(
        "--descriptors_npz",
        help="Optional atomic NPZ containing descriptors and path-safe indices",
    )
    parser.add_argument(
        "--retrieval_csv",
        help="Optional per-query retrieval CSV; stores image basenames, never parent paths",
    )
    parser.add_argument(
        "--retrieval_top_k",
        type=int,
        default=None,
        help="Rows per query in retrieval CSV; omitted or 0 saves the full ranking",
    )
    parser.add_argument(
        "--capture_matrix_npz",
        help="Optional atomic [capture,capture,view,view] score-matrix NPZ",
    )
    parser.add_argument(
        "--capture_pairs_csv",
        help="Optional one-row-per-capture-pair CSV with the complete view score matrix",
    )

    args = parser.parse_args()

    # LoopDetector is also used by the reconstruction pipeline and therefore
    # consumes the canonical nested config. Keep the standalone CLI on the
    # same code path instead of relying on its pre-config-refactor signature.
    config = {
        "Weights": {"SALAD": args.ckpt_path},
        "Loop": {
            "SALAD": {
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "single_view_similarity_threshold": (
                    args.single_view_similarity_threshold
                ),
                "top_k": args.top_k,
                "min_frame_gap": args.min_frame_gap,
                "min_gap_unit": args.min_gap_unit,
                "use_nms": bool(args.use_nms and not args.no_nms),
                "nms_threshold": args.nms_threshold,
                "nms_unit": args.nms_unit,
                "views_per_capture": args.views_per_capture,
                "diagnostics": {
                    "descriptors_npz": args.descriptors_npz,
                    "retrieval_csv": args.retrieval_csv,
                    "retrieval_top_k": args.retrieval_top_k,
                    "capture_matrix_npz": args.capture_matrix_npz,
                    "capture_pairs_csv": args.capture_pairs_csv,
                },
            }
        },
    }
    detector = LoopDetector(image_dir=args.image_dir, output=args.output, config=config)

    detector.run()


if __name__ == "__main__":
    main()
