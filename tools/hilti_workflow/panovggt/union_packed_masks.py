#!/usr/bin/env python3
"""Union a packed ERP mask bundle with per-frame ERP NPY masks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-packed", type=Path, required=True)
    parser.add_argument("--secondary-npy-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    bundle = np.load(args.primary_packed, allow_pickle=False)
    shape = tuple(int(value) for value in bundle["shape"])
    stems = [str(value) for value in bundle["stems"]]
    masks = np.unpackbits(bundle["masks"], axis=1, count=int(np.prod(shape[1:])))
    masks = masks.reshape(shape).astype(bool)
    secondary = []
    for stem in stems:
        path = args.secondary_npy_dir / f"{stem}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        mask = np.load(path, allow_pickle=False).astype(bool)
        if mask.shape != shape[1:]:
            raise ValueError(f"Shape mismatch {mask.shape} for {path}; expected {shape[1:]}")
        secondary.append(mask)
    other = np.stack(secondary)
    before = int(np.count_nonzero(masks))
    masks |= other
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, masks=np.packbits(masks.reshape(len(masks), -1), axis=1), shape=np.asarray(shape, np.int32), stems=np.asarray(stems))
    payload = {
        "primary": str(args.primary_packed),
        "secondary_npy_dir": str(args.secondary_npy_dir),
        "frames": len(stems),
        "shape": list(shape),
        "primary_fraction": before / masks.size,
        "secondary_fraction": float(other.mean()),
        "union_fraction": float(masks.mean()),
        "pixels_added": int(np.count_nonzero(masks)) - before,
        "output": str(args.output),
        "sha256": sha256(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
