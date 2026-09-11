#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PANOVGGT_ROOT="${PANOVGGT_ROOT:-$REPO_ROOT/.external/PanoVGGT}"
VERIFY_ONLY=0
INCLUDE_LOCAL_GDINO=0
PREFETCH_HF=0

for arg in "$@"; do
  case "$arg" in
    --verify-only) VERIFY_ONLY=1 ;;
    --include-local-grounding-dino) INCLUDE_LOCAL_GDINO=1 ;;
    --prefetch-hf) PREFETCH_HF=1 ;;
    -h|--help)
      sed -n '1,34p' "$0"
      exit 0
      ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

download_verified() {
  local url="$1" target="$2" expected="$3"
  mkdir -p "$(dirname "$target")"
  if [[ -f "$target" ]] && echo "$expected  $target" | sha256sum -c --status; then
    echo "[ok] $target"
    return
  fi
  if [[ "$VERIFY_ONLY" == "1" ]]; then
    echo "[missing-or-invalid] $target" >&2
    return 1
  fi
  echo "[download] $url"
  curl -fL --retry 3 --retry-delay 5 -C - -o "$target.part" "$url"
  echo "$expected  $target.part" | sha256sum -c -
  mv -f "$target.part" "$target"
}

DA3_BASE="https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1/resolve/main"
download_verified "$DA3_BASE/config.json" "$REPO_ROOT/da3_streaming/weights/config.json" "09adf89474017e717bc05aa86fd3a378708ba8914b036d61874eced328069468"
download_verified "$DA3_BASE/model.safetensors" "$REPO_ROOT/da3_streaming/weights/model.safetensors" "8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c"
download_verified "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt" "$REPO_ROOT/da3_streaming/weights/dino_salad.ckpt" "6b3f1720954293e83da6966c5cfcfc6713200d7fefadcca76fc51aeb80b3cada"
download_verified "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" "$REPO_ROOT/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt" "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
download_verified "https://huggingface.co/YijingGuo/PanoVGGT/resolve/main/model.pt" "$PANOVGGT_ROOT/checkpoints/model.pt" "4adab888064ef206c20ab42c08c5f973e3bd98a813fa2c3d849b4d445e278670"

if [[ "$INCLUDE_LOCAL_GDINO" == "1" ]]; then
  download_verified "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" "$REPO_ROOT/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth" "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799"
fi

if [[ "$PREFETCH_HF" == "1" ]]; then
  if [[ "$VERIFY_ONLY" == "1" ]]; then
    echo "[note] --prefetch-hf is ignored with --verify-only"
  else
    conda run -n gsam2 python -c "from transformers import AutoModelForZeroShotObjectDetection,AutoProcessor; r='12bdfa3120f3e7ec7b434d90674b3396eccf88eb'; m='IDEA-Research/grounding-dino-base'; AutoProcessor.from_pretrained(m,revision=r); AutoModelForZeroShotObjectDetection.from_pretrained(m,revision=r)"
  fi
fi

echo "[done] required HILTI model assets are ready"
