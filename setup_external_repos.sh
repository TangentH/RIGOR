#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-$ROOT/.external}"
PANO_ROOT="${PANOVGGT_ROOT:-$EXTERNAL_ROOT/PanoVGGT}"
PANO_URL="https://github.com/YijingGuo-June/PanoVGGT.git"
PANO_COMMIT="556bb7d2ec2d02bd3ee4ed535542e74290ba22cf"

mkdir -p "$EXTERNAL_ROOT"

if [[ ! -d "$PANO_ROOT/.git" ]]; then
  git clone "$PANO_URL" "$PANO_ROOT"
fi

git -C "$PANO_ROOT" fetch --tags origin
git -C "$PANO_ROOT" checkout --detach "$PANO_COMMIT"

actual="$(git -C "$PANO_ROOT" rev-parse HEAD)"
if [[ "$actual" != "$PANO_COMMIT" ]]; then
  echo "PanoVGGT commit mismatch: $actual" >&2
  exit 1
fi

echo "[ok] PanoVGGT $actual -> $PANO_ROOT"
