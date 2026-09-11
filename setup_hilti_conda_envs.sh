#!/usr/bin/env bash
set -euo pipefail

# Create the two environments used by the HILTI workflow without mixing their
# dependencies. This script is intentionally not run by the workflow itself.
#
# Defaults can be overridden from the shell:
#   DA3_ENV=da3 GSAM2_ENV=gsam2 PYTORCH_CUDA=12.8 ./setup_hilti_conda_envs.sh
# CPU-only reconstruction is not supported; evaluation can run on CPU.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GSAM2_ROOT="$REPO_ROOT/Grounded-SAM-2"
PANOVGGT_ROOT="${PANOVGGT_ROOT:-$REPO_ROOT/.external/PanoVGGT}"

DA3_ENV="${DA3_ENV:-da3}"
GSAM2_ENV="${GSAM2_ENV:-gsam2}"
# PYTHON_VERSION remains a backward-compatible override for older commands.
DA3_PYTHON_VERSION="${DA3_PYTHON_VERSION:-${PYTHON_VERSION:-3.11}}"
GSAM2_PYTHON_VERSION="${GSAM2_PYTHON_VERSION:-${PYTHON_VERSION:-3.10}}"
PYTORCH_CUDA="${PYTORCH_CUDA:-12.8}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.7.1}"
XFORMERS_VERSION="${XFORMERS_VERSION:-0.0.31.post1}"
INSTALL_XFORMERS="${INSTALL_XFORMERS:-1}"
FAISS_GPU_CU12_VERSION="${FAISS_GPU_CU12_VERSION:-1.10.0}"
FORCE="${FORCE:-0}"

MANAGED_REQUIREMENTS_REGEX='^[[:space:]]*(faiss-gpu|torch|torchvision|torchaudio|xformers)([=<>!~[:space:]]|$)'

# The HILTI mask pipeline uses Hugging Face Grounding DINO by default, so the
# local GroundingDINO CUDA extension is not needed for reproducible mask export.
INSTALL_LOCAL_GROUNDING_DINO="${INSTALL_LOCAL_GROUNDING_DINO:-0}"

# SAM2's optional CUDA extension is useful but not required for the batch image
# mask workflow. Keeping it off by default avoids compiler/CUDA-version coupling.
SAM2_BUILD_CUDA="${SAM2_BUILD_CUDA:-0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH. Source your conda init script first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

env_exists() {
  conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "$1"
}

create_env() {
  local env_name="$1" python_version="$2"
  if env_exists "$env_name"; then
    if [[ "$FORCE" == "1" ]]; then
      conda env remove -n "$env_name" -y
    else
      echo "Conda env '$env_name' already exists; skipping creation. Set FORCE=1 to recreate it."
      return 1
    fi
  fi
  conda create -n "$env_name" -y "python=$python_version"
}

install_pytorch() {
  local env_name="$1" install_xformers="${2:-0}"
  case "$PYTORCH_CUDA" in
    12.8|12.6) ;;
    *)
      echo "Use PYTORCH_CUDA=12.8 (tested) or 12.6 with a compatible NVIDIA driver." >&2
      return 1
      ;;
  esac
  local cuda_tag="cu${PYTORCH_CUDA/./}"
  local torch_index="https://download.pytorch.org/whl/$cuda_tag"
  local packages=(
    "torch==${PYTORCH_VERSION}+${cuda_tag}"
    "torchvision==${TORCHVISION_VERSION}+${cuda_tag}"
    "torchaudio==${TORCHAUDIO_VERSION}+${cuda_tag}"
  )
  if [[ "$install_xformers" == "1" && "$INSTALL_XFORMERS" == "1" ]]; then
    packages+=("xformers==${XFORMERS_VERSION}")
  fi
  conda run -n "$env_name" python -m pip install \
    --index-url "$torch_index" "${packages[@]}"
}

verify_pytorch() {
  local env_name="$1"
  conda run -n "$env_name" python -c \
    'import sys, torch; expected = sys.argv[1]; assert torch.__version__.split("+")[0] == expected, (torch.__version__, expected)' \
    "$PYTORCH_VERSION"
}

setup_da3() {
  echo "==> Setting up DA3 env: $DA3_ENV"
  if ! create_env "$DA3_ENV" "$DA3_PYTHON_VERSION"; then
    return
  fi

  install_pytorch "$DA3_ENV" 1
  conda run -n "$DA3_ENV" python -m pip install -U pip setuptools wheel
  conda run -n "$DA3_ENV" python -m pip install -e "$REPO_ROOT"


  # Keep FAISS on the same CUDA 12 runtime and NumPy ABI as PyTorch. Recent
  # Conda FAISS builds otherwise pull CUDA 13 and NumPy 2 into this env.
  if [[ "$PYTORCH_CUDA" == "12.8" ]]; then
    conda run -n "$DA3_ENV" python -m pip install \
      "faiss-gpu-cu12==${FAISS_GPU_CU12_VERSION}"
  else
    # Retrieval uses IndexFlatIP on CPU; avoid a second CUDA dependency solver.
    conda run -n "$DA3_ENV" python -m pip install "faiss-cpu==1.10.0" "numpy<2"
  fi

  local tmp_req
  tmp_req="$(mktemp)"
  # PyTorch, xFormers, and FAISS are pinned above for one compatible CUDA/ABI
  # stack; do not let legacy requirements replace those resolved versions.
  grep -vE "$MANAGED_REQUIREMENTS_REGEX" \
    "$REPO_ROOT/da3_streaming/requirements.txt" > "$tmp_req"
  conda run -n "$DA3_ENV" python -m pip install -r "$tmp_req"
  conda run -n "$DA3_ENV" python -m pip install rosbags

  if [[ -f "$PANOVGGT_ROOT/requirements.txt" ]]; then
    local pano_req
    pano_req="$(mktemp)"
    # Keep the PyTorch stack selected above and install only PanoVGGT extras.
    grep -vE "$MANAGED_REQUIREMENTS_REGEX" \
      "$PANOVGGT_ROOT/requirements.txt" > "$pano_req"
    conda run -n "$DA3_ENV" python -m pip install -r "$pano_req" pypose
    rm -f "$pano_req"
  else
    echo "[note] Run ./setup_external_repos.sh first to enable PanoVGGT."
  fi

  rm -f "$tmp_req"
  conda run -n "$DA3_ENV" python -m pip check
  verify_pytorch "$DA3_ENV"
}

setup_gsam2() {
  echo "==> Setting up Grounded-SAM-2 env: $GSAM2_ENV"
  if ! create_env "$GSAM2_ENV" "$GSAM2_PYTHON_VERSION"; then
    return
  fi

  install_pytorch "$GSAM2_ENV" 0
  conda run -n "$GSAM2_ENV" python -m pip install -U pip setuptools wheel

  SAM2_BUILD_CUDA="$SAM2_BUILD_CUDA" \
    conda run -n "$GSAM2_ENV" python -m pip install -e "$GSAM2_ROOT"

  conda run -n "$GSAM2_ENV" python -m pip install \
    "transformers==4.45.2" addict yapf timm "supervision>=0.22.0" pycocotools \
    opencv-python matplotlib scipy

  if [[ "$INSTALL_LOCAL_GROUNDING_DINO" == "1" ]]; then
    conda run -n "$GSAM2_ENV" python -m pip install --no-build-isolation -e "$GSAM2_ROOT/grounding_dino"
  fi
  conda run -n "$GSAM2_ENV" python -m pip check
  verify_pytorch "$GSAM2_ENV"
}

setup_da3
setup_gsam2

cat <<EOF

Done.

Created/checked:
  DA3 env:    $DA3_ENV
  GSAM2 env:  $GSAM2_ENV

Integrated repo paths:
  Grounded-SAM-2: $GSAM2_ROOT
  HILTI challenge: $REPO_ROOT/hilti-trimble-slam-challenge-2026

EOF
