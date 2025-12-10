#!/usr/bin/env bash
# download_model_weights.sh
#
# Usage:
#   bash download_model_weights.sh               # Download into ./tool_weights
#   bash download_model_weights.sh /path/to/dir  # Custom download root directory
#
# The final directory structure will look like:
#   MODELS_ROOT/
#     af2/
#       params_model_{1..5}.npz
#       params_model_{1..5}_ptm.npz
#       params_model_{1..5}_multimer_v3.npz
#       LICENSE
#     esmfold/
#       config.json
#       pytorch_model.bin
#       ...
#     mpnn/
#       ca_model_weights/
#       soluble_model_weights/
#       vanilla_model_weights/
#
# After downloading, update pxdbench/globals.py to point to these locations.

set -euo pipefail

# Set root directory for storing all model weights
MODELS_ROOT="${1:-$(pwd)/tool_weights}"

AF2_DIR="${MODELS_ROOT}/af2"
ESMFOLD_DIR="${MODELS_ROOT}/esmfold"
MPNN_DIR="${MODELS_ROOT}/mpnn"

echo "Model root directory: ${MODELS_ROOT}"
mkdir -p "${AF2_DIR}" "${ESMFOLD_DIR}" "${MPNN_DIR}"

# ===============================
# Git LFS Hard Check
# ===============================

if ! command -v git-lfs >/dev/null 2>&1 && ! command -v git-lfs >/dev/null 2>&1; then
  echo -e "\n❌ ERROR: git-lfs is NOT installed."
  echo
  echo "Large model weights are managed by Git LFS."
  echo "Without git-lfs, all downloaded weight files will be INVALID."
  echo
  echo "✅ Please install git-lfs first:"
  echo
  echo "  Ubuntu/Debian:"
  echo "    sudo apt install git-lfs"
  echo
  echo "  macOS (brew):"
  echo "    brew install git-lfs"
  echo
  echo "  Then run:"
  echo "    git lfs install"
  echo
  exit 1
fi

git lfs install >/dev/null
echo "✅ git-lfs detected."


########################################
# 1. AlphaFold2 parameters
########################################
echo "==> Downloading AlphaFold2 parameters ..."

AF2_TAR="alphafold_params_2022-12-06.tar"
AF2_URL="https://storage.googleapis.com/alphafold/${AF2_TAR}"

# If AF2 parameters already exist, skip download
if compgen -G "${AF2_DIR}/params_model_1*.npz" > /dev/null; then
  echo "  AlphaFold2 params appear to already exist — skipping download."
else
  tmp_tar="${MODELS_ROOT}/${AF2_TAR}"
  echo "  Downloading from: ${AF2_URL}"
  curl -L "${AF2_URL}" -o "${tmp_tar}"

  echo "  Extracting to: ${AF2_DIR}"
  tar -xf "${tmp_tar}" -C "${AF2_DIR}"
  rm -f "${tmp_tar}"

  echo "  AlphaFold2 parameters downloaded successfully."
fi

########################################
# 2. ESMFold weights (HuggingFace)
########################################
echo "==> Downloading ESMFold weights (HuggingFace, requires git-lfs) ..."

# Check git installation
if ! command -v git >/dev/null 2>&1; then
  echo "  ERROR: git is not installed. Please install git and re-run the script." >&2
  exit 1
fi

# If folder already cloned, just update it
if [ -d "${ESMFOLD_DIR}/.git" ]; then
  echo "  ESMFold directory already exists — pulling latest..."
  (cd "${ESMFOLD_DIR}" && git pull --ff-only || true)
else
  echo "  Cloning facebook/esmfold_v1 into ${ESMFOLD_DIR}"
  git clone https://huggingface.co/facebook/esmfold_v1 "${ESMFOLD_DIR}"
fi

echo "  ESMFold weights are ready in: ${ESMFOLD_DIR}"

########################################
# 3. ProteinMPNN weights
########################################
echo "==> Downloading ProteinMPNN weights ..."

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

echo "  Cloning dauparas/ProteinMPNN (shallow clone)..."
git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git "${TMP_DIR}"

# Copy each weight directory
for subdir in ca_model_weights soluble_model_weights vanilla_model_weights; do
  src="${TMP_DIR}/${subdir}"
  dst="${MPNN_DIR}/${subdir}"

  if [ -d "${dst}" ]; then
    echo "  ${subdir} already exists — skipping."
  else
    echo "  Copying ${subdir} → ${dst}"
    mkdir -p "${MPNN_DIR}"
    cp -r "${src}" "${dst}"
  fi
done

echo "  ProteinMPNN weights are ready in: ${MPNN_DIR}"

########################################

echo "==> All downloads completed."
echo "Model weight directories:"
echo "  AF2:      ${AF2_DIR}"
echo "  ESMFold:  ${ESMFOLD_DIR}"
echo "  MPNN:     ${MPNN_DIR}"
echo
echo "Remember to update pxdbench/globals.py to reflect these paths."
