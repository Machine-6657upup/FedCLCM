#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-fedclcm-static}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
REQ_FILE="${PROJECT_DIR}/requirements/fedclcm_static_runtime_3090_aligned.txt"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[FATAL] conda not found at ${CONDA_BIN}" >&2
  exit 1
fi

if ! "${CONDA_BIN}" run -n "${ENV_NAME}" python -V >/dev/null 2>&1; then
  echo "[ENV] creating ${ENV_NAME} with python=${PYTHON_VERSION}"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
else
  echo "[ENV] reusing existing env ${ENV_NAME}"
fi

echo "[ENV] upgrading pip toolchain"
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel

echo "[ENV] installing torch stack"
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install \
  torch==2.4.1 \
  torchvision==0.19.1 \
  torchaudio==2.4.1 \
  --index-url "${PYTORCH_INDEX_URL}"

echo "[ENV] installing runtime requirements"
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install -r "${REQ_FILE}"

echo "[ENV] verifying core imports"
"${CONDA_BIN}" run -n "${ENV_NAME}" python -c "import h5py, numpy, sklearn, torch, torchvision, ujson; print('python ok'); print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('cuda_available', torch.cuda.is_available())"

echo "[READY] env=${ENV_NAME}"
