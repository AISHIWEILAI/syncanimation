#!/usr/bin/env bash
# Full training in screen. Usage: bash scripts/train_screen.sh [ID]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ID="${1:-May}"
CONDA_SH="${CONDA_SH:-/home/shidang/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="syncanimation"
TORCH_LIB="/home/shidang/anaconda3/envs/${CONDA_ENV}/lib/python3.8/site-packages/torch/lib"
CUDA118_LIB="/usr/local/cuda-11.8/lib64"

SESSION="syncanimation_train_${ID}"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/train_${ID}.log"

mkdir -p "${LOG_DIR}"

if [[ ! -d "${ROOT}/data/${ID}" ]]; then
    echo "[ERROR] data directory not found: data/${ID}"
    echo "Run preprocessing first: bash scripts/preprocess.sh ${ID}"
    exit 1
fi

if screen -ls | grep -qF "${SESSION}"; then
    echo "[WARN] screen session '${SESSION}' already running"
    echo "  log: ${LOG_FILE}"
    exit 1
fi

if [[ -d "${ROOT}/model/${ID}" ]]; then
    backup="${ROOT}/model/${ID}_backup_$(date +%Y%m%d_%H%M%S)"
    echo "Archiving ${ROOT}/model/${ID} -> ${backup}"
    mv "${ROOT}/model/${ID}" "${backup}"
fi

screen -dmS "${SESSION}" bash -lc "
    set -eo pipefail
    source '${CONDA_SH}'
    conda activate '${CONDA_ENV}'
    export LD_LIBRARY_PATH='${TORCH_LIB}:${CUDA118_LIB}:/usr/local/cuda/lib64:'\${LD_LIBRARY_PATH:-}
    cd '${ROOT}'
    exec bash scripts/train.sh '${ID}' all 2>&1 | tee -a '${LOG_FILE}'
"

echo "[INFO] started screen session: ${SESSION}"
echo "[INFO] subject: ${ID}"
echo "[INFO] log: ${LOG_FILE}"
