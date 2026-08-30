#!/usr/bin/env bash
# Preprocess in screen. Usage: bash scripts/preprocess_screen.sh [ID]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ID="${1:-May}"
START_TASK="${2:-3}"
CONDA_SH="${CONDA_SH:-/home/shidang/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="syncanimation"
TORCH_LIB="/home/shidang/anaconda3/envs/${CONDA_ENV}/lib/python3.8/site-packages/torch/lib"
CUDA118_LIB="/usr/local/cuda-11.8/lib64"

SESSION="syncanimation_preprocess_${ID}"
VIDEO="${ROOT}/data/${ID}/${ID}.mp4"
DATA_DIR="${ROOT}/data/${ID}"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/preprocess_${ID}.log"

mkdir -p "${LOG_DIR}"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "[ERROR] data directory not found: data/${ID}"
    echo "Create data/${ID}/ and place ${ID}.mp4 inside."
    exit 1
fi

if screen -ls | grep -q "\.${SESSION}[[:space:]]"; then
    echo "[WARN] screen session '${SESSION}' already running"
    echo "  log: ${LOG_FILE}"
    exit 1
fi

if [[ -f "${LOG_FILE}" ]]; then
    mv "${LOG_FILE}" "${LOG_FILE}.$(date '+%Y%m%d_%H%M%S').bak"
fi

screen -dmS "${SESSION}" bash -lc "
    set -eo pipefail
    source '${CONDA_SH}'
    conda activate '${CONDA_ENV}'
    export LD_LIBRARY_PATH='${TORCH_LIB}:${CUDA118_LIB}:/usr/local/cuda/lib64'
    cd '${ROOT}'
    exec >> '${LOG_FILE}' 2>&1
    echo '========================================'
    echo '[INFO] preprocess start: '\$(date '+%F %T')
    echo '[INFO] subject: ${ID}'
    echo '[INFO] video:   ${VIDEO}'
    echo '[INFO] start_task: ${START_TASK}'
    python -c \"import torch; print('[INFO] torch', torch.__version__, 'cuda', torch.cuda.is_available())\"
    echo '========================================'

    if [[ ! -f '${VIDEO}' ]]; then
        echo '[ERROR] video not found: ${VIDEO}'
        exit 1
    fi

    for f in aud.wav aud_hu.npy aud_Xhu.npy; do
        if [[ -f '${DATA_DIR}/'\${f} ]]; then
            echo '[INFO] reuse existing data/${ID}/'\${f}
        else
            echo '[WARN] missing data/${ID}/'\${f}
        fi
    done

    tasks=()
    for t in 1 2 3 4 5 6 7 8 9 10; do
        if [[ \$t -ge ${START_TASK} ]]; then
            tasks+=(\$t)
        fi
    done
    filtered=()
    for t in \"\${tasks[@]}\"; do
        if [[ \$t -eq 1 ]]; then
            echo '[INFO] skip task 1 (audio features already prepared)'
            continue
        fi
        filtered+=(\$t)
    done

    for task in \"\${filtered[@]}\"; do
        echo ''
        echo '[INFO] ===== task '\${task}' start: '\$(date '+%F %T')' ====='
        python data_utils/process.py --path '${VIDEO}' --task \${task} --asr hubert
        echo '[INFO] ===== task '\${task}' done ====='
    done

    echo ''
    echo '[INFO] preprocess finished: '\$(date '+%F %T')
    if [[ -f '${DATA_DIR}/transforms_train.json' ]]; then
        echo '[INFO] transforms_train.json ready'
    else
        echo '[WARN] transforms_train.json not found, check log'
    fi
"

echo "[INFO] started screen session: ${SESSION}"
echo "[INFO] subject: ${ID}"
echo "[INFO] log: ${LOG_FILE}"
