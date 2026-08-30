#!/usr/bin/env bash
# Three-stage training + inference. Usage: bash scripts/train.sh [ID] [all|torso|face|lips|infer]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

CONDA_SH="${CONDA_SH:-/home/shidang/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="syncanimation"
TORCH_LIB="/home/shidang/anaconda3/envs/${CONDA_ENV}/lib/python3.8/site-packages/torch/lib"
CUDA118_LIB="/usr/local/cuda-11.8/lib64"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${TORCH_LIB}:${CUDA118_LIB}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

STAGES="all|torso|face|lips|infer"
ID="${1:-May}"
STAGE="${2:-all}"

if [[ "${ID}" =~ ^(${STAGES})$ ]]; then
    STAGE="${ID}"
    ID="May"
fi

if [[ ! "${STAGE}" =~ ^(${STAGES})$ ]]; then
    echo "Usage: bash scripts/train.sh [ID] [all|torso|face|lips|infer]"
    exit 1
fi

datapath="data/${ID}"
if [[ ! -d "${datapath}" ]]; then
    echo "[ERROR] data directory not found: ${datapath}"
    echo "Run preprocessing first: bash scripts/preprocess.sh ${ID}"
    exit 1
fi

TORSO_WS="model/${ID}/${ID}_trial_torso_audio"
FACE_WS="model/${ID}/${ID}_trial_audio"

latest_torso_ckpt() {
    local ckpt
    ckpt=$(ls -1 "${TORSO_WS}/checkpoints" | sort | tail -n 2 | head -n 1)
    cp "${TORSO_WS}/checkpoints/${ckpt}" "${TORSO_WS}/${ckpt}"
    echo "${TORSO_WS}/${ckpt}"
}

run_torso() {
    echo "========== [${ID}] Torso training (150k) =========="
    python audio_main.py --path "${datapath}" --fps 25 --asr_model hubert \
        --iters 150000 --workspace "${TORSO_WS}" \
        --torso --bs_au45
    python -c "import torch; torch.cuda.empty_cache()"
}

run_face() {
    echo "========== [${ID}] Face training (120k) =========="
    local torso_ckpt
    torso_ckpt="$(latest_torso_ckpt)"
    python audio_main.py --path "${datapath}" --fps 25 --asr_model hubert \
        --iters 120000 --workspace "${FACE_WS}" \
        --special --bs_loss --bs_start \
        --patch_size 64 --bs_au45 \
        --torso_ckpt "${torso_ckpt}"
    python -c "import torch; torch.cuda.empty_cache()"
}

run_infer() {
    echo "========== [${ID}] Inference =========="
    python audio_main.py --path "${datapath}" --fps 25 --asr_model hubert \
        --test \
        --workspace "${FACE_WS}" \
        --infer data/inference/c-eng-chi-chi_Xhu.npy \
        --aud data/inference/c-eng-chi-chi_hu.npy \
        --torso \
        --special \
        --bs_au45
    python -c "import torch; torch.cuda.empty_cache()"
}

run_lips() {
    echo "========== [${ID}] Lip finetune (160k) =========="
    local torso_ckpt
    torso_ckpt="$(latest_torso_ckpt)"
    python audio_main.py --path "${datapath}" --fps 25 --asr_model hubert \
        --iters 160000 --workspace "${FACE_WS}" \
        --special --bs_start --finetune_lips \
        --patch_size 64 --bs_au45 \
        --torso_ckpt "${torso_ckpt}"
    python -c "import torch; torch.cuda.empty_cache()"
}

case "${STAGE}" in
    torso) run_torso ;;
    face)  run_face ;;
    lips)  run_lips ;;
    infer) run_infer ;;
    all)
        run_torso
        run_face
        run_infer
        run_lips
        run_infer
        ;;
esac

echo "========== Done: ${ID} ${STAGE} =========="
