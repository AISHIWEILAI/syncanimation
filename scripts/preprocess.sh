#!/usr/bin/env bash
# Preprocess one subject. Usage: bash scripts/preprocess.sh [ID]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

ID="${1:-May}"
VIDEO="data/${ID}/${ID}.mp4"

if [[ ! -d "data/${ID}" ]]; then
    echo "[ERROR] data directory not found: data/${ID}"
    echo "Create data/${ID}/ and place ${ID}.mp4 inside."
    exit 1
fi

if [[ ! -f "${VIDEO}" ]]; then
    echo "[ERROR] video not found: ${VIDEO}"
    exit 1
fi

python data_utils/process.py --path "${VIDEO}" --task -1 --asr hubert
