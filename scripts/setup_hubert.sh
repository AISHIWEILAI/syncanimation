#!/usr/bin/env bash
# Download HuBERT weights to data_utils/facebook/
set -euo pipefail
cd "$(dirname "$0")/.."

DST="data_utils/facebook"
mkdir -p "${DST}"

echo "[INFO] Downloading facebook/hubert-large-ls960-ft weights..."
python - <<'PY'
from transformers import HubertModel, Wav2Vec2Processor
dst = "data_utils/facebook"
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
proc = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model.save_pretrained(dst)
proc.save_pretrained(dst)
print("[INFO] Saved to", dst)
PY

echo "[INFO] Done. Verify with:"
echo "  python data_utils/hubert.py --wav data/May/aud.wav"
