#!/usr/bin/env bash
# Profile Whisper inference with NVIDIA Nsight Systems.
#
# Produces a .nsys-rep file that can be opened in the Nsight Systems GUI
# to inspect kernel timelines, memory transfers, and CUDA API calls.
#
# Usage:
#   chmod +x 03-inference-optimization/05_nsight_profile.sh
#   ./03-inference-optimization/05_nsight_profile.sh
#
# Requirements:
#   nsys (NVIDIA Nsight Systems CLI) installed on the instance.
#   Typically available in NVIDIA NGC containers or via:
#     apt-get install nsight-systems-cli

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/results"
REPORT_NAME="whisper_profile"

mkdir -p "${OUTPUT_DIR}"

echo "=== Nsight Systems profiling of Whisper large-v3 ==="
echo "Output: ${OUTPUT_DIR}/${REPORT_NAME}.nsys-rep"

nsys profile \
  --output="${OUTPUT_DIR}/${REPORT_NAME}" \
  --trace=cuda,nvtx,cudnn,cublas \
  --capture-range=cudaProfilerApi \
  --stop-on-range-end=true \
  --force-overwrite=true \
  python - <<'PYEOF'
import torch
import whisper
import numpy as np
from datasets import load_dataset
from torch.cuda import nvtx

model = whisper.load_model("large-v3", device="cuda").half()
model.eval()

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:1]")
audio = np.array(ds[0]["audio"]["array"], dtype=np.float32)

# Warmup outside profiler range
for _ in range(3):
    with torch.no_grad():
        model.transcribe(audio, language="en", fp16=True)

# Signal profiler to start capturing
torch.cuda.cudart().cudaProfilerStart()

for i in range(5):
    nvtx.range_push(f"transcribe_iter_{i}")
    with torch.no_grad():
        model.transcribe(audio, language="en", fp16=True)
    torch.cuda.synchronize()
    nvtx.range_pop()

torch.cuda.cudart().cudaProfilerStop()
print("Profiling complete.")
PYEOF

echo ""
echo "=== Profile complete ==="
echo "Open ${OUTPUT_DIR}/${REPORT_NAME}.nsys-rep in Nsight Systems GUI"
echo "or run: nsys stats ${OUTPUT_DIR}/${REPORT_NAME}.nsys-rep"
