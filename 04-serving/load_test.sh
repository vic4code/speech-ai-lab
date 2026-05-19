#!/usr/bin/env bash
# Load test Triton whisper endpoint with perf_analyzer (Triton SDK tool).
#
# Measures: throughput (infer/s), P50/P99 latency under concurrent load.
# perf_analyzer is included in the Triton SDK container:
#   nvcr.io/nvidia/tritonserver:<ver>-py3-sdk

set -euo pipefail

SERVER_URL="${TRITON_SERVER:-localhost:8001}"
MODEL="${TRITON_MODEL:-whisper}"
CONCURRENCY="${CONCURRENCY:-8}"
DURATION_S="${DURATION_S:-60}"

echo "=== Triton Load Test ==="
echo "Server: ${SERVER_URL} | Model: ${MODEL} | Concurrency: ${CONCURRENCY} | Duration: ${DURATION_S}s"

perf_analyzer \
  -m "${MODEL}" \
  -u "${SERVER_URL}" \
  --concurrency-range "${CONCURRENCY}" \
  --measurement-interval $((DURATION_S * 1000)) \
  --input-data zero \
  --shape audio_signal:16000 \
  --shape sample_rate:1 \
  --protocol grpc \
  -f results/load_test_results.csv

echo ""
echo "=== Results saved to results/load_test_results.csv ==="
