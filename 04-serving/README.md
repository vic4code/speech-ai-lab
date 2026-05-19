# Phase 4 — Model Serving with Triton Inference Server

Deploy Whisper on NVIDIA Triton for production-grade, multi-model, multi-tenant serving.

## Scripts

| File | Purpose |
|------|---------|
| `model_repository/whisper/config.pbtxt` | Triton model configuration |
| `triton_client.py` | Python gRPC client for sending audio to Triton |
| `load_test.sh` | `locust`-based load test: measure throughput and P99 latency |
| `nim_comparison.py` | Compare Triton latency against NVIDIA NIM microservice |

## How to Run

```bash
# Start Triton server (Docker)
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/04-serving/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models

# Send a test request
python 04-serving/triton_client.py --audio data/sample_audio/test.wav

# Load test (100 concurrent users, 60 s)
bash 04-serving/load_test.sh

# Compare against NIM
python 04-serving/nim_comparison.py
```
