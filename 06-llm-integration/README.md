# Phase 6 — LLM Integration

Chain Whisper ASR output into LLM pipelines for summarization, RAG, and real-time streaming.

## Scripts

| Script | Purpose |
|--------|---------|
| `asr_llm_pipeline.py` | ASR → LLM pipeline: transcribe then summarize with Claude |
| `rag_transcripts.py` | Build a vector store from transcripts; answer questions via RAG |
| `streaming_pipeline.py` | Real-time streaming ASR with token-by-token LLM output |

## How to Run

```bash
export ANTHROPIC_API_KEY=<your-key>

# Transcribe audio then summarize
python 06-llm-integration/asr_llm_pipeline.py --audio data/sample_audio/meeting.wav

# Build RAG index from a directory of audio files
python 06-llm-integration/rag_transcripts.py --audio-dir data/sample_audio --query "What were the action items?"

# Streaming pipeline
python 06-llm-integration/streaming_pipeline.py --audio data/sample_audio/meeting.wav
```
