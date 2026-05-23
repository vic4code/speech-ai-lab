"""
Build a RAG (Retrieval-Augmented Generation) system over audio transcripts.

Pipeline:
  1. Transcribe all audio files in a directory with Whisper
  2. Chunk transcripts into overlapping passages
  3. Embed passages with a sentence-transformer model
  4. Store in a FAISS vector index
  5. Answer user queries by retrieving top-k passages and prompting Claude

Demonstrates: offline indexing, semantic retrieval, grounded generation to
reduce LLM hallucination on domain-specific content.
"""

import argparse
import os
from pathlib import Path
from typing import Any

import anthropic
import faiss
import numpy as np
import soundfile as sf
import torch
import whisper
from sentence_transformers import SentenceTransformer
from rich.console import Console

console = Console()

CHUNK_WORDS = 100
CHUNK_OVERLAP = 20
TOP_K = 5


def transcribe_directory(audio_dir: Path, whisper_model_name: str) -> dict[str, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(whisper_model_name, device=device)
    transcripts: dict[str, str] = {}

    for audio_file in sorted(audio_dir.glob("*.wav")) + sorted(audio_dir.glob("*.flac")):
        console.print(f"Transcribing {audio_file.name} ...")
        audio, _ = sf.read(str(audio_file), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        result = model.transcribe(audio, language="en", fp16=(device == "cuda"))
        transcripts[audio_file.name] = result["text"].strip()

    return transcripts


def chunk_transcripts(transcripts: dict[str, str]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for source, text in transcripts.items():
        words = text.split()
        for i in range(0, len(words), CHUNK_WORDS - CHUNK_OVERLAP):
            chunk_words = words[i : i + CHUNK_WORDS]
            chunks.append({"source": source, "text": " ".join(chunk_words), "word_offset": i})
    return chunks


def build_index(chunks: list[dict[str, Any]], embed_model: SentenceTransformer) -> faiss.IndexFlatIP:
    texts = [c["text"] for c in chunks]
    console.print(f"Embedding {len(texts)} chunks ...")
    embeddings = embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    return index


def answer_query(
    query: str,
    chunks: list[dict[str, Any]],
    index: faiss.IndexFlatIP,
    embed_model: SentenceTransformer,
    client: anthropic.Anthropic,
) -> str:
    q_emb = embed_model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(q_emb.astype(np.float32), TOP_K)

    context_parts = []
    for idx in indices[0]:
        c = chunks[idx]
        context_parts.append(f"[{c['source']}] {c['text']}")
    context = "\n\n".join(context_parts)

    message = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Answer the question using only the provided transcript excerpts.\n\n"
                    f"Question: {query}\n\nExcerpts:\n{context}"
                ),
            }
        ],
    )
    return message.content[0].text


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG over audio transcripts")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("Set ANTHROPIC_API_KEY environment variable")

    transcripts = transcribe_directory(args.audio_dir, args.whisper_model)
    chunks = chunk_transcripts(transcripts)
    embed_model = SentenceTransformer(args.embed_model)
    index = build_index(chunks, embed_model)

    client = anthropic.Anthropic(api_key=api_key)
    answer = answer_query(args.query, chunks, index, embed_model, client)

    console.rule("Answer")
    console.print(answer)


if __name__ == "__main__":
    main()
