"""End-to-end demo: ingest sample_docs and ask the agent a few questions.

Pre-req:
    docker compose up -d
    alembic upgrade head
    cp .env.example .env  # set ANTHROPIC_API_KEY
"""
from __future__ import annotations

import logging
from pathlib import Path

from agent import DocumentAnalystAgent
from pipeline import Pipeline

SAMPLE_DIR = Path(__file__).parent / "sample_docs"

DEMO_QUESTIONS = [
    "Ce clauze de reziliere avem în contracte și care e preavizul?",
    "Care este totalul facturii FAC-2026-042 și ce TVA s-a aplicat?",
    "Ce penalități sunt menționate pentru încălcarea NDA-ului?",
    "Există o clauză de confidențialitate? Cât durează?",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Silence per-request HTTP logs from the Anthropic SDK's httpx client.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    files = sorted(p for p in SAMPLE_DIR.iterdir() if p.is_file())
    if not files:
        print(f"No sample files in {SAMPLE_DIR}")
        return

    print(f"=== Ingesting {len(files)} files from {SAMPLE_DIR} ===")
    pipeline = Pipeline()
    results = pipeline.process_batch(files)
    for r in results:
        if r.success:
            print(f"  OK   {r.filename} -> id={r.document_id} chunks={r.chunk_count}")
        else:
            print(f"  FAIL {r.filename}: {r.error}")

    print("\n=== Asking the agent ===")
    agent = DocumentAnalystAgent()
    for q in DEMO_QUESTIONS:
        print(f"\n>>> {q}")
        answer = agent.run(q, verbose=False)
        print(f"<<< {answer}")


if __name__ == "__main__":
    main()
