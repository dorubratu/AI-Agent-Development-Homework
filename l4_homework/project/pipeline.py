"""Document ingestion pipeline: load -> chunk -> extract -> embed -> store.

Entry point:
    process(file: Path) -> ExtractionResult

Idempotent on filename: if a document with the same name already exists in
the DB, we skip it (and surface the existing id in the result).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from database import transaction
from extraction import (
    DocumentExtraction,
    ExtractionResult,
    LoaderError,
    StructuredExtractor,
    load_document,
    split_text,
)
from rag import Embedder
from repositories import ChunkRepository, DocumentRepository

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        extractor: StructuredExtractor | None = None,
        embedder: Embedder | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> None:
        self.extractor = extractor or StructuredExtractor()
        self.embedder = embedder or Embedder()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process(self, file: Path) -> ExtractionResult:
        file = Path(file)

        # 1. Load
        try:
            content = load_document(file)
        except LoaderError as exc:
            return ExtractionResult(success=False, filename=file.name, error=str(exc))

        # 2. Skip if filename already in DB (idempotent ingestion)
        with transaction() as db:
            existing = DocumentRepository(db).get_by_filename(file.name)
            if existing is not None:
                logger.info("skip duplicate: %s (id=%d)", file.name, existing.id)
                return ExtractionResult(
                    success=True,
                    filename=file.name,
                    document_id=existing.id,
                    chunk_count=len(existing.chunks),
                )

        # 3. Chunk
        chunks = split_text(content, self.chunk_size, self.chunk_overlap)
        if not chunks:
            return ExtractionResult(
                success=False, filename=file.name, error="no chunks produced"
            )

        # 4. Extract structured metadata via LLM (forced tool use)
        try:
            extraction = self.extractor.extract(filename=file.name, content=content)
        except Exception as exc:
            logger.exception("extraction failed for %s", file.name)
            return ExtractionResult(
                success=False, filename=file.name, error=f"extract: {exc}"
            )

        # 5. Embed all chunks in one batch (sentence-transformers is fast)
        embeddings = self.embedder.encode(chunks)

        # 6. Persist atomically: document + all chunks share one transaction
        with transaction() as db:
            doc_repo = DocumentRepository(db)
            chunk_repo = ChunkRepository(db)

            doc = doc_repo.create(
                filename=extraction.filename,
                content=extraction.content,
                doc_type=extraction.doc_type,
                doc_metadata=extraction.to_db_metadata(),
            )
            chunk_repo.create_chunks_batch(
                document_id=doc.id,
                contents=chunks,
                embeddings=embeddings,
            )
            doc_id = doc.id

        logger.info(
            "saved %s id=%d type=%s chunks=%d",
            file.name,
            doc_id,
            extraction.doc_type,
            len(chunks),
        )

        return ExtractionResult(
            success=True,
            filename=file.name,
            document_id=doc_id,
            extraction=extraction,
            chunk_count=len(chunks),
        )

    def process_batch(self, files: list[Path]) -> list[ExtractionResult]:
        results: list[ExtractionResult] = []
        for path in files:
            try:
                results.append(self.process(path))
            except Exception as exc:
                logger.exception("unexpected failure on %s", path)
                results.append(
                    ExtractionResult(success=False, filename=path.name, error=str(exc))
                )
        return results


def _dump_json(extraction: DocumentExtraction, target: Path) -> None:
    payload = extraction.model_dump(exclude={"content"})
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG store.")
    parser.add_argument("paths", nargs="+", help="Files or directories to ingest.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Where to write per-document JSON dumps.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Silence per-request HTTP logs from the Anthropic SDK's httpx client.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    files: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(p for p in sorted(path.iterdir()) if p.is_file())
        elif path.is_file():
            files.append(path)
        else:
            logger.warning("not found: %s", path)

    if not files:
        logger.warning("no files to process")
        return

    pipeline = Pipeline()
    results = pipeline.process_batch(files)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    for r in results:
        if r.success:
            ok += 1
            if r.extraction is not None:
                _dump_json(r.extraction, args.output_dir / f"{Path(r.filename).stem}.json")
            print(
                f"OK   {r.filename} -> id={r.document_id} chunks={r.chunk_count}"
            )
        else:
            fail += 1
            print(f"FAIL {r.filename}: {r.error}")

    print(f"\nDone: {ok} ok, {fail} failed")


if __name__ == "__main__":
    main()
