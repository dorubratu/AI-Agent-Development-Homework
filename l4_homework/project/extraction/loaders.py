from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

LOADER_REGISTRY: dict[str, Callable[[Path], str]] = {}


class LoaderError(Exception):
    pass


def register_loader(*extensions: str) -> Callable[[Callable[[Path], str]], Callable[[Path], str]]:
    def decorator(func: Callable[[Path], str]) -> Callable[[Path], str]:
        for ext in extensions:
            LOADER_REGISTRY[ext.lower()] = func
        return func
    return decorator


@register_loader(".pdf")
def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p for p in pages if p.strip())


@register_loader(".docx")
def _load_docx(path: Path) -> str:
    import docx2txt

    text = docx2txt.process(str(path))
    return text or ""


@register_loader(".txt", ".md")
def _load_txt(path: Path) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise LoaderError(f"Cannot decode {path} with utf-8 or latin-1")


def load_document(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise LoaderError(f"File not found: {path}")

    ext = path.suffix.lower()
    loader = LOADER_REGISTRY.get(ext)
    if loader is None:
        raise LoaderError(
            f"Unsupported extension: {ext}. Available: {sorted(LOADER_REGISTRY.keys())}"
        )

    try:
        text = loader(path)
    except Exception as exc:
        raise LoaderError(f"Failed to load {path}: {exc}") from exc

    text = text.strip()
    if not text:
        raise LoaderError(f"Empty text extracted from {path}")
    return text
