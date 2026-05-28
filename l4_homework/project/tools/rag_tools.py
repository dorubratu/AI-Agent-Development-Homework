from __future__ import annotations

from database import transaction
from rag import RAGService

from .params_models import SearchDocumentsParams
from .registry import register_tool


@register_tool
def search_documents(params: SearchDocumentsParams) -> str:
    """Caută în documentele ingerate cu RAG (cosine similarity peste pgvector).
    Returnează chunks relevante, fiecare cu filename, chunk_index, score și
    text. Folosește acest tool când întrebarea vizează clauze, sume, date sau
    detalii din documentele utilizatorului — nu inventa răspunsuri."""
    with transaction() as db:
        rag = RAGService(db)
        hits = rag.search_with_threshold(
            params.query, top_k=params.top_k, min_score=params.min_score
        )
        if not hits:
            return (
                "Niciun chunk peste pragul de similaritate. "
                "Spune utilizatorului explicit că documentele nu conțin răspunsul."
            )
        return rag.render_context(hits)
