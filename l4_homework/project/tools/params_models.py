from pydantic import BaseModel, Field


class CalculatorParams(BaseModel):
    expression: str = Field(
        description="Expresia matematică de evaluat (ex: '2 + 3 * 4').",
        min_length=1,
    )


class GetDatetimeParams(BaseModel):
    timezone: str = Field(
        default="UTC",
        description="Numele zonei orare (ex: 'UTC', 'Europe/Bucharest').",
    )


class SearchDocumentsParams(BaseModel):
    query: str = Field(
        description=(
            "Întrebarea sau termenii cheie căutați în documentele ingerate. "
            "Cosine similarity peste embeddings pe chunks."
        ),
        min_length=2,
    )
    top_k: int = Field(
        default=3,
        description="Numărul maxim de chunks returnate (1-10).",
        ge=1,
        le=10,
    )
    min_score: float = Field(
        default=0.35,
        description="Prag minim de similaritate (0-1). Sub prag → chunk ignorat.",
        ge=0.0,
        le=1.0,
    )
