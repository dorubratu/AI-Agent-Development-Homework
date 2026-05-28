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


class WebSearchParams(BaseModel):
    query: str = Field(
        description="Termenii de căutat pe web.",
        min_length=2,
    )
    max_results: int = Field(
        default=3,
        description="Numărul maxim de rezultate returnate.",
        ge=1,
        le=10,
    )
