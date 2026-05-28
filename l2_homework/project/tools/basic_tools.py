import ast
import operator as op
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from .params_models import CalculatorParams, GetDatetimeParams, WebSearchParams
from .registry import register_tool

_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.FloorDiv: op.floordiv,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Construct nepermis în expresie: {ast.dump(node)}")


@register_tool
def calculator(params: CalculatorParams) -> str:
    """Evaluează o expresie matematică simplă (+, -, *, /, %, **, //) și
    returnează rezultatul ca string. Nu acceptă apeluri de funcții, nume sau
    importuri — doar aritmetică pură pe numere."""
    tree = ast.parse(params.expression, mode="eval")
    result = _safe_eval(tree.body)
    return str(result)


@register_tool
def get_datetime(params: GetDatetimeParams) -> str:
    """Returnează data și ora curentă în formatul ISO 8601 pentru zona orară
    specificată. Folosește nume IANA precum 'UTC' sau 'Europe/Bucharest'."""
    try:
        tz = ZoneInfo(params.timezone)
    except ZoneInfoNotFoundError:
        return f"Eroare: zona orară '{params.timezone}' nu este recunoscută."
    return datetime.now(tz).isoformat()


@register_tool
def web_search(params: WebSearchParams) -> str:
    """Caută pe web folosind API-ul public DuckDuckGo Instant Answer și
    returnează un rezumat textual al primelor rezultate. Util pentru întrebări
    factuale despre subiecte generale, definiții sau evenimente."""
    try:
        response = httpx.get(
            "https://api.duckduckgo.com/",
            params={"q": params.query, "format": "json", "no_html": "1"},
            timeout=10.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return f"Eroare la apelul web_search: {exc}"

    data = response.json()
    bucati: list[str] = []

    if abstract := data.get("AbstractText"):
        bucati.append(f"Rezumat: {abstract}")
        if source := data.get("AbstractURL"):
            bucati.append(f"Sursă: {source}")

    related = data.get("RelatedTopics", [])[: params.max_results]
    for item in related:
        text = item.get("Text") or (item.get("Topics", [{}])[0].get("Text") if item.get("Topics") else None)
        if text:
            bucati.append(f"- {text}")

    if not bucati:
        return f"Niciun rezultat util pentru '{params.query}'."
    return "\n".join(bucati)
