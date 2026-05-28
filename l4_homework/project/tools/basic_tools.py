import ast
import operator as op
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .params_models import CalculatorParams, GetDatetimeParams
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
