"""
guardrails.py — Protecția serverului MCP (Cerința 3 din temă).

Două straturi, aplicate în ordine pe ORICE input care intră în server:

  1. Input validation (ieftin, determinist):
       - tip (trebuie să fie string)
       - dimensiune (min/max lungime)
       - câmpuri permise (allow-list de chei în argumente)

  2. Prompt-injection detection:
       - REGEX (rapid, fără cost) — prinde tiparele clasice de jailbreak /
         injection / exfiltrare / cod periculos
       - LLM-as-Judge (opțional) — pentru cazuri ambigue pe care regex-ul le ratează

Design: un singur apel `Guardrail.check(field, value, extra_args)` returnează un
`GuardrailResult`. Handlerele tool-urilor îl apelează ÎNAINTE de a invoca agentul;
dacă input-ul e blocat, ridică `GuardrailError` (care în MCP devine eroare JSON-RPC).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


# ── Configurare validare ──────────────────────────────────────────────────
MIN_LENGTH = 3
MAX_LENGTH = 2000  # un query de business nu are nevoie de mai mult


# ── Tipare prompt-injection (regex) ────────────────────────────────────────
# Fiecare tuplă: (etichetă, pattern compilat). case-insensitive, multilingv (ro/en).
_INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ignore_instructions", re.compile(
        r"\b(ignore|disregard|forget|uită[\s-]*de|ignoră)\b.{0,30}"
        r"\b(previous|above|prior|earlier|all|toate|anterioare|de\s*mai\s*sus|instruc[țt]i|prompt|rules?|reguli)\b",
        re.IGNORECASE | re.DOTALL)),
    ("override_role", re.compile(
        r"\b(you\s+are\s+now|act\s+as|pretend\s+to\s+be|from\s+now\s+on|"
        r"e[șs]ti\s+acum|comport[ăa]-te\s+ca|de\s+acum\s+(înainte|inainte))\b",
        re.IGNORECASE)),
    ("system_prompt_leak", re.compile(
        r"\b(system\s*prompt|your\s+instructions|initial\s+prompt|"
        r"reveal|print|show|repeat|spune-mi)\b.{0,40}"
        r"\b(prompt|instruc[țt]i|system|configura|rules?|reguli)\b",
        re.IGNORECASE | re.DOTALL)),
    ("secret_exfiltration", re.compile(
        r"\b(api[\s_-]*key|secret|password|paro[lț][ăa]|token|credential|"
        r"connection\s*string|env(ironment)?\s*var|\.env|ANTHROPIC_API_KEY|"
        r"OPENAI_API_KEY|DATABASE_URL)\b",
        re.IGNORECASE)),
    ("jailbreak_markers", re.compile(
        r"\b(jailbreak|DAN\s*mode|developer\s*mode|do\s+anything\s+now|"
        r"unfiltered|bypass.{0,20}(guardrail|filter|safety|restric))\b",
        re.IGNORECASE)),
    ("code_injection", re.compile(
        r"(os\.system|subprocess\.|__import__|\beval\s*\(|\bexec\s*\(|"
        r"`[^`]*`|\$\([^)]*\)|;\s*(rm|del|drop)\s)",
        re.IGNORECASE)),
    ("sql_destructive", re.compile(
        r"\b(drop\s+table|delete\s+from|truncate\s+table|update\s+\w+\s+set|"
        r"insert\s+into|alter\s+table|;\s*--)\b",
        re.IGNORECASE)),
]


@dataclass
class GuardrailResult:
    """Rezultatul verificării. allowed=False => input blocat."""
    allowed: bool
    stage: str            # "validation" | "regex" | "llm_judge" | "ok"
    reason: str = ""
    matched: list[str] = field(default_factory=list)


class GuardrailError(Exception):
    """Ridicată când un input este blocat de guardrail."""

    def __init__(self, result: GuardrailResult):
        self.result = result
        super().__init__(f"[{result.stage}] {result.reason}")


class Guardrail:
    """
    Guardrail-ul serverului MCP.

    Args:
        allowed_fields: chei permise în dict-ul de argumente al unui tool.
        llm: provider LLM opțional (skillab LLMProvider) pentru LLM-as-Judge.
             Dacă e None, se folosește doar regex (suficient pentru temă).
        use_llm_judge: activează stratul LLM-as-Judge pentru inputuri ambigue.
    """

    def __init__(
        self,
        allowed_fields: set[str] | None = None,
        llm=None,
        use_llm_judge: bool = False,
    ):
        self.allowed_fields = allowed_fields or {"question", "query"}
        self.llm = llm
        self.use_llm_judge = use_llm_judge and llm is not None
        self._judge_prompt = self._load_judge_prompt()

    # ── 1. INPUT VALIDATION ────────────────────────────────────────────────
    def validate_arguments(self, arguments: dict) -> GuardrailResult:
        """Validează câmpurile permise în dict-ul de argumente."""
        if not isinstance(arguments, dict):
            return GuardrailResult(False, "validation", "arguments must be an object")

        extra = set(arguments) - self.allowed_fields
        if extra:
            return GuardrailResult(
                False, "validation",
                f"Câmpuri nepermise: {sorted(extra)}. Permise: {sorted(self.allowed_fields)}",
                matched=sorted(extra),
            )
        return GuardrailResult(True, "ok")

    def validate_value(self, value) -> GuardrailResult:
        """Validează tipul și dimensiunea unei valori text de input."""
        if not isinstance(value, str):
            return GuardrailResult(False, "validation", f"Input trebuie să fie string, nu {type(value).__name__}")

        stripped = value.strip()
        if len(stripped) < MIN_LENGTH:
            return GuardrailResult(False, "validation", f"Input prea scurt (min {MIN_LENGTH} caractere)")
        if len(value) > MAX_LENGTH:
            return GuardrailResult(False, "validation", f"Input prea lung (max {MAX_LENGTH} caractere, primit {len(value)})")
        return GuardrailResult(True, "ok")

    # ── 2a. PROMPT INJECTION — REGEX ────────────────────────────────────────
    def scan_regex(self, text: str) -> GuardrailResult:
        """Caută tipare cunoscute de prompt injection / cod periculos."""
        hits = [label for label, pat in _INJECTION_PATTERNS if pat.search(text)]
        if hits:
            return GuardrailResult(
                False, "regex",
                f"Posibil prompt injection / input periculos: {hits}",
                matched=hits,
            )
        return GuardrailResult(True, "ok")

    # ── 2b. PROMPT INJECTION — LLM-AS-JUDGE ─────────────────────────────────
    def scan_llm(self, text: str) -> GuardrailResult:
        """
        Strat opțional: cere unui LLM să clasifice input-ul SAFE/UNSAFE.
        Fail-open la eroare de model (nu blocăm business pe un timeout), dar
        logăm. Pentru un setup mai strict, schimbă în fail-closed.
        """
        if not self.use_llm_judge:
            return GuardrailResult(True, "ok")

        import json
        prompt = self._judge_prompt.replace("{{ text }}", text)
        try:
            response = self.llm.generate_sync([{"role": "user", "content": prompt}])
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM-as-Judge indisponibil (%s) — continui doar cu regex.", e)
            return GuardrailResult(True, "ok")

        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            logger.warning("LLM-as-Judge a returnat un răspuns neparsabil — permit.")
            return GuardrailResult(True, "ok")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return GuardrailResult(True, "ok")

        if str(data.get("verdict", "")).upper() == "UNSAFE":
            return GuardrailResult(
                False, "llm_judge",
                f"LLM-as-Judge: {data.get('reason', 'input nesigur')}",
                matched=["llm_judge"],
            )
        return GuardrailResult(True, "ok")

    # ── ORCHESTRARE ─────────────────────────────────────────────────────────
    def check(self, field_name: str, value, arguments: dict | None = None) -> GuardrailResult:
        """
        Rulează întregul lanț pe un input. Returnează primul rezultat care blochează,
        altfel un rezultat "ok". Ordinea: validare câmpuri → tip/dimensiune → regex → LLM.
        """
        if arguments is not None:
            r = self.validate_arguments(arguments)
            if not r.allowed:
                return r

        r = self.validate_value(value)
        if not r.allowed:
            return r

        r = self.scan_regex(value)
        if not r.allowed:
            return r

        r = self.scan_llm(value)
        if not r.allowed:
            return r

        return GuardrailResult(True, "ok", reason="passed all guardrails")

    def enforce(self, field_name: str, value, arguments: dict | None = None) -> str:
        """
        Ca `check`, dar ridică `GuardrailError` dacă input-ul e blocat.
        Returnează valoarea (curățată de spații) dacă trece.
        """
        result = self.check(field_name, value, arguments)
        if not result.allowed:
            logger.warning("GUARDRAIL BLOCK [%s] field=%s reason=%s", result.stage, field_name, result.reason)
            raise GuardrailError(result)
        logger.info("GUARDRAIL PASS field=%s", field_name)
        return value.strip()

    # ── intern ──────────────────────────────────────────────────────────────
    def _load_judge_prompt(self) -> str:
        """Încarcă template-ul prompt pentru LLM-as-Judge din YAML."""
        path = PROMPTS_DIR / "guardrail_judge.yaml"
        if not path.exists():
            return "Clasifică SAFE/UNSAFE:\n{{ text }}"
        try:
            import yaml
            data = yaml.safe_load(path.read_text())
            return data.get("template", "")
        except Exception:  # noqa: BLE001 — yaml opțional
            text = path.read_text()
            marker = "template: |"
            if marker in text:
                return text.split(marker, 1)[1]
            return text
