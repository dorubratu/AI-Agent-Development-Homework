from .registry import TOOL_REGISTRY


class ToolWrapper:
    @staticmethod
    def call(name: str, args: dict) -> str:
        if name not in TOOL_REGISTRY:
            return f"Eroare: tool '{name}' nu există."

        tool = TOOL_REGISTRY[name]

        try:
            params = tool["params_model"](**args)
        except Exception as exc:
            return f"Eroare validare pentru '{name}': {exc}"

        try:
            return str(tool["func"](params))
        except Exception as exc:
            return f"Eroare execuție '{name}': {exc}"

    @staticmethod
    def catalog() -> list[dict]:
        return [
            {
                "name": name,
                "description": tool["description"],
                "input_schema": tool["params_model"].model_json_schema(),
            }
            for name, tool in TOOL_REGISTRY.items()
        ]
