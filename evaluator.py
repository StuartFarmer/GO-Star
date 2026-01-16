from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, Protocol


class Evaluator(Protocol):
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Return a dict that may include should_stop, score, feedback."""
        ...


class NoopEvaluator:
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"should_stop": False, "score": None, "feedback": ""}


def load_evaluator(path: Optional[str], kwargs: Optional[Dict[str, Any]] = None) -> Evaluator:
    if not path:
        return NoopEvaluator()

    module_path, _, attr = path.partition(":")
    attr = attr or "Evaluator"
    module = importlib.import_module(module_path)
    target = getattr(module, attr)

    if callable(target):
        return target(**(kwargs or {}))

    return target
