from __future__ import annotations

import importlib
import importlib.util
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Protocol


class Evaluator(Protocol):
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Return a JSON-serializable dict or JSON string with evaluation details."""
        ...


class NoopEvaluator:
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"score": None, "metrics": {}, "feedback": ""}


def load_evaluator(path: Optional[str], kwargs: Optional[Dict[str, Any]] = None) -> Evaluator:
    if not path:
        return NoopEvaluator()

    module_path, _, attr = path.partition(":")
    attr = attr or "Evaluator"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        file_path = Path(module_path)
        if file_path.suffix != ".py":
            file_path = file_path.with_suffix(".py")
        if not file_path.is_absolute():
            file_path = Path.cwd() / file_path
        if not file_path.exists():
            raise
        module_name = f"_ds_star_eval_{file_path.stem}_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            raise
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    target = getattr(module, attr)

    if callable(target):
        return target(**(kwargs or {}))

    return target
