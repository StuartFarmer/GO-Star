from __future__ import annotations

import re
from typing import Any, Dict, Optional


class ToyEvaluator:
    def __init__(self, target: float = 42.0, tolerance: float = 0.5):
        self.target = float(target)
        self.tolerance = float(tolerance)

    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raw_result = context.get("result", "")
        value = self._extract_number(raw_result)
        if value is None:
            return {
                "metrics": {"target": self.target, "value": None, "abs_error": None},
                "score": None,
                "should_stop": False,
                "feedback": "No numeric value found in execution result.",
            }

        abs_error = abs(value - self.target)
        score = -abs_error
        return {
            "metrics": {"target": self.target, "value": value, "abs_error": abs_error},
            "score": score,
            "should_stop": abs_error <= self.tolerance,
            "feedback": f"Abs error {abs_error:.4f} vs target {self.target:.4f}.",
        }

    def _extract_number(self, text: Any) -> Optional[float]:
        match = re.search(r"-?\d+(?:\.\d+)?", str(text))
        if not match:
            return None
        return float(match.group(0))
