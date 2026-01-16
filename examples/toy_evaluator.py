from __future__ import annotations

import json
import re
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple


class ToyEvaluator:
    def __init__(self, target: float = 42.0, tolerance: float = 0.5, python_executable: Optional[str] = None):
        self.target = float(target)
        self.tolerance = float(tolerance)
        self.python_executable = python_executable or sys.executable

    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        stdout, stderr, returncode = self._run_code(context.get("code", ""), context.get("cwd"))
        if returncode != 0:
            return {
                "stdout": stdout,
                "stderr": stderr,
                "error": "Execution failed",
                "score": None,
                "metrics": {},
                "should_stop": False,
                "feedback": "Code execution failed.",
            }

        parsed = self._parse_json(stdout)
        if parsed is None or "value" not in parsed:
            value = self._extract_number(stdout)
            if value is None:
                return {
                    "stdout": stdout,
                    "parsed": parsed,
                    "metrics": {"target": self.target, "value": None, "abs_error": None},
                    "score": None,
                    "should_stop": False,
                    "feedback": "Invalid output (expected JSON with key 'value' or a numeric stdout).",
                }
        else:
            value = float(parsed["value"])

        abs_error = abs(value - self.target)
        score = -abs_error
        return {
            "stdout": stdout,
            "parsed": parsed,
            "metrics": {"target": self.target, "value": value, "abs_error": abs_error},
            "score": score,
            "should_stop": abs_error <= self.tolerance,
            "feedback": f"Abs error {abs_error:.4f} vs target {self.target:.4f}.",
        }

    def _run_code(self, code: str, cwd: Optional[str]) -> Tuple[str, str, int]:
        with subprocess.Popen(
            [self.python_executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd or None,
        ) as proc:
            stdout, stderr = proc.communicate()
            return stdout, stderr, proc.returncode

    def _parse_json(self, text: Any) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(str(text))
        except Exception:
            return None

    def _extract_number(self, text: Any) -> Optional[float]:
        match = re.search(r"-?\d+(?:\.\d+)?", str(text))
        if not match:
            return None
        return float(match.group(0))
