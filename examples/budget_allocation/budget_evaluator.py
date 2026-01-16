from __future__ import annotations

import json
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple


class BudgetAllocationEvaluator:
    def __init__(self, python_executable: Optional[str] = None):
        self.python_executable = python_executable or sys.executable
        self.weights = {"A": 3.0, "B": 2.0, "C": 1.0, "D": 4.0}
        self.limits = {"A": 0.40, "B": 0.50, "C": 0.30, "D": 0.60}

    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        code = context.get("code", "")
        timeout = float(context.get("execution_timeout") or 30)
        stdout, stderr, returncode = self._run_code(code, context.get("cwd"), timeout)
        if returncode != 0:
            return self._failure_result(stdout, stderr, "Execution failed")

        parsed = self._parse_json(stdout)
        if not parsed or "allocations" not in parsed:
            return self._failure_result(stdout, stderr, "Invalid JSON output (missing 'allocations')")

        allocations = parsed.get("allocations", {})
        if not isinstance(allocations, dict):
            return self._failure_result(stdout, stderr, "Allocations must be an object")

        metrics, penalty = self._score_allocations(allocations)
        score = metrics["impact"] - penalty
        should_stop = penalty == 0.0 and metrics["impact"] >= 3.2
        return {
            "stdout": stdout,
            "parsed": {"allocations": allocations, "metrics": metrics},
            "metrics": metrics,
            "score": score,
            "should_stop": should_stop,
            "feedback": f"Impact {metrics['impact']:.4f}, penalty {penalty:.4f}.",
        }

    def _score_allocations(self, allocations: Dict[str, Any]) -> Tuple[Dict[str, float], float]:
        penalty = 0.0
        total = 0.0
        impact = 0.0
        for key, weight in self.weights.items():
            value = float(allocations.get(key, 0.0))
            if value < 0.0:
                penalty += abs(value)
            limit = self.limits[key]
            if value > limit:
                penalty += value - limit
            total += value
            impact += weight * value

        if abs(total - 1.0) > 1e-6:
            penalty += abs(total - 1.0) * 10.0

        metrics = {
            "impact": impact,
            "sum": total,
            "penalty": penalty,
        }
        return metrics, penalty

    def _run_code(self, code: str, cwd: Optional[str], timeout: float) -> Tuple[str, str, int]:
        result = subprocess.run(
            [self.python_executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or None,
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode

    def _parse_json(self, text: Any) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(str(text))
        except Exception:
            return None

    def _failure_result(self, stdout: str, stderr: str, message: str) -> Dict[str, Any]:
        return {
            "stdout": stdout,
            "stderr": stderr,
            "error": message,
            "metrics": {},
            "score": 0.0,
            "should_stop": False,
            "feedback": message,
        }
