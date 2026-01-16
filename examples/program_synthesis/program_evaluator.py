from __future__ import annotations

import json
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple


class ProgramSynthesisEvaluator:
    def __init__(self, expected_key: str = "is_sorted_unique", python_executable: Optional[str] = None):
        self.expected_key = expected_key
        self.python_executable = python_executable or sys.executable
        self.tests: List[Tuple[str, bool]] = [
            ("1,2,3\n", True),
            ("3,2,1\n", False),
            ("1,2,2\n", False),
            ("\n", True),
            ("-5,-1,0,2\n", True),
        ]

    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        code = context.get("code", "")
        timeout = float(context.get("execution_timeout") or 30)
        results = []
        passed = 0

        for idx, (stdin_text, expected) in enumerate(self.tests):
            stdout, stderr, returncode = self._run_code(code, stdin_text, context.get("cwd"), timeout)
            if returncode != 0:
                return self._failure_result(stdout, stderr, f"Execution failed on test {idx}")

            parsed = self._parse_json(stdout)
            if not parsed or self.expected_key not in parsed:
                return self._failure_result(stdout, stderr, f"Invalid JSON output on test {idx}")

            actual = bool(parsed[self.expected_key])
            ok = actual == expected
            results.append({"test": idx, "expected": expected, "actual": actual, "ok": ok})
            passed += 1 if ok else 0

        score = passed / len(self.tests)
        report = {"passed": passed, "total": len(self.tests), "results": results}
        return {
            "stdout": json.dumps(report, ensure_ascii=True),
            "parsed": report,
            "metrics": {"passed": passed, "total": len(self.tests)},
            "score": score,
            "should_stop": score == 1.0,
            "feedback": f"Passed {passed}/{len(self.tests)} tests.",
        }

    def _run_code(self, code: str, stdin_text: str, cwd: Optional[str], timeout: float) -> Tuple[str, str, int]:
        result = subprocess.run(
            [self.python_executable, "-c", code],
            input=stdin_text,
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
