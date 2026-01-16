# Toy Example (Verifier + Evaluator)

This toy run proves the evaluator JSON and verifier goals wiring is working.

## What it does
- Query asks for a number close to 42.
- The evaluator extracts the first number printed by the code.
- The verifier judges whether the metrics align with the goals.

## Run
```bash
uv run dsstar --config examples/toy_config.yaml
```
Note: requires a configured model API key (see README).

## Files
- examples/data/hello.txt: placeholder data file for the analyzer
- examples/toy_evaluator.py: evaluator that scores proximity to target
- examples/toy_config.yaml: config with goals and evaluator settings
