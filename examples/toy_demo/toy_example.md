# Toy Example (Verifier + Evaluator)

This toy run proves the evaluator JSON and verifier goals wiring is working.

## What it does
- Query asks to print 42 to stdout.
- Problem context explains JSON or plain numeric stdout.
- The evaluator executes the generated code and parses stdout.
- The verifier judges whether the metrics align with the goals.

## Run
```bash
uv run dsstar --config examples/toy_demo/toy_config.yaml
```
Note: requires a configured model API key (see README).

## Files
- examples/toy_demo/toy_evaluator.py: evaluator that scores proximity to target
- examples/toy_demo/toy_config.yaml: config with goals and evaluator settings
- examples/toy_evaluator.py: evaluator that scores proximity to target
- examples/toy_config.yaml: config with goals and evaluator settings
