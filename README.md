# GO-STAR: A General Optimization Agentic Framework

GO-STAR is built off of the DS-STAR self-improving agent that can generally solve and optimize problems rather than the original DS-STAR constraint of answering data science queries.

You can read about the original DS-STAR agent here: [DS-STAR: A State-of-the-Art Versatile Data Science Agent](https://research.google/blog/ds-star-a-state-of-the-art-versatile-data-science-agent/). [Paper](https://arxiv.org/pdf/2509.21825)

## Features

- **Agentic Workflow**: Implements a pipeline of specialized AI agents (Planner, Coder, Verifier, Router, Debugger, Finalyzer) that collaborate to solve and optimize problems.
- **Reproducibility**: Every step of the pipeline is saved, including prompts, generated code, execution results, and metadata. This allows for complete auditability and reproducibility of results.
- **Interactive & Resume-able**: Runs can be paused and resumed. The interactive mode allows for step-by-step execution.
- **Code Editing & Debugging**: Allows users to manually edit the generated code during a run and features an auto-debug agent to fix execution errors.
- **Configuration-driven**: Project settings, model parameters, evaluator, and run configurations are managed through a `config.yaml` file.

## How it Works

The GO-STAR pipeline is composed of several phases and agents:

1.  **Iterative Planning & Execution**:
    *   The `Planner` creates an initial plan to address the task.
    *   The `Coder` generates Python code to execute the current step of the plan.
    *   The code is executed, and the result is captured (either by the evaluator or the default runtime).
    *   An automatic `Debugger` agent attempts to fix any code that fails.
    *   The `Verifier` checks if the result sufficiently aligns with the goals.
    *   The `Router` decides what to do next: either finalize the plan or add a new step for refinement.
    *   This loop continues until the plan is deemed sufficient or the maximum number of refinement rounds is reached.
2.  **Finalization**: The `Finalyzer` agent takes the final code and results and formats them into a clean, specified output format (e.g., JSON).

All artifacts for each run are stored in the `runs/` directory, organized by `run_id`.

## Evaluator Pattern

GO-STAR uses an evaluator to score candidate solutions and decide when to stop refining.
The evaluator can optionally execute the generated code itself and return structured
JSON metrics, which the Verifier uses to judge progress against your goals.

Key config options:
- `query`: the task description (required in config YAML).
- `problem_context`: extra constraints and output format requirements.
- `evaluator`: import path or file path to the evaluator class.
- `evaluator_kwargs`: constructor args for the evaluator.
- `evaluator_runs_code`: if true, evaluator executes code and provides stdout/metrics.
- `evaluator_python`: optional Python executable for evaluator-run code.

Example (program synthesis):
```yaml
query: "Write code that checks whether a list of integers is strictly increasing."
problem_context: |
  Read a single line from stdin. Output strict JSON:
  {"is_sorted_unique": true|false}
evaluator_runs_code: true
evaluator: "examples/program_synthesis/program_evaluator.py:ProgramSynthesisEvaluator"
```

## Project Structure

```
/
├─── src/ds_star/            # Package source (agent logic, CLI, prompts)
├─── config.yaml             # Main configuration file
├─── examples/               # Runnable toy examples and evaluators
├─── tests/                  # Test suite
├─── pyproject.toml          # Project metadata and dependencies (uv format)
├─── uv.lock                 # Locked dependency versions for reproducibility
├─── .python-version         # Python version specification for uv
└─── runs/                   # Directory where all experiment runs and artifacts are stored
```

## Getting Started

### Prerequisites

- Python 3.11+
- An API key for Google's Gemini models.
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

### Installation

#### Using uv (Recommended)

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd DS-Star
    ```

2.  **Install uv (if not already installed):**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3.  **Install dependencies with uv:**
    ```bash
    uv sync
    ```

### Configuration

1.  **Set your API Key:**
    The application requires a Gemini API key. You can set it as an environment variable:
    ```bash
    export GEMINI_API_KEY='your-api-key'
    ```
    Alternatively, you can add it to the `config.yaml` file.

2.  **Customize `config.yaml`:**
    Create a `config.yaml` file in the root of the project and customize the settings. See the "Configuration" section below for details.

    ```yaml
    # config.yaml
    model_name: 'gemini-1.5-flash'
    max_refinement_rounds: 5
    interactive: false
    # api_key: 'your-api-key' # Alternatively, place it here
    
    # Optional: Configure specific models for different agents
    agent_models:
      PLANNER: 'gpt-4'
      CODER: 'gemini-1.5-pro'
      VERIFIER: 'gemini-1.5-flash'
    ```

## Usage

### Starting a New Run

Define the task in your config YAML and run with `--config`.

Using uv:
```bash
uv run dsstar --config examples/program_synthesis/program_config.yaml
```

### Resuming a Run

If a run was interrupted, you can resume it using its `run_id`.

```bash
uv run dsstar --resume <run_id>
```

### Editing Code During a Run

You can manually edit the last generated piece of code and re-run it. This is useful for manual debugging or tweaking the agent's logic.

```bash
uv run dsstar --edit-last --resume <run_id>
```
This will open the last code file in your default text editor (`nano`, `vim`, etc.). After you save and close the editor, the script will re-execute the modified code.

### Interactive Mode

To review each step before proceeding, use the interactive flag.

```bash
uv run dsstar --interactive --query "..."
```

## UV Package Manager

This project uses `uv` for fast and reliable dependency management. Here are some useful commands:

### Common UV Commands

- **Install dependencies**: `uv sync`
- **Add a new dependency**: `uv add package-name`
- **Remove a dependency**: `uv remove package-name`
- **Update dependencies**: `uv sync --upgrade`
- **Run a command in the virtual environment**: `uv run python script.py`
- **Show installed packages**: `uv pip list`

### Benefits of UV

- **Speed**: uv is 10-100x faster than pip
- **Reliability**: Consistent dependency resolution with lock files
- **No virtual environment activation needed**: Use `uv run` to execute commands directly
- **Better dependency resolution**: Automatically resolves complex dependency conflicts

## Configuration

The following options are available in `config.yaml` and can be overridden by CLI arguments:

- `run_id` (string): The ID of a run to resume.
- `max_refinement_rounds` (int): The maximum number of times the agent will try to refine its plan.
- `api_key` (string): Your Gemini API key.
- `model_name` (string): The Gemini model to use (e.g., `gemini-1.5-flash`).
- `interactive` (bool): If true, waits for user input before executing each step.
- `auto_debug` (bool): If true, the `Debugger` agent will automatically try to fix failing code.
- `execution_timeout` (int): Timeout in seconds for code execution.
- `preserve_artifacts` (bool): If true, all step artifacts are saved to the `runs` directory.
- `agent_models` (dict): A dictionary mapping agent names (e.g., `PLANNER`, `CODER`) to specific model names. If not specified, `model_name` is used.
- `problem_context` (string): Optional per-run context passed to every agent.
- `verifier_goals` (string): Optional goals used by the verifier.
- `evaluator` (string): Evaluator import path or file path (e.g., `examples/toy_demo/toy_evaluator.py:ToyEvaluator`).
- `evaluator_kwargs` (dict): Keyword args passed to the evaluator constructor.
- `evaluator_runs_code` (bool): If true, the evaluator executes generated code and returns stdout + metrics.
- `evaluator_python` (string): Optional Python executable for evaluator-run code.

## Providers

DS-STAR supports multiple AI model providers. Each provider requires specific environment variables to be configured:

### Google Gemini

**Provider Identifier**: Default provider (no prefix required)

**Environment Variable**:
```bash
export GEMINI_API_KEY='your-gemini-api-key'
```

**Model Examples**:`gemini-2.5-pro`, `gemini-2.0-flash`

### OpenAI

**Provider Identifier**: Models prefixed with `gpt` or `o1`

**Environment Variable**:
```bash
export OPENAI_API_KEY='your-openai-api-key'
```

**Model Examples**: `gpt-4`, `gpt-4-turbo`, `o1`

### Ollama

**Provider Identifier**: Models prefixed with `ollama/`

**Environment Variables**:
```bash
export OLLAMA_API_KEY='your-ollama-api-key'  # Optional
export OLLAMA_HOST='http://localhost:11434'  # Optional, defaults to http://localhost:11434
```

**Model Examples**: `ollama/llama3`, `ollama/qwen3-coder`
## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.

```
