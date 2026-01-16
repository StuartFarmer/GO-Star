import os
import json
import subprocess
import re
import uuid
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import sys
import atexit
import yaml
from dataclasses import dataclass, asdict, field
from datetime import datetime
from importlib import resources

from .provider import ModelProvider, GeminiProvider, OllamaProvider, OpenAIProvider
from .evaluator import load_evaluator

# =============================================================================
# CONFIGURATION & PROMPT TEMPLATES
# =============================================================================

with resources.files("ds_star").joinpath("prompt.yaml").open("r", encoding="utf-8") as f:
    PROMPT_TEMPLATES = yaml.safe_load(f)

@dataclass
class DSConfig:
    """Centralized configuration for the entire pipeline."""
    run_id: str = None
    max_refinement_rounds: int = 5
    api_key: Optional[str] = None
    model_name: str = None
    interactive: bool = False
    auto_debug: bool = True
    # debug attempts defaults to inf for backwards compatibility
    debug_attempts: float = float('inf')
    execution_timeout: int = 60
    preserve_artifacts: bool = True
    runs_dir: str = "runs"
    data_dir: str = "data"
    code_library_dir: str = "code_library"
    agent_models: Dict[str, str] = field(default_factory=dict)
    evaluator: Optional[str] = None
    evaluator_kwargs: Dict[str, Any] = field(default_factory=dict)
    verifier_goals: Optional[str] = None
    problem_context: Optional[str] = None
    evaluator_runs_code: bool = False
    evaluator_python: Optional[str] = None
    
    def __post_init__(self):
        if self.run_id is None:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:6]}"
        if self.agent_models is None:
            self.agent_models = {}
        if self.evaluator_kwargs is None:
            self.evaluator_kwargs = {}




# =============================================================================
# ARTIFACT STORAGE SYSTEM
# =============================================================================

class ArtifactStorage:
    """Persistently stores every step of the pipeline for reproducibility."""
    
    def __init__(self, config: DSConfig):
        self.config = config
        self.run_dir = Path(config.runs_dir) / config.run_id
        self._setup_directories()
        
    def _setup_directories(self):
        """Create directory structure for this run."""
        dirs = [
            self.run_dir,
            self.run_dir / "steps",
            self.run_dir / "data_cache",
            self.run_dir / "logs",
            self.run_dir / "final_output"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            
    def save_step(self, step_id: str, step_type: str, prompt: str, 
                  code: Optional[str], result: str, metadata: Dict[str, Any]):
        """Save all artifacts for a single pipeline step."""
        step_dir = self.run_dir / "steps" / step_id
        step_dir.mkdir(exist_ok=True)
        
        # Save prompt
        (step_dir / "prompt.md").write_text(prompt, encoding='utf-8')
        
        # Save generated code
        if code:
            (step_dir / "code.py").write_text(code, encoding='utf-8')
            
        # Save execution result
        (step_dir / "result.txt").write_text(result, encoding='utf-8')
        
        # Save metadata
        metadata.update({
            "timestamp": datetime.now().isoformat(),
            "step_type": step_type,
            "step_id": step_id
        })
        with open(step_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
    def get_step(self, step_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a previous step's artifacts."""
        step_dirs = list(self.run_dir.glob(f"steps/{step_id}_*"))
        if not step_dirs:
            return None
        
        step_dir = step_dirs[0]
        return {
            "prompt": (step_dir / "prompt.md").read_text(encoding='utf-8'),
            "code": (step_dir / "code.py").read_text(encoding='utf-8') 
                   if (step_dir / "code.py").exists() else None,
            "result": (step_dir / "result.txt").read_text(encoding='utf-8'),
            "metadata": json.loads((step_dir / "metadata.json").read_text())
        }
    
    def list_steps(self) -> List[Dict[str, Any]]:
        """List all steps in chronological order."""
        steps = []
        for step_path in sorted(self.run_dir.glob("steps/*")):
            with open(step_path / "metadata.json") as f:
                metadata = json.load(f)
                steps.append(metadata)
        return steps
    
    def get_current_state(self) -> Dict[str, Any]:
        """Load the pipeline state."""
        state_file = self.run_dir / "pipeline_state.json"
        if state_file.exists():
            return json.loads(state_file.read_text())
        return {"current_step": 0, "completed_steps": [], "plan": [], "data_descriptions": {}}
    
    def save_state(self, state: Dict[str, Any]):
        """Save the pipeline state."""
        state_file = self.run_dir / "pipeline_state.json"
        state_file.write_text(json.dumps(state, indent=2))

# =============================================================================
# STATE MANAGEMENT & EXECUTION CONTROL
# =============================================================================

class PipelineController:
    """Manages pipeline execution with resume and editing capabilities."""
    
    def __init__(self, config: DSConfig, storage: ArtifactStorage, agent: 'DS_STAR_Agent'):
        self.config = config
        self.storage = storage
        self.agent = agent
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup structured logging."""
        import logging
        log_file = self.storage.run_dir / "logs" / "pipeline.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    def should_execute_step(self, step_index: int) -> bool:
        """Check if step should be executed (for resuming)."""
        state = self.storage.get_current_state()
        return step_index >= state["current_step"]
    
    def execute_step(self, step_name: str, step_func, **kwargs) -> Any:
        """Execute a single step with full artifact preservation."""
        step_id = f"{self._get_next_step_index():03d}_{step_name}"
        
        self.logger.info(f"{'='*50}")
        self.logger.info(f"STEP {step_id}")
        self.logger.info(f"{'='*50}")
        
        # Execute the step
        result = step_func(**kwargs)
        
        # Save artifacts
        if self.config.preserve_artifacts:
            metadata = kwargs.copy()
            code = result.get("code") if isinstance(result, dict) else None
            self.storage.save_step(
                step_id=step_id,
                step_type=step_name,
                prompt=kwargs.get("prompt", ""),
                code=code,
                result=str(result.get("result") if isinstance(result, dict) else result),
                metadata=metadata
            )
        
        # Update state
        state = self.storage.get_current_state()
        state["current_step"] = self._get_next_step_index()
        state["completed_steps"].append(step_id)
        self.storage.save_state(state)
        
        # Interactive mode
        if self.config.interactive:
            input(f"Step {step_id} complete. Press Enter to continue...")
            
        return result
    
    def _get_next_step_index(self) -> int:
        """Get the next step index."""
        steps = self.storage.list_steps()
        return len(steps)
    
    def edit_last_step_code(self):
        """Allow manual editing of the last generated code."""
        steps = self.storage.list_steps()
        if not steps:
            return
        
        last_step = steps[-1]
        step_dir = self.storage.run_dir / "steps" / f"{last_step['step_id']}_{last_step['step_type']}"
        code_file = step_dir / "code.py"
        
        if code_file.exists():
            self.logger.info(f"Opening {code_file} for editing...")
            # Use system editor
            editor = os.environ.get('EDITOR', 'nano')
            subprocess.run([editor, str(code_file)])
            self.logger.info("Code updated. Re-executing step...")
            # Re-execute the modified code
            code = code_file.read_text()
            result, error = self.agent._execute_code(code)
            (step_dir / "result.txt").write_text(result)
            (step_dir / "metadata.json").write_text(json.dumps({**last_step, "edited": True}, indent=2))

# =============================================================================
# CORE AGENT (Refactored)
# =============================================================================

class DS_STAR_Agent:
    """DS-STAR agent with persistent artifact storage."""
    
    def __init__(self, config: DSConfig):
        self.config = config
        self.storage = ArtifactStorage(config)
        self.controller = PipelineController(config, self.storage, self)
        # Initialize providers for each agent type
        self.providers = {}
        default_model = config.model_name
        
        # List of known agents
        agents = ["PLANNER", "CODER", "VERIFIER", "ROUTER", "DEBUGGER", "FINALYZER"]
        
        def get_provider_for_model(model_name: str, config: DSConfig) -> ModelProvider:
            provider_cls = None
            for provider in [OllamaProvider, OpenAIProvider, GeminiProvider]:
                if provider.provider_instance(model_name):
                    provider_cls = provider
                    break

            if not provider_cls:
                raise ValueError(f"No provider found for model {model_name}")
            
            return provider_cls(config.api_key, model_name)

        for agent in agents:
            model_name = config.agent_models.get(agent, default_model)
            self.providers[agent] = get_provider_for_model(model_name, config)
            self.controller.logger.info(f"Initialized {agent} with model: {model_name}")
        
        # Setup execution environment
        self.exec_dir = Path(config.runs_dir) / config.run_id / "exec_env"
        self.exec_dir.mkdir(exist_ok=True)

        self.evaluator = load_evaluator(config.evaluator, config.evaluator_kwargs)
        self._setup_tee_logging()
        
    def _setup_tee_logging(self):
        """Tee stdout/stderr to both console and log file."""
        log_path = self.storage.run_dir / "logs" / "execution.log"
        self.log_file = open(log_path, 'a', encoding='utf-8')
        
        class _Tee:
            def __init__(self, *writers):
                self.writers = writers
            def write(self, data):
                for w in self.writers:
                    try:
                        w.write(data)
                        w.flush()
                    except Exception:
                        pass
            def flush(self):
                for w in self.writers:
                    try:
                        w.flush()
                    except Exception:
                        pass
        
        sys.stdout = _Tee(sys.stdout, self.log_file)
        sys.stderr = _Tee(sys.stderr, self.log_file)
        
        atexit.register(lambda: self.log_file.close())
    
    def _call_model(self, agent_name: str, prompt: str) -> str:
        """Call the appropriate model provider for the agent."""
        try:
            provider = self.providers[agent_name]
            response_text = provider.generate_content(prompt)
            self.controller.logger.info(f"[{agent_name}] Response received ({len(response_text)} chars)")
            return response_text
        except Exception as e:
            error_msg = f"Error calling model for {agent_name}: {str(e)}"
            self.controller.logger.error(error_msg)
            raise
    
    def _extract_code_block(self, response: str) -> str:
        """Extract Python code from markdown blocks."""
        code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', response, re.DOTALL)
        return code_blocks[0] if code_blocks else response.strip()
    
    def _execute_code(self, code_script: str, data_files: Optional[List[str]] = None) -> Tuple[str, Optional[str]]:
        """Execute code in isolated environment."""
        self.controller.logger.info("Executing code...")
        
        # Validate data files
        if data_files:
            missing = []
            for f in data_files:
                p = Path(f)
                if p.is_absolute():
                    if not p.exists():
                        missing.append(f)
                else:
                    if not (Path(self.config.data_dir) / f).exists():
                        missing.append(f)
            if missing:
                return "", f"Missing data files: {missing}"
        
        # Write to persistent location
        exec_id = uuid.uuid4().hex[:8]
        exec_path = self.exec_dir / f"exec_{exec_id}.py"
        exec_path.write_text(code_script, encoding='utf-8')
        
        try:
            result = subprocess.run(
                [sys.executable, str(exec_path)],
                capture_output=True,
                text=True,
                timeout=self.config.execution_timeout,
                cwd=Path.cwd()
            )
            
            if result.returncode == 0:
                self.controller.logger.info("Execution successful")
                return result.stdout, None
            else:
                error_msg = result.stderr or "Unknown execution error"
                self.controller.logger.error(f"Execution failed: {error_msg}")
                return "", error_msg
                
        except subprocess.TimeoutExpired:
            return "", f"Timeout after {self.config.execution_timeout}s"
        except Exception as e:
            return "", f"Execution error: {str(e)}"

    def _execute_and_debug_code(
        self,
        code: str,
        data_files: List[str],
        data_desc: str,
        problem_context: str,
    ) -> str:
        exec_result, error = self._execute_code(code, data_files)

        # Debug loop
        attempts = 0
        while error and self.config.auto_debug and attempts < self.config.debug_attempts:
            self.controller.logger.warning("Debugging...")
            code = self._debug_code(code, error, data_desc, data_files, problem_context)
            exec_result, error = self._execute_code(code, data_files)
            attempts += 1

        if error:
            self.controller.logger.fatal(f"Execution error: {error}")
        return exec_result

    def _normalize_eval_result(self, eval_result: Any) -> Dict[str, Any]:
        if isinstance(eval_result, dict):
            normalized = dict(eval_result)
            parsed = normalized.get("parsed") or normalized.get("parsed_output") or normalized.get("json")
            raw_text = normalized.get("raw_text")
            if raw_text is None:
                raw_text = normalized.get("stdout")
            if raw_text is None:
                raw_text = json.dumps(eval_result, ensure_ascii=True)
            normalized["raw_text"] = raw_text
            normalized["parsed"] = parsed
            return normalized
        if isinstance(eval_result, str):
            parsed = None
            try:
                parsed = json.loads(eval_result)
            except Exception:
                parsed = None
            return {"raw_text": eval_result, "parsed": parsed}
        return {"raw_text": str(eval_result), "parsed": None}

    def _evaluate_candidate(
        self,
        code: str,
        result: str,
        plan: List[str],
        query: str,
        data_desc: str,
        data_files: Optional[List[str]] = None,
        problem_context: str = "",
    ) -> Dict[str, Any]:
        context = {
            "code": code,
            "result": result,
            "plan": plan,
            "query": query,
            "data_desc": data_desc,
            "data_files": data_files or [],
            "problem_context": problem_context,
            "cwd": str(Path.cwd()),
            "execution_timeout": self.config.execution_timeout,
            "python_executable": self.config.evaluator_python,
        }
        try:
            eval_result = self.evaluator.evaluate(context)
            return self._normalize_eval_result(eval_result)
        except Exception as e:
            self.controller.logger.error(f"Evaluator error: {e}")
            return {"raw_text": "", "parsed": None}

    def _execute_with_evaluator(
        self,
        code: str,
        data_files: List[str],
        data_desc: str,
        plan: List[str],
        query: str,
        problem_context: str,
    ) -> Tuple[str, Dict[str, Any]]:
        attempts = 0
        while True:
            eval_result = self._evaluate_candidate(
                code,
                "",
                plan,
                query,
                data_desc,
                data_files=data_files,
                problem_context=problem_context,
            )
            error = eval_result.get("error") or eval_result.get("stderr")
            if not error or not self.config.auto_debug or attempts >= self.config.debug_attempts:
                return eval_result.get("raw_text", ""), eval_result
            self.controller.logger.warning("Evaluator execution error; debugging...")
            code = self._debug_code(code, error, data_desc, data_files, problem_context)
            attempts += 1


    def analyze_data(self, filename: str, problem_context: str) -> Dict[str, str]:
        prompt = PROMPT_TEMPLATES["analyzer"].format(
            filename=filename,
            problem_context=problem_context,
        )
        
        result = self.controller.execute_step(
            "analyzer",
            step_func=lambda prompt=prompt, **kwargs: self._call_model("ANALYZER", prompt),  # FIXED
            prompt=prompt,
            filename=filename
        )
        
        code = self._extract_code_block(result)
        exec_result = self._execute_and_debug_code(code, [filename], data_desc="", problem_context=problem_context)
        
        return {"code": code, "result": exec_result, "filename": filename}

    def plan_next_step(
        self,
        query: str,
        data_desc: str,
        current_plan: List[str],
        last_result: Optional[str],
        problem_context: str,
    ) -> str:
        if not current_plan:
            prompt = PROMPT_TEMPLATES["planner_init"].format(
                question=query,
                summaries=data_desc,
                problem_context=problem_context,
            )
            step_type = "planner_init"
        else:
            plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(current_plan))
            prompt = PROMPT_TEMPLATES["planner_next"].format(
                question=query,
                summaries=data_desc,
                plan=plan_str,
                result=last_result,
                current_step=current_plan[-1],
                problem_context=problem_context,
            )
            step_type = "planner_next"
        
        return self.controller.execute_step(
            step_type,
            step_func=lambda prompt=prompt, **kwargs: self._call_model("PLANNER", prompt),  # FIXED
            prompt=prompt,
            query=query,
            plan_length=len(current_plan)
        )

    def generate_code(
        self,
        plan: List[str],
        data_desc: str,
        base_code: Optional[str] = None,
        problem_context: str = "",
    ) -> str:
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        
        if not base_code:
            prompt = PROMPT_TEMPLATES["coder_init"].format(
                summaries=data_desc,
                plan=plan_str,
                problem_context=problem_context,
            )
        else:
            prompt = PROMPT_TEMPLATES["coder_next"].format(
                summaries=data_desc,
                base_code=base_code,
                plan=plan_str,
                current_plan=plan[-1],
                problem_context=problem_context,
            )
        
        result = self.controller.execute_step(
            "coder",
            step_func=lambda prompt=prompt, **kwargs: self._call_model("CODER", prompt),  # FIXED
            prompt=prompt,
            plan_length=len(plan),
            has_base_code=base_code is not None
        )
        
        return self._extract_code_block(result)

    def verify_plan(
        self,
        plan: List[str],
        code: str,
        result: str,
        query: str,
        data_desc: str,
        eval_result: Dict[str, Any],
        goals: Optional[str],
        prev_eval_result: Optional[Dict[str, Any]] = None,
        problem_context: str = "",
    ) -> str:
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        goals_str = goals or "No explicit goals provided."
        eval_text = eval_result.get("raw_text", "")
        prev_eval_text = ""
        if prev_eval_result:
            prev_eval_text = prev_eval_result.get("raw_text", "")
        prompt = PROMPT_TEMPLATES["verifier"].format(
            plan=plan_str,
            code=code,
            result=result,
            question=query,
            summaries=data_desc,
            current_step=plan[-1],
            goals=goals_str,
            eval_result=eval_text,
            prev_eval_result=prev_eval_text,
            problem_context=problem_context,
        )
        
        return self.controller.execute_step(
            "verifier",
            step_func=lambda prompt=prompt, **kwargs: self._call_model("VERIFIER", prompt),  # FIXED
            prompt=prompt,
            plan_length=len(plan)
        ).strip()

    def route_plan(
        self,
        plan: List[str],
        query: str,
        result: str,
        data_desc: str,
        problem_context: str,
    ) -> str:
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        prompt = PROMPT_TEMPLATES["router"].format(
            question=query,
            summaries=data_desc,
            plan=plan_str,
            result=result,
            current_step=plan[-1],
            problem_context=problem_context,
        )
        
        return self.controller.execute_step(
            "router",
            step_func=lambda prompt=prompt, **kwargs: self._call_model("ROUTER", prompt),  # FIXED
            prompt=prompt,
            plan_length=len(plan)
        ).strip()

    def _debug_code(
        self,
        code: str,
        error: str,
        data_desc: str,
        filenames: List[str],
        problem_context: str,
    ) -> str:
        prompt = PROMPT_TEMPLATES["debugger"].format(
            summaries=data_desc, code=code,
            bug=error,
            filenames=", ".join(filenames),
            problem_context=problem_context,
        )
        
        result = self.controller.execute_step(
            "debugger",
            step_func=lambda prompt=prompt, **kwargs: self._call_model("DEBUGGER", prompt),  # FIXED
            prompt=prompt,
            error_type=error.split(":")[0]
        )
        
        return self._extract_code_block(result)

    def finalize_solution(
        self,
        code: str,
        result: str,
        query: str,
        guidelines: str,
        data_desc: str,
        problem_context: str,
    ) -> str:
        prompt = PROMPT_TEMPLATES["finalyzer"].format(
            summaries=data_desc,
            code=code,
            result=result,
            question=query,
            guidelines=guidelines,
            problem_context=problem_context,
        )
        
        result = self.controller.execute_step(
            "finalyzer",
            step_func=lambda prompt=prompt, **kwargs: self._call_model("FINALYZER", prompt),  # FIXED
            prompt=prompt
        )
        
        return self._extract_code_block(result)

    def run_pipeline(self, query: str, data_files: Optional[List[str]]) -> Dict[str, Any]:
        """Main pipeline with full persistence and resume capability."""
        self.controller.logger.info(f"Starting pipeline: {self.config.run_id}")
        self.controller.logger.info(f"Query: {query}")
        self.controller.logger.info(f"Data files: {data_files}")
        if data_files:
            self.controller.logger.warning("Data files are ignored; analyzer is disabled.")
        
        # Check for resume state
        state = self.storage.get_current_state()
        if state["completed_steps"]:
            self.controller.logger.info(f"Resuming from step {state['current_step']}")
        
        # Initialize variables that might be loaded from previous runs
        code = None
        exec_result = None
        
        # Analyzer phase removed; data descriptions are empty by design.
        data_descriptions = {}
        absolute_data_files = []
        data_desc_str = ""
        
        # PHASE 1: Iterative Planning & Execution
        if self.controller.should_execute_step(0):
            self.controller.logger.info("=== PHASE 1: ITERATIVE PLANNING & VERIFICATION ===")
            plan = []
            plan.append(
                self.plan_next_step(
                    query,
                    data_desc_str,
                    plan,
                    "",
                    problem_context=self.config.problem_context or "",
                )
            )
            
            code = self.generate_code(plan, data_desc_str, problem_context=self.config.problem_context or "")
            if self.config.evaluator_runs_code:
                exec_result, eval_result = self._execute_with_evaluator(
                    code,
                    absolute_data_files,
                    data_desc_str,
                    plan,
                    query,
                    problem_context=self.config.problem_context or "",
                )
            else:
                exec_result = self._execute_and_debug_code(
                    code,
                    absolute_data_files,
                    data_desc_str,
                    problem_context=self.config.problem_context or "",
                )
                eval_result = None
            
            # Refinement rounds
            prev_eval_result = None
            for round_idx in range(self.config.max_refinement_rounds):
                self.controller.logger.info(f"--- Refinement Round {round_idx+1} ---")

                if not self.config.evaluator_runs_code:
                    eval_result = self._evaluate_candidate(
                        code,
                        exec_result,
                        plan,
                        query,
                        data_desc_str,
                        data_files=absolute_data_files,
                        problem_context=self.config.problem_context or "",
                    )
                if eval_result.get("parsed") and eval_result["parsed"].get("should_stop"):
                    self.controller.logger.info("Evaluator requested stop; proceeding to finalization.")
                    break

                verdict = self.verify_plan(
                    plan,
                    code,
                    exec_result,
                    query,
                    data_desc_str,
                    eval_result=eval_result or {"raw_text": "", "parsed": None},
                    goals=self.config.verifier_goals,
                    prev_eval_result=prev_eval_result,
                    problem_context=self.config.problem_context or "",
                )
                prev_eval_result = eval_result
                
                if verdict.lower() == "yes":
                    self.controller.logger.info("Plan verified as sufficient!")
                    break
                
                routing = self.route_plan(
                    plan,
                    query,
                    exec_result,
                    data_desc_str,
                    problem_context=self.config.problem_context or "",
                )
                
                if "is wrong!" in routing:
                    # Truncate plan and retry
                    try:
                        step_to_remove = int(routing.split()[1]) - 1
                        plan = plan[:step_to_remove]
                        self.controller.logger.info(f"Truncated plan to step {step_to_remove}")
                    except:
                        plan = []
                else:
                    self.controller.logger.info("Adding new step...")
                
                # Generate next step
                next_plan = self.plan_next_step(
                    query,
                    data_desc_str,
                    plan,
                    exec_result,
                    problem_context=self.config.problem_context or "",
                )
                plan.append(next_plan)
                
                # Generate and execute new code
                code = self.generate_code(
                    plan,
                    data_desc_str,
                    base_code=code,
                    problem_context=self.config.problem_context or "",
                )
                if self.config.evaluator_runs_code:
                    exec_result, eval_result = self._execute_with_evaluator(
                        code,
                        absolute_data_files,
                        data_desc_str,
                        plan,
                        query,
                        problem_context=self.config.problem_context or "",
                    )
                else:
                    exec_result = self._execute_and_debug_code(
                        code,
                        absolute_data_files,
                        data_desc_str,
                        problem_context=self.config.problem_context or "",
                    )
                    eval_result = None
            else:
                self.controller.logger.warning("Max refinement rounds reached")
        
        # Load code and exec_result from previous run if not defined (resuming case)
        if code is None or exec_result is None:
            steps = self.storage.list_steps()
            
            # Find the last step with code
            for step in reversed(steps):
                step_data = self.storage.get_step(step['step_id'])
                if step_data and step_data.get('code'):
                    code = step_data['code']
                    exec_result = step_data.get('result', '')
                    self.controller.logger.info(f"Loaded code and results from step {step['step_id']}")
                    break
            
            # If still not found, this is an error
            if code is None:
                raise ValueError("Could not load code from previous steps. Please ensure the pipeline has been run before.")
            if exec_result is None:
                exec_result = ""  # Default to empty string if not found
        
        # PHASE 3: Finalization
        self.controller.logger.info("=== PHASE 3: FINALIZING ===")
        final_code = self.finalize_solution(
            code,
            exec_result,
            query,
            "Format as JSON with key 'final_answer'",
            data_desc_str,
            problem_context=self.config.problem_context or "",
        )
        
        if self.config.evaluator_runs_code:
            final_result, _ = self._execute_with_evaluator(
                final_code,
                absolute_data_files,
                data_desc_str,
                plan if "plan" in locals() else [],
                query,
                problem_context=self.config.problem_context or "",
            )
        else:
            final_result = self._execute_and_debug_code(
                final_code,
                absolute_data_files,
                data_desc_str,
                problem_context=self.config.problem_context or "",
            )
        
        # Save final output
        output_file = self.storage.run_dir / "final_output" / "result.json"
        output_file.write_text(final_result)
        
        self.controller.logger.info("Pipeline completed successfully!")
        
        return {
            "run_id": self.config.run_id,
            "final_result": final_result,
            "output_file": str(output_file),
            "total_steps": len(self.storage.list_steps())
        }
# =============================================================================
# CLI & USAGE
# =============================================================================

def main():
    """CLI interface with resume and edit capabilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DS-STAR Data Science Agent")
    parser.add_argument("--resume", type=str, help="Resume from run ID")
    parser.add_argument("--interactive", action="store_true", help="Pause between steps")
    parser.add_argument("--edit-last", action="store_true", help="Edit last generated code")
    parser.add_argument("--data-files", nargs="+", help="Data files to analyze")
    parser.add_argument("--query", type=str, help="Deprecated: use query in config YAML instead")
    parser.add_argument("--max-rounds", type=int, help="Max refinement rounds")
    parser.add_argument("--config", type=str, help="Path to config file", default="config.yaml")
    args = parser.parse_args()

    # Load config from file to set defaults
    try:
        with open(args.config, 'r') as f:
            config_defaults = yaml.safe_load(f) or {}
    except FileNotFoundError:
        config_defaults = {}
    
    # Combine config sources (CLI args take precedence)
    config_params = {
        'run_id': args.resume or config_defaults.get('run_id'),
        'interactive': args.interactive or config_defaults.get('interactive', False),
        'max_refinement_rounds': args.max_rounds or config_defaults.get('max_refinement_rounds', 5),
        'model_name': config_defaults.get('model_name'),
        'preserve_artifacts': config_defaults.get('preserve_artifacts', True),
        'evaluator': config_defaults.get('evaluator'),
        'evaluator_kwargs': config_defaults.get('evaluator_kwargs'),
        'verifier_goals': config_defaults.get('verifier_goals'),
        'problem_context': config_defaults.get('problem_context'),
        'evaluator_runs_code': config_defaults.get('evaluator_runs_code', False),
        'evaluator_python': config_defaults.get('evaluator_python'),
    }
    
    # Filter out None values so dataclass defaults are used
    config_params = {k: v for k, v in config_params.items() if v is not None}
    
    config = DSConfig(**config_params)
    if not config.model_name:
        parser.error("Model name must be specified via config file.")
    
    agent = DS_STAR_Agent(config)
    
    # Edit mode
    if args.edit_last and config.run_id:
        agent.controller.edit_last_step_code()
        return

    # Check for required arguments for a new run
    if args.query:
        print("Warning: --query is deprecated; define 'query' in the config YAML instead.")
    query = args.query or config_defaults.get('query')
    data_files = args.data_files or config_defaults.get('data_files')

    if not query:
        parser.error("Query is required in the config YAML for a new run.")

    # Run pipeline
    result = agent.run_pipeline(query, data_files)
    print(f"\n{'='*60}")
    print(f"RUN COMPLETED: {result['run_id']}")
    print(f"OUTPUT: {result['output_file']}")
    print(f"FINAL RESULT:\n{result['final_result']}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
