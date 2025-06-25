import importlib
import io
import logging
import os
import sys
import json
import tempfile
from time import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, Tuple, Any

from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

class Problem:
    """Class for problem definition and evaluation metrics."""
    
    def __init__(self, name, description, workspace, interface, debugger_agent, initial_code, max_retry_times: int = 5):
        """
        Initialize a problem with config.
        
        Args:
            config: Configuration object containing problem settings
        """
        self.name = name
        self.description = description
        self.workspace = workspace
        self.interface = interface
        self.debugger_agent = debugger_agent
        self.max_retry_times = max_retry_times
        self.execution_time = []

        if f"# === {self.interface} ===" not in initial_code:
            raise ValueError("Initial code does not contain the interface file.")

    async def _debugging(self, code: str, message: str, retry_count: int):
        """
        Debug the code and return the debugged code.
        """
        if self.debugger_agent is not None and retry_count < self.max_retry_times:
            try:
                console.print(f"[bold yellow] Attempting to debug code...[/bold yellow]")
                debugged_code = await self.debugger_agent.debug(code, message)
                retry_count += 1

                # # save to self.workspace/problem_name/tmp/debugged_code_{retry_count}.py
                # os.makedirs(os.path.join(self.workspace, "tmp"), exist_ok=True)
                # with open(os.path.join(self.workspace, "tmp", f"debugged_code_{retry_count}.py"), "w") as f:
                #     f.write(debugged_code)

                logger.info(f"Retrying after debugging (attempt {retry_count}/{self.max_retry_times})...")
                return True, debugged_code, retry_count
            except Exception as debug_error:
                console.print(f"[bold red] Failed to debug code: {debug_error}[/bold red]")
                return False, code, retry_count
        else:
            logger.info(f"Skipping debugging because debugger_agent is not provided or {retry_count} >= max retry times ({self.max_retry_times}).")
            return False, code, retry_count

    async def evaluate(
        self, code: str, program_id: str, is_initial: bool = False,
    ) -> Tuple[Dict[str, float], str]:
        """
        Execute the evaluation function of the code and return a tuple of (metrics dict, code).
        """
        if is_initial:
            metrics_path = os.path.join(self.workspace, "initial_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                return metrics, code
            else:
                logger.warning(f"Initial metrics file not found at {metrics_path}. Running initial evaluation from scratch.")

        current_code = code
        retry_count = 0
        if self.debugger_agent is not None:
            logger.info(f"Starting evaluation with {retry_count} retries.")
        else:
            logger.info(f"Starting evaluation without debugging.")
        while retry_count <= self.max_retry_times:
            # Parse the concatenated code to extract individual files
            files: Dict[str, str] = {}
            current_file = None
            current_content = []

            for line in current_code.split("\n"):
                if line.startswith("# === ") and line.endswith(" ==="):
                    if current_file is not None:
                        files[current_file] = "\n".join(current_content)
                    current_file = line[6:-4]  # remove "# === " and " ==="
                    current_content = []
                else:
                    if current_file is not None:
                        current_content.append(line)
            if current_file is not None:
                files[current_file] = "\n".join(current_content)

            # Ensure the interface file is present
            if self.interface not in files:
                debug_success, debug_result, retry_count = await self._debugging(current_code, f"Interface file {self.interface} not found in the code.", retry_count)
                if debug_success:
                    current_code = debug_result
                    continue
                else:
                    return {"combined_score": -1.0}, current_code

            #  Write each file into a temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                for filename, content in files.items():
                    file_path = os.path.join(tmpdir, filename)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)

                # Insert tmpdir at front of sys.path so imports resolve to these files
                sys.path.insert(0, tmpdir)

                # Unload any existing modules that overlap with inner filenames
                original_modules: Dict[str, Any] = {}
                for filename in files:
                    if not filename.endswith(".py"):
                        continue
                    mod_name = Path(filename).stem
                    if mod_name in sys.modules:
                        original_modules[mod_name] = sys.modules.pop(mod_name)

                try:
                    if len(self.execution_time) > 0:
                        logger.info(f"Executing the program with estimated time (if success): {sum(self.execution_time) / len(self.execution_time) / 60:.2f} minutes")
                    captured_output = io.StringIO()
                    
                    # Import the interface module by its bare name
                    interface_name = Path(self.interface).stem  # e.g. "deepevolve_interface"
                    
                    with redirect_stdout(captured_output):
                        start_time = time()
                        try:
                            interface_module = importlib.import_module(interface_name)
                            execute_success, message = interface_module.deepevolve_interface()
                        except Exception as e:
                            debug_success, debug_result, retry_count = await self._debugging(current_code, f"interface_module implementation failed with error: {e}", retry_count)
                            if debug_success:
                                current_code = debug_result
                                continue
                            else:
                                return {"combined_score": 0.0}, current_code
                        end_time = time()
                    
                    if execute_success:
                        self.execution_time.append(end_time - start_time)
                    
                    # Get captured output, not used for now
                    # program_output = captured_output.getvalue().strip()
                    # print(program_output)
                    
                    if execute_success:
                        logger.info(f"Program successfully executed with return metrics: {message}")
                        assert isinstance(message, dict), "The interface function should return message as a dictionary if success."
                        if is_initial:
                            metrics_path = os.path.join(self.workspace, "initial_metrics.json")
                            with open(metrics_path, "w") as f:
                                json.dump(message, f, indent=2)
                            logger.info(f"Initial metrics saved to {metrics_path}")
                        return message, current_code
                    else:
                        if is_initial:
                            logger.info(f"Program failed to execute with error: {message}")
                            raise Exception(f"Initial program failed to execute. Please debug the initial code and try again.")
  
                        logger.info(f"Program failed to execute with error: {message}, with remaining retry attempts: {self.max_retry_times - retry_count}")
                        debug_success, debug_result, retry_count = await self._debugging(current_code, message, retry_count)
                        if debug_success:
                            current_code = debug_result
                            continue
                        else:
                            return {"combined_score": 0.0}, current_code

                finally:
                    # Clean up sys.path and restore any popped modules
                    sys.path.remove(tmpdir)
                    for name, module_obj in original_modules.items():
                        sys.modules[name] = module_obj
        
        # If exhausted all retries, return 0.0
        logger.info(f"Program failed to execute after {self.max_retry_times} retries.")
        return {"combined_score": 0.0}, current_code