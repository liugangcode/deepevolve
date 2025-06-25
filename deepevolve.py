import asyncio
import logging
import os
import time
import uuid
import json
from pathlib import Path
from typing import Optional
import hydra
from omegaconf import DictConfig
from agents.tracing import gen_trace_id

from coder import CoderAgent
from researcher import ResearcherAgent
from problem import Problem
from database import Program, ProgramDatabase
from utils.code import get_files_and_code, parse_evolve_blocks, save_code_to_files
from utils.datatypes import IdeaData
from utils.format import format_metrics_safe, format_improvement_safe

from rich.console import Console

logger = logging.getLogger(__name__)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

class DeepEvolve:
    """
    DeepEvolve: Evolutionary Optimization of Scientific Algorithms with Deep Research
    """
    def __init__(self, config: DictConfig, query: str):
        self.config = config
        self.query = query
        self.language = "python"
        self.code_extension = ".py"
        self.problem_name = self.config.problem
        self.workspace = os.path.join(self.config.workspace, self.problem_name)
        self.checkpoint = self.config.get("checkpoint", "checkpoints")

        self.researcher = ResearcherAgent(**self.config.researcher)
        self.coder = CoderAgent(**self.config.coder)
        self.trace_id = gen_trace_id()
        self._setup_logging()
        self.console = Console()

        if os.path.exists(os.path.join(self.workspace, "info.json")):
            with open(os.path.join(self.workspace, "info.json"), "r", encoding="utf-8") as f:
                info = json.load(f)
            problem_info = info['problem']
            initial_idea_info = info['initial_idea']
        else:
            raise ValueError(f"info.json not found in the task directory {self.workspace}, which should provide two keys: problem and initial_idea.")

        _, initial_code = get_files_and_code(
            local_path=os.path.join(self.workspace, "initial_code"),
            online_link=None,
            workspace_dir=self.workspace,
            code_extension=self.code_extension,
        )
        if len(initial_code) == 0:
            raise ValueError(f"No initial code found in the task directory {self.workspace}, which should provide one or more code files in the initial_code folder.")

        self.problem = Problem(
            self.problem_name,
            problem_info["description"],
            self.workspace,
            problem_info["interface"],
            debugger_agent=self.coder,
            initial_code=initial_code,
            max_retry_times=self.config.max_debug_retry,
        )

        # Store problem context
        self.initial_idea_info = initial_idea_info
        self.initial_code = initial_code
        self.database = ProgramDatabase(self.config.database)

        # debug only
        self.debugging = False
        self.cache_dir = Path(f"examples/{self.problem_name}/tmp")
        if self.debugging:
            os.makedirs(self.cache_dir, exist_ok=True)        

    def _setup_logging(self) -> None:
        """Set up logging (remove old handlers and include module name in each record)."""
        # Remove any pre-existing handlers
        root = logging.getLogger()

        for handler in root.handlers[:]:
            root.removeHandler(handler)

        # Create log directory
        log_dir = self.config.log_dir or os.path.join(self.workspace, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Set root level
        root.setLevel(getattr(logging, self.config.log_level))

        # File handler: include module name and line number
        log_file = os.path.join(
            log_dir, f"deepevolve_{time.strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_fmt = "%(asctime)s - %(module)s:%(lineno)d - %(name)s - %(levelname)s - %(message)s"
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(file_fmt))
        root.addHandler(fh)

        # Console handler: show module name too
        console_fmt = "%(asctime)s - %(module)s:%(lineno)d - %(levelname)s - %(message)s"
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(console_fmt))
        root.addHandler(ch)

        logger.info(f"Logging to {log_file}")

    async def run(
        self,
        iterations: Optional[int] = None,
        target_score: Optional[float] = None,
    ) -> Program:
        """
        Run the evolution process

        Args:
            iterations: Maximum number of iterations (uses config if None)
            target_score: Target score to reach (continues until reached if specified)

        Returns:
            Best program found
        """

        self.researcher.update_topic(
            self.query,
            self.problem_name,
            self.problem.description,
            self.config.search_time_bias,
        )
        self.coder.update_topic(
            self.query,
            self.problem_name,
            self.problem.description,
        )

        # Define start_iteration before creating the initial program
        max_iterations = iterations or self.config.max_iterations
        start_iteration = self.database.last_iteration

        should_add_initial = (
            start_iteration == 0
            and len(self.database.programs) == 0
            and not any(
                p.code == self.initial_code for p in self.database.programs.values()
            )
        )

        if should_add_initial:
            self.console.rule("[bold green]Adding Initial Program to Database")
            logger.info("Adding initial program to database")

            if os.path.exists(os.path.join(self.workspace, "initial_idea.json")):
                with open(os.path.join(self.workspace, "initial_idea.json"), "r", encoding="utf-8") as f:
                    initial_idea = json.load(f)
                initial_idea = IdeaData(**initial_idea)
                self.console.print(
                    f"[green]Loaded initial idea from cache: {initial_idea}[/green]"
                )
            else:
                self.console.print(
                    f"[yellow]Cache file for the initial idea not found, running researcher...[/yellow]"
                )
                initial_idea = await self.researcher.read_paper(
                    self.initial_idea_info["title"], self.initial_idea_info["content"], self.initial_idea_info["supplement"]
                )
                with open(os.path.join(self.workspace, "initial_idea.json"), "w", encoding="utf-8") as f:
                    json.dump(initial_idea.model_dump(), f, indent=2)
                self.console.print(
                    f"[green]Cached initial idea to {os.path.join(self.workspace, 'initial_idea.json')}[/green]"
                )

            initial_metrics, initial_code = await self.problem.evaluate(
                self.initial_code,
                'root',
                is_initial=True,
            )

            initial_program = Program(
                id='root',
                code=self.initial_code,
                idea=initial_idea,
                parent_id="root",
                language=self.language,
                metrics=initial_metrics,
                iteration_found=start_iteration,
                evolution_history=[],
                report=self.initial_idea_info["content"],
            )
            self.database.add(initial_program)
        else:
            logger.info(
                f"Skipping initial program addition (resuming from iteration {start_iteration} with {len(self.database.programs)} existing programs)"
            )

        logger.info(
            f"Starting evolution from iteration {start_iteration} for remaining {max_iterations - start_iteration} iterations (total: {max_iterations})"
        )

        # Island-based evolution variables
        programs_per_island = max(
            1, self.config.database.population_size // self.config.database.num_islands
        )  # Dynamic allocation
        current_island_counter = 0

        logger.info(
            f"Using island-based evolution with {self.config.database.num_islands} islands"
        )
        self.database.log_island_status()

        for i in range(start_iteration, max_iterations):
            self.console.rule(f"[bold green]Iteration {i+1}")
            iteration_start = time.time()

            # Manage island evolution - switch islands periodically
            if i > start_iteration and current_island_counter >= programs_per_island:
                self.database.next_island()
                current_island_counter = 0
                logger.debug(f"Switched to island {self.database.current_island}")

            current_island_counter += 1

            # step 1: sampling parent and inspirations
            self.console.print(f"[yellow]Step 1: Sampling parent and inspirations...[/yellow]")
            parent, inspirations = self.database.sample()

            # step 2: deep research
            self.console.print(f"[yellow]Step 2: Running deep research...[/yellow]")
            research_plans, search_results, research_reports = (
                await self.researcher.run(
                    parent,
                    inspirations,
                    trace_id=self.trace_id,
                    max_reflection_times=self.config.max_research_reflect,
                )
            )

            research_report = research_reports[-1]
            new_idea = research_report.idea

            logger.info(f'-------------------------------- Iteration {i+1} Deep Research Outcome All START --------------------------------')            
            logger.info(f"Research plans ({len(research_plans)} plan(s)):")
            for idx, plan in enumerate(research_plans):
                logger.info(f"  Plan {idx+1}: {plan.model_dump_json(indent=2)}")            
            logger.info(f"Research reports ({len(research_reports)} report(s)):")
            for idx, report in enumerate(research_reports):
                logger.info(f"  Report {idx+1}: {report.markdown_report}")
            logger.info(f'-------------------------------- Iteration {i+1} Deep Research Outcome All END --------------------------------')
            logger.info(f"The new idea in iteration {i+1}:\n{new_idea.model_dump_json(indent=2)}")

            # step 3: coding
            self.console.print(f"[yellow]Step 3: Running algorithm coding...[/yellow]")
            all_diff_text, all_program_code = await self.coder.run(
                new_idea,
                parent,
                inspirations,
                trace_id=self.trace_id,
                max_reflection_times=self.config.max_coding_reflect,
            )

            all_blocks = []
            for program_code in all_program_code:
                blocks = parse_evolve_blocks(program_code)
                all_blocks.extend(blocks)
            if len(all_blocks) == 0:
                logger.warning(
                    f"Iteration {i+1}: No valid diff blocks are found in response, which has two implications: 1. the code is not changed, 2. the code is changed but not strictly following instructions to add valid block markers."
                )
                if self.debugging:
                    with open(
                        os.path.join(self.workspace, "tmp", "check_no_change_input.py"), "w", encoding="utf-8"
                    ) as f:
                        f.write(parent.code)
                    with open(
                        os.path.join(self.workspace, "tmp", "check_no_change_output.py"),
                        "w", encoding="utf-8"
                    ) as f:
                        f.write(all_program_code[-1])
            
            if self.debugging:
                last_diff_text = all_diff_text[-1]
                with open(
                    os.path.join(self.workspace, "tmp", "check_last_diff.py"), "w", encoding="utf-8"
                ) as f:
                    f.write(last_diff_text)
                with open(
                    os.path.join(self.workspace, "tmp", "check_last_program.py"), "w", encoding="utf-8"
                ) as f:
                    f.write(all_program_code[-1])

            child_code = all_program_code[-1]
            child_id = str(uuid.uuid4())
            
            # step 4: evaluation
            self.console.print(f"[yellow]Step 4: Running evaluation...[/yellow]")
            child_metrics, child_code = await self.problem.evaluate(
                child_code, child_id, is_initial=False
            )

            child_program = Program(
                id=child_id,
                code=child_code,
                idea=new_idea,
                parent_id=parent.id,
                language=self.language,
                metrics=child_metrics,
                iteration_found=i + 1,
                evolution_history=parent.evolution_history + [new_idea],
                report=research_report.markdown_report,
                metadata={
                    "parent_metrics": parent.metrics,
                },
            )

            # Add to database
            self.console.print(f"[yellow]After evaluation, updating database...[/yellow]")
            self.database.add(child_program, iteration=i + 1)

            # Increment generation for current island
            self.database.increment_island_generation()

            # Check if migration should occur
            if self.database.should_migrate():
                logger.info(f"Performing migration at iteration {i+1}")
                self.database.migrate_programs()
                self.database.log_island_status()

            # Log progress
            iteration_time = time.time() - iteration_start
            self._log_iteration(i, parent, child_program, iteration_time)

            # Specifically check if this is the new best program
            if self.database.best_program_id == child_program.id:
                logger.info(
                    f"🌟 New best program found at iteration {i+1}: {child_program.id}"
                )
                logger.info(f"Metrics: {format_metrics_safe(child_program.metrics)}")

            # Save checkpoint
            if (
                i == max_iterations - 1
                or (i + 1) % self.config.checkpoint_interval == 0
            ):
                self._save_checkpoint(i + 1)
                # Also log island status at checkpoints
                logger.info(f"Island status at checkpoint {i+1}:")
                self.database.log_island_status()

            # Check if target score reached
            if target_score is not None:
                avg_score = sum(child_metrics.values()) / max(1, len(child_metrics))
                if avg_score >= target_score:
                    logger.info(
                        f"Target score {target_score} reached after {i+1} iterations"
                    )
                    break

        # Get the best program using our tracking mechanism
        best_program = None
        if self.database.best_program_id:
            best_program = self.database.get(self.database.best_program_id)
            logger.info(f"Using tracked best program: {self.database.best_program_id}")

        # Check if there's a better program by combined_score that wasn't tracked
        best_by_combined = self.database.get_best_program(metric="combined_score")
        if (
            best_by_combined
            and best_by_combined.id != best_program.id
            and "combined_score" in best_by_combined.metrics
        ):
            logger.warning(
                f"Found program with better combined_score: {best_by_combined.id}"
            )
            logger.warning(
                f"Score difference: {best_program.metrics['combined_score']:.4f} vs {best_by_combined.metrics['combined_score']:.4f}"
            )
            best_program = best_by_combined

        if best_program:
            logger.info(
                f"Evolution complete. Best program has metrics: "
                f"{format_metrics_safe(best_program.metrics)}"
            )

            # Save the best program (using our tracked best program)
            self._save_best_program()
            if best_program.id == 'root':
                logger.warning("The best program is the initial program. No better performing program found.")

            return best_program
        else:
            logger.warning("No valid programs found during evolution")
            # Return None if no programs found instead of undefined initial_program
            return None

    def _log_iteration(
        self,
        iteration: int,
        parent: Program,
        child: Program,
        elapsed_time: float,
    ) -> None:
        """
        Log iteration progress

        Args:
            iteration: Iteration number
            parent: Parent program
            child: Child program
            elapsed_time: Elapsed time in seconds
        """
        improvement_str = format_improvement_safe(parent.metrics, child.metrics)

        logger.info(
            f"Iteration {iteration+1}: Child {child.id} from parent {parent.id} "
            f"in {elapsed_time:.2f}s. Metrics: "
            f"{format_metrics_safe(child.metrics)} "
            f"(Δ: {improvement_str})"
        )

    def _save_checkpoint(self, iteration: int) -> None:
        """
        Save a checkpoint

        Args:
            iteration: Current iteration number
        """
        checkpoint_dir = os.path.join(self.workspace, self.checkpoint)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create specific checkpoint directory
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save the database
        self.database.save(checkpoint_path, iteration)

        # Save the best program found so far
        best_program = None
        if self.database.best_program_id:
            best_program = self.database.get(self.database.best_program_id)
        else:
            best_program = self.database.get_best_program()

        if best_program:
            self._save_best_program()

            logger.info(
                f"Saved best program at checkpoint {iteration} with metrics: "
                f"{format_metrics_safe(best_program.metrics)}"
            )

        logger.info(f"Saved checkpoint at iteration {iteration} to {checkpoint_path}")

    def _save_best_program(self, program: Optional[Program] = None) -> None:
        """
        Save the best program

        Args:
            program: Best program (if None, uses the tracked best program)
        """
        # If no program is provided, use the tracked best program from the database
        if program is None:
            if self.database.best_program_id:
                program = self.database.get(self.database.best_program_id)
            else:
                # Fallback to calculating best program if no tracked best program
                program = self.database.get_best_program()

        if not program:
            logger.warning("No best program found to save")
            return

        best_dir = os.path.join(self.workspace, self.checkpoint, "best")
        os.makedirs(best_dir, exist_ok=True)

        # Use the extension from the initial program file
        filename = f"best_program_concatenated{self.code_extension}"
        code_path = os.path.join(best_dir, filename)
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(program.code)
        save_code_to_files(program.code, best_dir)

        # Save complete program info including metrics
        info_path = os.path.join(best_dir, "best_program_info.json")
        idea_evolution = program.evolution_history
        if len(idea_evolution) > 0:
            idea_evolution = " -> ".join(
                [f"[{i}] {idea.description}" for i, idea in enumerate(idea_evolution)]
            )
        else:
            idea_evolution = "Initial idea"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "id": program.id,
                    "parent_id": program.parent_id,
                    "idea": program.idea.model_dump(),
                    "generation": len(program.evolution_history),
                    "iteration_found": program.iteration_found,
                    "metrics": program.metrics,
                    "language": program.language,
                    "report": program.report,
                    "evolution_history": idea_evolution,
                    "saved_at": time.time(),
                    "timestamp": program.timestamp,
                },
                f,
                indent=2,
            )
        logger.info(
            f"Saved best program to {code_path} with program info to {info_path}"
        )
        if program.id == 'root':
            logger.warning("The best program is the initial program.")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    if "OPENAI_API_KEY" not in os.environ:
        openai_api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = openai_api_key
        print("OpenAI API key set from user input")
    else:
        print("Use the OpenAI API key set from environment variable")
    
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    query = config.get("query", "")
    if "problem" not in config:
        raise ValueError("Problem is not in the config")
    if not query:
        query = f"Improve machine learning methods for {config.problem}"
        
    deep_evolve = DeepEvolve(config=config, query=query)
    asyncio.run(deep_evolve.run())

if __name__ == "__main__":
    main()