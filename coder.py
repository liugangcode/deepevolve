from __future__ import annotations

import logging
from rich.console import Console

from agents import Agent, Runner
from agents.tracing import gen_trace_id, trace
from agents.model_settings import ModelSettings
from black import format_str, Mode

from database import Program
from utils.code import apply_diff, parse_evolve_blocks
from utils.datatypes import IdeaData, reasoning_models
from utils.format import format_metrics_safe

logger = logging.getLogger(__name__)

console = Console()

CODER_INSTRUCTIONS = """You are a researcher with strong software engineering skills, improving algorithmic code through iterative, performance-driven modifications in multiple rounds.

Your task:
You will receive a research question, a proposed idea, and an existing implementation with performance metrics. Your goal is to analyze the current code and apply precise changes that enhance the specified metrics, based on the research idea and prior feedback.

You MUST use the exact SEARCH/REPLACE diff format. Do NOT use Git diff format. Do NOT use line prefixes like `+`, `-`, or `@@`.
Use this structure exactly:
```
<<<<<<< SEARCH
# Original code (must match exactly)
=======
### >>> DEEPEVOLVE-BLOCK-START: <research idea>
# New code here
### <<< DEEPEVOLVE-BLOCK-END
>>>>>>> REPLACE
```
Example 1 for the code modification outside of `DEEPEVOLVE` blocks:
```
<<<<<<< SEARCH
def f():
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
=======
def f():
    # DEEPEVOLVE-BLOCK-START: Reordered loops for better cache performance
    for i in range(m):
        for k in range(n):
            for j in range(p):
                C[i, j] += A[i, k] * B[k, j]
    ### <<< DEEPEVOLVE-BLOCK-END
>>>>>>> REPLACE
```

Example 2 for the code modification inside of `DEEPEVOLVE` blocks:
```
<<<<<<< SEARCH
### >>> DEEPEVOLVE-BLOCK-START: <research idea>
# Code to be modified
### <<< DEEPEVOLVE-BLOCK-END
=======
### >>> DEEPEVOLVE-BLOCK-START: <update idea>
# New code here
### <<< DEEPEVOLVE-BLOCK-END
>>>>>>> REPLACE
```

Task Guidelines:
1. Think before coding, understand the research idea and current performance bottlenecks.
2. Propose specific, actionable changes that are aligned with the target metrics.
3. You may suggest multiple improvements beyond the research idea based on your understanding of optimization and machine learning.
4. When you are updating the code, please check the following:
    - When a NEW parameter or behavior is added, verify it is invoked in all call sites or in the overall workflow.
    - If a NEW parameter has a default value of None, confirm that passing a non-None value triggers the intended code path.
    - Walk through or simulate function calls to confirm that each new branch or change will be executed. Avoid unreachable modifications.

Code Format Guidelines:
1. All `SEARCH` blocks must match the original code exactly.
2. When you need to modify code that is not already inside a `DEEPEVOLVE` block, wrap your changes with `### >>> DEEPEVOLVE-BLOCK-START: <research idea>` and `### <<< DEEPEVOLVE-BLOCK-END` markers.
3. If you are updating code that is already marked by a `DEEPEVOLVE` block, edit only the lines within that block and adjust the existing modification comment to reflect your new change.
4. Do NOT nest one `DEEPEVOLVE` block inside another. Each region you modify should have exactly one pair of start/end markers.
    i.e., AVOID doing the following:
    ```
    ### >>> DEEPEVOLVE-BLOCK-START: first modification
    # First code to be modified
    ### >>> DEEPEVOLVE-BLOCK-START: second modification ! It is not allowed to nest one DEEPEVOLVE block inside another.
    # Second code to be modified
    ### <<< DEEPEVOLVE-BLOCK-END
    ### <<< DEEPEVOLVE-BLOCK-END
    ```
    instead, DO the following:
    ```
    ### >>> DEEPEVOLVE-BLOCK-START: first modification, second modification
    # code that has been modified twice
    ### <<< DEEPEVOLVE-BLOCK-END
    ```

5. Limit your changes to what is strictly necessary. Do not rewrite the entire file.
6. Ensure that all modified code remains correct and consistent, including any function signatures, parameter lists, and calls.
7. Preserve the original code's indentation and formatting. Place the lines of `### >>> DEEPEVOLVE-BLOCK-START: <research idea>` and `### <<< DEEPEVOLVE-BLOCK-END` at the same indentation level as the code they annotate.
"""

DEBUGGER_INSTRUCTIONS = """You are an expert developer and researcher who ensures modified code runs correctly and properly implements research ideas.
Your task is to analyze code, identify any kind of errors, including syntax errors, runtime errors, or logical issues, and verify functionality.
Provide detailed diagnostics and specific fixes when problems are found.
Consider edge cases and ensure the code fully addresses the research requirements.

You MUST use the exact SEARCH/REPLACE diff format. Do NOT use Git diff format. Do NOT use line prefixes like `+`, `-`, or `@@`.

Use this structure exactly:
```
<<<<<<< SEARCH
# Code with error (must match exactly)
=======
# DEBUG: <comment>
# Fixed code here
>>>>>>> REPLACE
```
Example 1 for debugging a syntax error:
```
<<<<<<< SEARCH
def compute_mean(values):
    total = sum(values
    return total / len(values)
=======
def compute_mean(values):
    # DEBUG: missing parenthesis in function call, fixed by adding parenthesis
    total = sum(values)
    return total / len(values)
>>>>>>> REPLACE
```

Use Comments like `# DEBUG: <comment>` to indicate the changes you made when debugging.
"""

INSPIRATION_TEMPLATE = """### Inspiration {inspiration_number}
- Research Idea : {idea}
- Performance: {performance}
- Code changes: {code_changes}
"""

# User message template for diff-based evolution
DIFF_CODE_TEMPLATE = """
User query: {query}
Research problem: {problem}

Inspirations:
{inspirations}

Current idea:
{current_idea}

Evolution history:
{idea_evolution}

Pseudocode:
{pseudocode}

Implementation notes:
{implementation_notes}

Current performance:
{current_performance}

Task:
Improve and debug the code based on the context above using your expertise in optimization and machine learning.

Code (multiple files separated by `# === filename.py ===`):
```{language}
{current_program}
"""

REFLECTION_CONTENT = """
1. Code Correctness
   - Are there any syntax errors or runtime errors?
   - Are there inconsistencies in variable names or logic flow?
   - Are there any new functions used but not been defined or implemented?
   - Avoid hiding missing modules or errors with a bare try/except that simply passes. Handle exceptions with clear warnings or errors.

2. Alignment with Research Idea
   - Does the code accurately implement the stated research idea?
      - Please make sure the changes in the function have actually been implemented in the workflow.
      - Avoid the code parts that suppress errors silently

3. Machine Learning Performance
   - Can compute efficiency be improved with minimal code changes?
   - Are there hyperparameters that could be tuned to boost performance?

4. Other Issues
   - At the end of each code review, provide a short summary of checks performed.
   - Avoid the code parts that suppress errors silently.
   - Are there any other issues you think are important?
"""


DEBUGGER_TEMPLATE = """
Resolve the following error in a multi-file Python codebase.

An error occurred during execution:
```
{error_message}
```

Below is the code that caused the error:
```{language}
{modified_code}
````

The modification was made to implement the idea:
```json
{idea}
```

Your responsibilities:

- Identify and fix the cause of the error in the modified code.
- Ensure that all involved files and components integrate correctly and run without errors.
- Ensure the code modification do not break the research idea.
- Ensure the new code within the `DEEPEVOLVE` block is reachable in the workflow. New code should be executed as new idea but not suppressed by error handling or cheated by None values.
- If necessary, update function inputs or implementations to ensure consistency.
- If the code depends on a library that is not available, use the standard library instead.

Please analyze the error and return the corrected code using `diff` format.
"""

class CoderAgent:
    def __init__(self, developer: str, debugger: str, reasoning_effort: str = 'medium'):
        self.developer = Agent(
            name="Code development agent",
            instructions=CODER_INSTRUCTIONS,
            model=developer,
            model_settings=ModelSettings(reasoning={'effort': reasoning_effort}) if developer in reasoning_models else ModelSettings(),
            output_type=str,
        )
        
        self.debugger = Agent(
            name="Code debugging agent",
            instructions=DEBUGGER_INSTRUCTIONS,
            model=debugger,
            model_settings=ModelSettings(reasoning={'effort': reasoning_effort}) if debugger in reasoning_models else ModelSettings(),
            output_type=str,
        )

        self.query = None
        self.problem_description = None
        self.language = None
        self.trace_id = None
        self.problem_name = 'NA'

    def update_topic(self, query: str, problem_name: str, problem_description: str):
        self.query = query
        self.problem_name = problem_name
        self.problem_description = problem_description

    async def debug(
        self, input_code: str, error_message: str,
    ) -> str:
        trace_id = self.trace_id
        if trace_id is None:
            trace_id = gen_trace_id()
            self.trace_id = trace_id

        with trace(f"DeepEvolve_{self.problem_name}", trace_id=trace_id, disabled=False):
            debugger_input = DEBUGGER_TEMPLATE.format(
                # query=self.query,
                error_message=error_message,
                modified_code=input_code,
                idea=self.idea.model_dump(),
                language=self.language,
            )
            result = await Runner.run(self.debugger, debugger_input)

            logger.info(f"Debugger error message:\n {error_message}")
            logger.info(f"Debugger changes:\n {result.final_output_as(str)}")

            diff_with_text = result.final_output_as(str)
            output_code = apply_diff(input_code, diff_with_text)
            
            try:
                output_code = format_str(output_code, mode=Mode())
            except Exception as e:
                logger.warning(f"Error when formatting code: {e}")
                pass
            return output_code

    async def run(
        self,
        new_idea: IdeaData,
        program: Program,
        inspirations: list[Program],
        trace_id: str = None,
        max_reflection_times: int = 1,
    ) -> str:
        """Run the full code improvement pipeline with research context."""
        if trace_id is None:
            trace_id = gen_trace_id()
        self.trace_id = trace_id
        self.language = program.language
        self.idea = new_idea
        # format new idea
        idea_evolution = program.evolution_history
        if len(idea_evolution) > 0:
            idea_evolution = (
                " -> ".join(
                    [
                        f"[{i}] {idea.description}"
                        for i, idea in enumerate(idea_evolution)
                    ]
                )
                + " -> "
                + new_idea.description
            )
        else:
            idea_evolution = "Initial idea -> " + new_idea.description

        # format inspirations
        inspiration_str = ""
        for idx in range(len(inspirations)):
            performance_str = format_metrics_safe(inspirations[idx].metrics)
            code_changes = parse_evolve_blocks(inspirations[idx].code)
            code_changes_str = ""
            for start_line, end_line, block_content in code_changes:
                code_changes_str += f"Line {start_line} to {end_line}: ```{self.language}\n{block_content}```\n"
            inspiration_str += INSPIRATION_TEMPLATE.format(
                inspiration_number=idx,
                idea=inspirations[idx].idea,
                performance=performance_str,
                code_changes=code_changes_str,
            )
        if inspiration_str == "":
            inspiration_str = "No prior inspirations."

        program_code = program.code
        last_input_list = []
        all_diff_text = []
        all_program_code = []
        
        with trace(f"DeepEvolve_{self.problem_name}", trace_id=trace_id, disabled=False):
            logger.info(f"Starting code development ...")
            for ref_idx in range(max_reflection_times + 1):
                if ref_idx > 0:
                    console.print(
                        f"[bold green] coding reflection: {ref_idx} / {max_reflection_times}[/bold green]"
                    )
                    
                current_performance = format_metrics_safe(program.metrics)
                code_prompt = DIFF_CODE_TEMPLATE.format(
                    query=self.query,
                    problem=self.problem_description,
                    inspirations=inspiration_str,
                    current_idea=new_idea.description,
                    idea_evolution=idea_evolution,
                    pseudocode=new_idea.pseudocode,
                    implementation_notes=new_idea.implementation_notes,
                    language=self.language,
                    current_performance=current_performance,
                    current_program=program_code,
                )

                if ref_idx > 0:
                    code_prompt += f"\n\nGiven the previous diff: ```{self.language}\n{all_diff_text[-1]}```"
                    code_prompt += f"\n\nPlease review the code and reflect on the content below: {REFLECTION_CONTENT}"
                    code_prompt += (
                        f"\n\nPlease provide the new diff to improve the code."
                    )

                code_input = last_input_list + [
                    {"content": code_prompt, "role": "user"}
                ]

                result = await Runner.run(self.developer, input=code_input)
                last_input_list = result.to_input_list()
                diff_with_text = result.final_output_as(str)
                program_code = apply_diff(program_code, diff_with_text)
                
                try:
                    program_code = format_str(program_code, mode=Mode())
                except Exception as e:
                    logger.warning(f"Error when formatting code: {e}")
                    pass

                all_diff_text.append(diff_with_text)
                all_program_code.append(program_code)

            logger.info(f"Completed code development with {max_reflection_times} reflection rounds.")
            return all_diff_text, all_program_code