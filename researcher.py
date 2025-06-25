from __future__ import annotations

import asyncio
import logging
from rich.console import Console
from datetime import datetime

from agents import Agent, WebSearchTool, Runner
from agents.tracing import gen_trace_id, trace, custom_span
from agents.model_settings import ModelSettings

from database import Program
from utils.datatypes import (
    ReportData,
    IdeaData,
    WebSearchPlan,
    WebSearchItem,
    ReflectionPlan,
    reasoning_models,
)
from utils.format import format_metrics_safe

logger = logging.getLogger(__name__)

console = Console()

INSPIRATION_TEMPLATE = """### Inspiration {inspiration_number}
- Research Idea : {idea}
- Performance: {performance}
"""

PLANNER_INSTRUCTIONS = """You are a professor responsible for planning deep and effective research strategies. 

You will be provided with the context of:
 - a research problem based on an initial research question
 - a starting research idea, possibly with a history showing how idea evolves through previous attempt
 - inspirations from earlier attempts

Your task is to develop search queries that identify directions for researchers to advance the idea in a transformative way. 
Rather than combining existing inspirations in small increments, the queries should guide researchers toward substantial evolutions. 
Because other researchers will rely on this plan, it must emphasize major, novel approaches instead of minor refinements.

You will also be told whether the research progress is early or mature:
- If the progress is early, focus on ideas that are feasible and practical, and can grow later and have great future potential.
- If the progress is mature, focus on bold, high-impact shifts that challenge the current approach.

Your plan should follow two steps:
1. Formulate 5 to 10 precise and diverse search queries. Make sure the queries are diverse—cover different perspectives, challenge untested assumptions, and explore alternative methods. 
2. For each query, include a short note explaining why you chose it and what you hope it will reveal.
"""

REFLECTION_INSTRUCTIONS = """
You are an expert research assistant. You will receive a research report (in Markdown) and a newly proposed idea for that report's research problem. Your job is to identify any gaps or issues—such as missing details, logical flaws, or questionable evaluations of novelty, impact, or implementation difficulty.  

- If the report and idea contain all necessary information, do not generate any follow-up questions.  
- If you detect a knowledge gap or something that needs deeper exploration, generate one or more self-contained follow-up queries. Each query must include enough context so that a web search could answer it, For each query, give a short note explaining why you use the query and what you hope it will reveal.
- Focus on technical details, implementation specifics, and any emerging methods or references that were overlooked.  
- Use clear, direct language and avoid unnecessary jargon.  

"""

SEARCH_INSTRUCTIONS = (
    "You are a research assistant. Given a search term, you search the web for that term and "
    "produce a concise summary of the results. The summary must be 2-3 paragraphs and less than 300 "
    "words. Capture the main points. Write succinctly, no need to have complete sentences or good "
    "grammar. This will be consumed by someone synthesizing a report for a new idea, so its vital you capture the "
    "essence and ignore any fluff. Do not include any additional commentary other than the summary "
    "itself."
)

WRITER_INSTRUCTIONS = """You are a senior researcher responsible for proposing new ideas to address a defined research problem. You will receive:
- The research problem, including its evaluation metric and available data
- A starting research idea, possibly with its evolution history
- Inspirations from earlier attempts
- A list of related online search results
- A research progress score (0-100%) indicating how far the idea has advanced

Your goal is to identify future research directions that address the target problem, using the starting point, prior attempts, and related works. You should analyze existing methods, identify connections, and propose practical algorithms that can be implemented with the available data.

Follow this structure to think and write:

1. **Extract insights**  
   Identify 3-5 scientific insights from the starting point and 3-5 from related works. For each insight, explain in 2-3 sentences how it relates to the target problem.

2. **Organize research directions**  
   Group the insights into 3-5 coherent directions (for example, learning objectives, model classes, or optimization methods).

3. **Build a structured framework**  
   Create a conceptual map (such as a taxonomy, grid, or matrix) that unifies existing methods, reveals patterns, and highlights gaps.

4. **Generate and evaluate ideas**  
   - First, propose 3-10 algorithmic ideas of varying originality and complexity. Each idea should be:
     - As simple, minimal, and atomic as possible but not trivial  
     - Include brief pseudocode or logical steps where helpful.  
     - Include references to the related works.
   - For each idea, critically assess as a senior researcher with one positive and one negative reason:
     - Originality (0-10): Is the idea new? Is the idea a novel combination of well-known techniques? Is it clearly different from previous contributions?
     - Future Potential (0-10): Will others build on these ideas? Does this idea solve a hard problem more effectively than prior work? Does it point to a new research direction?
     - Code Difficulty (0-10): How complex is the implementation? How much code is required? How much time is required to implement?
   - Then, select the single best idea from that list for detailed reporting, based on the research progress score:
     - If progress is relatively early, prioritize feasible, easy-to-implement ideas with long-term promise.
     - If progress is relatively mature, prioritize seminal ideas with high-impacts for the next-generation research
     - Otherwise, balance ambition and implementation feasibility

5. **Write the report in Markdown**  
   For the selected idea, include:
   - A synthesis of insights and proposed directions  
   - The structured framework of existing methods and the new algorithm  
   - A list of new ideas with their assessment scores
   - Detailed description of the chosen/best idea, including rationale, pseudocode, and implementation notes

The report must be focused, technically accurate. Being concise with 200-500 words without trivial and redundant information.
Support all claims with evidence and references, and remain tightly aligned with the target problem.
"""


USER_TEMPLATE = """
## User Query
{query}

## Research Problem
{problem}

## Starting Research Idea
{starting_point}

## Idea Evolution History
{idea_evolution}

## Research Progress
{evolution_progress}

## Previous Inspirations
{inspirations}
"""

PAPER_READER_INSTRUCTIONS = """
You are a paper reader. You will be provided with a title of the idea with the content.

If the content is an online link, your task is to search the paper online and summarize the core ideas of the paper.

If the content is the description of the idea, your task is to read the description and summarize the core ideas of the idea.

You may be provided supplmentary information about the idea, such as the code, the implementation notes, the pseudocode, etc.
"""

REFLECTION_CONTENT = """
- Should we consider other ideas in the report or a totally new idea?
- Are the ratings for originality, future potential, and code difficulty accurate?
- Are there any logical inconsistencies or gaps in the methodology?
- Are any implementation steps or references missing?
- Is every step described clearly enough to reproduce results?
- Does the idea suffer from overfitting or shortcut learning?
- Are there any other issues you think are important about the new idea?
"""


class ResearcherAgent:
    def __init__(
        self,
        planner: str = "o3-mini",
        searcher: str = "gpt-4o",
        writer: str = "o3-mini",
        reasoning_effort: str = 'medium',
    ):
        self.planner_agent = Agent(
            name="Planner Agent",
            instructions=PLANNER_INSTRUCTIONS,
            model=planner,
            output_type=WebSearchPlan,
            model_settings=ModelSettings(reasoning={'effort': reasoning_effort}) if planner in reasoning_models else ModelSettings(),
        )
        self.reflection_agent = Agent(
            name="Reflection Agent",
            instructions=REFLECTION_INSTRUCTIONS,
            model=planner,
            output_type=ReflectionPlan,
            model_settings=ModelSettings(reasoning={'effort': reasoning_effort}) if planner in reasoning_models else ModelSettings(),
        )
        self.search_agent = Agent(
            name="Search Agent",
            instructions=SEARCH_INSTRUCTIONS,
            tools=[WebSearchTool()],
            model=searcher,
            model_settings=ModelSettings(reasoning={'effort': reasoning_effort}, tool_choice="required") if searcher in reasoning_models else ModelSettings(tool_choice="required"),
        )
        self.writer_agent = Agent(
            name="Writing Agent",
            instructions=WRITER_INSTRUCTIONS,
            model=writer,
            output_type=ReportData,
            model_settings=ModelSettings(reasoning={'effort': reasoning_effort}) if writer in reasoning_models else ModelSettings(),
        )
        self.reader_agent = Agent(
            name="Paper Reader Agent",
            instructions=PAPER_READER_INSTRUCTIONS,
            tools=[WebSearchTool()],
            model=searcher,
            output_type=IdeaData,
            model_settings=ModelSettings(reasoning={'effort': reasoning_effort}) if searcher in reasoning_models else ModelSettings(),
        )
        self.search_time_bias = False
        self.problem_name = 'NA'

    def update_topic(
        self, query: str, problem_name: str, problem_description: str, search_time_bias: bool = False
    ):
        self.query = query
        self.problem_name = problem_name
        self.problem_description = problem_description
        self.search_time_bias = search_time_bias

    async def read_paper(self, title: str, content: str, supplementary_info: str = None) -> IdeaData:
        query = f"title: {title} \ncontent: {content}"
        if supplementary_info is not None:
            query += f"\n supplementary_info: {supplementary_info}"
        result = await Runner.run(
            self.reader_agent,
            query,
        )
        return result.final_output_as(IdeaData)

    async def run(
        self,
        program: Program,
        inspirations: list[Program],
        trace_id: str = None,
        max_reflection_times: int = 1,
        max_generations: int = 10,
    ) -> tuple[str, list, list, str]:
        """
        Execute the research process from planning to report generation.

        Args:
            query: The research question to investigate
            idea_evolution: Evolution history of the idea
            evolution_progress: Current evolution progress/research stage
            trace_id: Optional trace identifier for logging

        Returns:
            Tuple containing report, related work, new ideas, and structured framework
        """
        idea_evolution = program.evolution_history
        evolution_progress = (
            len(program.evolution_history) / max_generations * 100
        )
        evolution_progress = f"{evolution_progress:.2f}%"
        if len(idea_evolution) > 0:
            idea_evolution = " -> ".join(
                [f"[{i}] {idea.description}" for i, idea in enumerate(idea_evolution)]
            )
        else:
            idea_evolution = "Initial idea"

        inspiration_str = ""
        for idx in range(len(inspirations)):
            performance_str = format_metrics_safe(inspirations[idx].metrics)
            inspiration_str += INSPIRATION_TEMPLATE.format(
                inspiration_number=idx,
                idea=inspirations[idx].idea,
                performance=performance_str,
            )
        if inspiration_str == "":
            inspiration_str = "No prior inspirations."

        if trace_id is None:
            trace_id = gen_trace_id()
        logger.info(f"Starting deep research with trace_id: {trace_id}")

        user_input = USER_TEMPLATE.format(
            query=self.query,
            problem=self.problem_description,
            starting_point=program.idea.description,
            idea_evolution=idea_evolution,
            evolution_progress=evolution_progress,
            inspirations=inspiration_str,
        )

        # console.print("[bold blue]User Input of the Researcher Agent[/bold blue]")
        # console.print(user_input)
        # console.print()

        last_input = None
        all_search_plans = []
        all_search_results = []
        all_reports = []
        with trace(
            f"DeepEvolve_{self.problem_name}",
            metadata={"query": self.query},
            trace_id=trace_id,
            disabled=False,
        ):
            logger.info(f"Performing Deep Research ...")
            for ref_idx in range(max_reflection_times + 1):

                if ref_idx == 0 or last_input is None:
                    search_plan = await self._plan_searches(user_input)
                    all_search_plans.append(search_plan)
                else:
                    reflection_result = await self._reflection(user_input, last_input)
                    if reflection_result.is_sufficient:
                        break
                    else:
                        console.print(
                            f"[bold red]Reflection {ref_idx}: current report is not sufficient because {reflection_result.knowledge_gaps}, generating follow-up queries[/bold red]"
                        )
                        search_plan = WebSearchPlan(
                            searches=reflection_result.follow_up_queries
                        )
                        all_search_plans.append(search_plan)

                search_results = await self._perform_searches(search_plan)
                all_search_results.append(search_results)
                report_result, last_input = await self._write_report(
                    user_input, search_results, last_input=last_input
                )
                all_reports.append(report_result)

        logger.info("Research completed successfully")
        return all_search_plans, all_search_results, all_reports

    async def _plan_searches(self, user_input: str) -> WebSearchPlan:
        logger.info(f"Starting search planning for query: {self.query} ...")

        if self.search_time_bias:
            today = datetime.now().strftime("%Y-%m-%d")
            user_input += f"\n*Important: Today's date is {today}. Prioritize recent search results.*\n"

        result = await Runner.run(
            self.planner_agent,
            user_input,
        )

        logger.info(
            f"Completed search planning: {len(result.final_output.searches)} searches identified"
        )
        return result.final_output_as(WebSearchPlan)

    async def _reflection(self, user_input: str, last_input: list) -> WebSearchPlan:
        new_content = f"""
        Given the following user input, please identify any issues or gaps in the research report:
        {user_input}

        Here are the reflection points you should check about the new idea:
        {REFLECTION_CONTENT}

        If you think the new idea is good enough, do not ask any follow-up questions. Otherwise, write one or more follow-up queries that include relevant context for further investigation.
        """

        reflection_input = last_input + [{"role": "user", "content": new_content}]
        
        try:
            reflection_plan = await Runner.run(
            self.reflection_agent,
                reflection_input,
            )
            return reflection_plan.final_output_as(ReflectionPlan)

        except Exception as e:
            console.print(f"[bold red]Error in reflection: {e}[/bold red]")
            console.print(f"[bold red]Reflection input: {reflection_input}[/bold red]")
            raise e
        
    async def _perform_searches(self, search_plan: WebSearchPlan) -> list[str]:
        with custom_span("Search the web"):
            logger.info(
                f"Starting web searches, total: {len(search_plan.searches)} ..."
            )
            num_completed = 0
            tasks = [
                asyncio.create_task(self._search(item, i + 1))
                for i, item in enumerate(search_plan.searches)
            ]
            results = []
            for task in asyncio.as_completed(tasks):
                result = await task
                if result is not None:
                    results.append(result)
                num_completed += 1
            logger.info(
                f"Completed {len(results)}/{len(search_plan.searches)} searches successfully"
            )
            return results

    async def _search(self, item: WebSearchItem, source_id: int) -> str | None:
        input = f"Search term: {item.query}\nReason for searching: {item.reason}"
        try:
            result = await Runner.run(
                self.search_agent,
                input,
            )
            return str(result.final_output)
        except Exception:
            return None

    async def _write_report(
        self, user_input: str, search_results: list[str], last_input: list = None
    ) -> ReportData:
        logger.info("Starting report writing ...")

        summaries_block = "\n\n---\n\n".join(search_results)

        if last_input is not None:
            new_content = f"""
            Please review and reflect on the report and the new idea based on below reflection points:
            {REFLECTION_CONTENT}

            and more search results on these reflection points:
            {summaries_block}
            
            You can revise the current idea, add new ones, or select a different top idea.
            Important: Edit only within the existing report. Keep its full structure and format unchanged.
            Do not add introductory phrases like "In reviewing the report and the proposed idea, several reflections arise..."
            Retain every detail; focus on strengthening the report, not generating a new report or a reflection document.
            """
            user_input = last_input + [{"content": new_content, "role": "user"}]
        else:
            user_input += f"\n\n ## Search results\n{summaries_block}"

        result = await Runner.run(
            self.writer_agent,
            user_input,
        )
        
        logger.info("Completed report writing")
        return result.final_output_as(ReportData), result.to_input_list()