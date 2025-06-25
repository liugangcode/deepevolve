from pydantic import BaseModel

# Used in Researcher

reasoning_models = ["o4-mini", "o3-mini", "o1-mini", "o1", "o3", "o1-pro"]

class ResearchWork(BaseModel):
    title: str
    "The title of the research paper."

    link: str
    "The link to the research paper."

    contributions: list[str]
    "A list of contributions of the research paper."

    limitations: list[str]
    "A list of limitations of the research paper."


class EvaluationData(BaseModel):
    score: int
    "The score of the idea between 0 and 10. Higher is better."

    positive: str
    "A positive reason for the evaluation."

    negative: str
    "A negative reason for the evaluation."


class IdeaData(BaseModel):
    description: str
    "One or two sentences describing the new idea including (1) the problem the idea solves, (2) how the idea solves it, and (3) what makes the idea new."

    motivation: str
    "The motivation for the new idea on why it is different from existing methods and why it can improve the existing methods for the target problem."

    implementation_notes: str
    "Notes on how to implement the new idea (e.g. pseudocode, logic, etc.)."

    pseudocode: str
    "A pseudocode implementation of the new idea if available."

    originality: EvaluationData
    "Self-assessment of the originality of the new idea."

    future_potential: EvaluationData
    "Self-assessment of the future potential of the new idea."

    code_difficulty: EvaluationData
    "Self-assessment of the difficulty of implementing the new idea."


class ReportData(BaseModel):
    markdown_report: str 
    """The final report"""

    idea: IdeaData 
    """The new idea from the research report."""

    related_work: list[ResearchWork] 
    """A list of existing research works that are relevant to the query."""

class WebSearchItem(BaseModel):
    reason: str
    "Your reasoning for why this search is important to the query."

    query: str
    "The search term to use for the web search."


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem]
    """A list of web searches to perform to best answer the query."""


class ReflectionPlan(BaseModel):
    is_sufficient: bool
    "Whether the report is sufficient to answer the query."

    knowledge_gaps: list[str]
    "The information that the report lacks. If is_sufficient is true, this should be empty."

    follow_up_queries: list[WebSearchItem]
    "A list of follow-up queries to perform to best answer the query. If is_sufficient is true, this should be empty."