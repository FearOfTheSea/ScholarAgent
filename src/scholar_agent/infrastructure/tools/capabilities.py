"""Explicit capability catalog exposed to the study-agent planner."""

from scholar_agent.application.dtos.agent import StudyTask
from scholar_agent.application.output_ports.tool_executor import (
    StudyToolDefinition,
    ToolArgumentDefinition,
    ToolArgumentKind,
)
from scholar_agent.application.services.generation_count_policy import (
    GenerationCountPolicy,
)

STUDY_CAPABILITIES = (
    StudyToolDefinition(
        task=StudyTask.ANSWER_QUESTION,
        description="Answer a question using cited evidence from the selected PDF.",
        arguments=(
            ToolArgumentDefinition(
                name="question",
                kind=ToolArgumentKind.TEXT,
                required=True,
            ),
        ),
    ),
    StudyToolDefinition(
        task=StudyTask.SUMMARIZE_DOCUMENT,
        description="Summarize the selected PDF.",
    ),
    StudyToolDefinition(
        task=StudyTask.GENERATE_QUIZ,
        description="Generate a quiz from the selected PDF.",
        arguments=(
            ToolArgumentDefinition(
                name="question_count",
                kind=ToolArgumentKind.POSITIVE_INTEGER,
                required=False,
                default=GenerationCountPolicy.QUIZ_DEFAULT,
            ),
        ),
    ),
    StudyToolDefinition(
        task=StudyTask.GENERATE_FLASHCARDS,
        description="Generate flashcards from the selected PDF.",
        arguments=(
            ToolArgumentDefinition(
                name="card_count",
                kind=ToolArgumentKind.POSITIVE_INTEGER,
                required=False,
                default=GenerationCountPolicy.FLASHCARDS_DEFAULT,
            ),
        ),
    ),
)
