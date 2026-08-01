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


MISSION_CAPABILITIES = (
    StudyToolDefinition(
        task=StudyTask.SEMANTIC_SEARCH,
        description="Search cited evidence in the selected PDF.",
        arguments=(
            ToolArgumentDefinition(
                name="query", kind=ToolArgumentKind.TEXT, required=True
            ),
            ToolArgumentDefinition(
                name="limit",
                kind=ToolArgumentKind.POSITIVE_INTEGER,
                required=False,
                default=4,
            ),
        ),
    ),
    StudyToolDefinition(
        task=StudyTask.SUMMARIZE_DOCUMENT,
        description="Create a cited summary of the selected PDF.",
    ),
    StudyToolDefinition(
        task=StudyTask.GENERATE_QUIZ,
        description="Create cited quiz questions from the selected PDF.",
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
        description="Create cited flashcards from the selected PDF.",
        arguments=(
            ToolArgumentDefinition(
                name="card_count",
                kind=ToolArgumentKind.POSITIVE_INTEGER,
                required=False,
                default=GenerationCountPolicy.FLASHCARDS_DEFAULT,
            ),
        ),
    ),
    StudyToolDefinition(
        task=StudyTask.CITATION_LOOKUP,
        description="Look up one to twenty cited chunks from the selected PDF.",
        arguments=(
            ToolArgumentDefinition(
                name="chunk_ids",
                kind=ToolArgumentKind.STRING_ARRAY,
                required=True,
            ),
        ),
    ),
    StudyToolDefinition(
        task=StudyTask.BUILD_DOCUMENT_MAP,
        description="Build a cited concept map for the selected PDF.",
    ),
    StudyToolDefinition(
        task=StudyTask.EXPLAIN_CONCEPT,
        description="Explain one cited objective and ask a comprehension check.",
        arguments=(
            ToolArgumentDefinition(
                name="objective_id", kind=ToolArgumentKind.TEXT, required=True
            ),
            ToolArgumentDefinition(
                name="source_chunk_ids",
                kind=ToolArgumentKind.STRING_ARRAY,
                required=True,
            ),
            ToolArgumentDefinition(
                name="learner_question", kind=ToolArgumentKind.TEXT, required=False
            ),
            ToolArgumentDefinition(
                name="style",
                kind=ToolArgumentKind.TEXT,
                required=False,
                default="concise",
            ),
        ),
    ),
    StudyToolDefinition(
        task=StudyTask.ASSESS_LEARNER_RESPONSE,
        description="Assess a pending learner response with cited feedback.",
        arguments=(
            ToolArgumentDefinition(
                name="objective_id", kind=ToolArgumentKind.TEXT, required=True
            ),
            ToolArgumentDefinition(
                name="pending_question", kind=ToolArgumentKind.TEXT, required=True
            ),
            ToolArgumentDefinition(
                name="learner_response", kind=ToolArgumentKind.TEXT, required=True
            ),
            ToolArgumentDefinition(
                name="source_chunk_ids",
                kind=ToolArgumentKind.STRING_ARRAY,
                required=True,
            ),
        ),
    ),
)
