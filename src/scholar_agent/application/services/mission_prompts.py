"""Prompt contracts for the two learner-facing mission capabilities."""

from scholar_agent.application.dtos.retrieval import DocumentChunk


def explain_concept_prompt(
    objective_id: str,
    learner_question: str | None,
    style: str,
    source_text: str,
) -> str:
    """Build the strict explanation prompt."""
    question = learner_question or "No additional learner question was supplied."
    return (
        "Explain one learning objective using only the supplied excerpts. Return "
        "only JSON with exactly explanation, check_question, and citations. "
        "citations must contain exact supplied chunk IDs. Do not reveal an answer "
        "to the check question.\n"
        f"OBJECTIVE_ID: {objective_id}\nSTYLE: {style}\n"
        f"LEARNER_QUESTION: {question}\nSOURCES:\n{source_text}"
    )


def assess_response_prompt(
    objective_id: str,
    pending_question: str,
    learner_response: str,
    source_text: str,
) -> str:
    """Build the strict learner-assessment prompt."""
    return (
        "Assess the learner response using only the supplied excerpts. Return only "
        "JSON with exactly score, feedback, missing_concepts, next_question, and "
        "citations. score must be 0, 1, 2, or 3. citations must contain exact "
        "supplied chunk IDs. Do not reveal the answer to next_question.\n"
        f"OBJECTIVE_ID: {objective_id}\nPENDING_QUESTION: {pending_question}\n"
        f"LEARNER_RESPONSE: {learner_response}\nSOURCES:\n{source_text}"
    )


def chunks_to_mission_source_text(chunks: tuple[DocumentChunk, ...]) -> str:
    """Render evidence with the exact identifiers accepted by the parser."""
    return "\n\n".join(
        f"[{chunk.chunk_id}|page={chunk.page_number}]\n{chunk.content}"
        for chunk in chunks
    )
