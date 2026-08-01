import json
import re

from scholar_agent.application.dtos.retrieval import DocumentChunk, RetrievedChunk
from scholar_agent.domain.entities.study_session import SourceReference


def parse_items(
    raw_output: str, first_key: str, second_key: str
) -> tuple[tuple[str, str], ...]:
    """Parse a model-generated JSON array containing two string fields."""
    start_index = raw_output.find("[")
    end_index = raw_output.rfind("]")
    if start_index == -1 or end_index == -1 or end_index < start_index:
        normalized_output = (
            raw_output.strip().removeprefix("```json").removeprefix("```")
        )
        normalized_output = normalized_output.removesuffix("```").strip()
    else:
        normalized_output = raw_output[start_index : end_index + 1]

    # Clean up trailing commas inside arrays/objects
    normalized_output = re.sub(r",(\s*[\]}])", r"\1", normalized_output)

    try:
        payload = json.loads(normalized_output)
    except json.JSONDecodeError as error:
        raise ValueError("The local model did not return valid JSON.") from error
    if not isinstance(payload, list):
        raise ValueError("The local model must return a JSON array.")

    parsed_items: list[tuple[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each generated item must be a JSON object.")
        first_value = item.get(first_key)
        second_value = item.get(second_key)
        if not isinstance(first_value, str) or not isinstance(second_value, str):
            raise ValueError("Generated item fields must be strings.")
        parsed_items.append((first_value.strip(), second_value.strip()))
    return tuple(parsed_items)


def parse_cited_items(
    raw_output: str,
    first_key: str,
    second_key: str,
    chunks: tuple[DocumentChunk | RetrievedChunk | SourceReference, ...],
) -> tuple[tuple[str, str, tuple[SourceReference, ...]], ...]:
    """Parse generated items and require citations from the supplied chunk set."""
    payload = _json_array(raw_output)
    references_by_id = _references_by_id(chunks)
    parsed: list[tuple[str, str, tuple[SourceReference, ...]]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each generated item must be a JSON object.")
        first_value = item.get(first_key)
        second_value = item.get(second_key)
        if (
            not isinstance(first_value, str)
            or not first_value.strip()
            or not isinstance(second_value, str)
            or not second_value.strip()
        ):
            raise ValueError("Generated item fields must be non-blank strings.")
        citations = _citation_references(item.get("citations"), references_by_id)
        parsed.append((first_value.strip(), second_value.strip(), citations))
    return tuple(parsed)


def parse_cited_summary(
    raw_output: str,
    chunks: tuple[DocumentChunk | RetrievedChunk | SourceReference, ...],
) -> tuple[str, tuple[SourceReference, ...]]:
    """Parse a generated summary and validate every cited chunk identifier."""
    references_by_id = _references_by_id(chunks)
    payload = _json_object(raw_output)
    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("Generated summary must contain non-blank 'summary'.")
    citations = _citation_references(payload.get("citations"), references_by_id)
    return summary.strip(), citations


def parse_explanation(
    raw_output: str,
    chunks: tuple[DocumentChunk | RetrievedChunk | SourceReference, ...],
) -> tuple[str, str, tuple[SourceReference, ...]]:
    """Parse a cited explanation contract."""
    payload = _json_object(raw_output)
    if set(payload) != {"explanation", "check_question", "citations"}:
        raise ValueError(
            "Explanation must contain exactly explanation, check_question, "
            "and citations."
        )
    explanation = _nonblank(payload.get("explanation"), "explanation")
    check_question = _nonblank(payload.get("check_question"), "check_question")
    citations = _citation_references(
        payload.get("citations"), _references_by_id(chunks)
    )
    return explanation, check_question, citations


def parse_assessment(
    raw_output: str,
    chunks: tuple[DocumentChunk | RetrievedChunk | SourceReference, ...],
) -> tuple[int, str, tuple[str, ...], str, tuple[SourceReference, ...]]:
    """Parse a cited learner-assessment contract."""
    payload = _json_object(raw_output)
    required = {"score", "feedback", "missing_concepts", "next_question", "citations"}
    if set(payload) != required:
        raise ValueError(
            "Assessment must contain exactly score, feedback, missing_concepts, "
            "next_question, and citations."
        )
    score = payload.get("score")
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 3:
        raise ValueError("Assessment score must be an integer from 0 to 3.")
    feedback = _nonblank(payload.get("feedback"), "feedback")
    next_question = _nonblank(payload.get("next_question"), "next_question")
    missing = payload.get("missing_concepts")
    if not isinstance(missing, list) or not all(
        isinstance(item, str) and item.strip() for item in missing
    ):
        raise ValueError("missing_concepts must be an array of non-blank strings.")
    citations = _citation_references(
        payload.get("citations"), _references_by_id(chunks)
    )
    return (
        score,
        feedback,
        tuple(item.strip() for item in missing),
        next_question,
        citations,
    )


def references_for_chunks(
    chunks: tuple[DocumentChunk | RetrievedChunk | SourceReference, ...],
) -> tuple[SourceReference, ...]:
    """Convert selected chunks into stable, citation-safe domain references."""
    return tuple(_references_by_id(chunks).values())


def _json_array(raw_output: str) -> list[object]:
    start_index = raw_output.find("[")
    end_index = raw_output.rfind("]")
    if start_index < 0 or end_index < start_index:
        raise ValueError("The local model did not return a JSON array.")
    normalized_output = re.sub(
        r",(\s*[\]}])", r"\1", raw_output[start_index : end_index + 1]
    )
    try:
        payload = json.loads(normalized_output)
    except json.JSONDecodeError as error:
        raise ValueError("The local model did not return valid JSON.") from error
    if not isinstance(payload, list):
        raise ValueError("The local model must return a JSON array.")
    return payload


def _json_object(raw_output: str) -> dict[str, object]:
    start_index = raw_output.find("{")
    end_index = raw_output.rfind("}")
    if start_index < 0 or end_index < start_index:
        raise ValueError("The local model did not return a JSON object.")
    try:
        payload = json.loads(raw_output[start_index : end_index + 1])
    except json.JSONDecodeError as error:
        raise ValueError("The local model did not return valid JSON.") from error
    if not isinstance(payload, dict):
        raise ValueError("The local model must return a JSON object.")
    return {str(key): value for key, value in payload.items()}


def _references_by_id(
    chunks: tuple[DocumentChunk | RetrievedChunk | SourceReference, ...],
) -> dict[str, SourceReference]:
    result: dict[str, SourceReference] = {}
    for chunk in chunks:
        if isinstance(chunk, SourceReference):
            result[chunk.chunk_id] = chunk
            continue
        result[chunk.chunk_id] = SourceReference(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            page_number=chunk.page_number,
            excerpt=chunk.content[:500],
        )
    return result


def _citation_references(
    value: object,
    references_by_id: dict[str, SourceReference],
) -> tuple[SourceReference, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("Every generated item requires at least one citation.")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError("Citations must be an array of non-blank chunk IDs.")
    references: list[SourceReference] = []
    seen: set[str] = set()
    for raw_id in value:
        chunk_id = raw_id.strip()
        if chunk_id in seen:
            raise ValueError(f"Duplicate citation chunk: {chunk_id}.")
        reference = references_by_id.get(chunk_id)
        if reference is None:
            raise ValueError(f"Unknown citation chunk: {chunk_id}.")
        references.append(reference)
        seen.add(chunk_id)
    return tuple(references)


def _nonblank(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field}' must be non-blank text.")
    return value.strip()
