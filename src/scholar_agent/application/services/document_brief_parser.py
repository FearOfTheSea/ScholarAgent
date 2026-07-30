"""Strict parsing for cited document briefs."""

import json
from collections.abc import Mapping

from scholar_agent.application.dtos.retrieval import DocumentChunk
from scholar_agent.domain.entities.study_session import (
    ConceptNode,
    DocumentBrief,
    GlossaryTerm,
    LearningObjective,
    SourceReference,
)
from scholar_agent.domain.value_objects.document_id import DocumentId


def document_brief_prompt(
    document_id: DocumentId,
    chunks: tuple[DocumentChunk, ...],
) -> str:
    """Build the structured document-analysis prompt."""
    sources = "\n\n".join(
        f"[{chunk.chunk_id}|page={chunk.page_number}]\n{chunk.content}"
        for chunk in _representative_chunks(chunks)
    )
    contract = (
        "Return only a JSON object—no Markdown or commentary—with exactly this "
        "shape: "
        '{"synopsis":"...","objectives":[{"id":"objective-1","title":"...",'
        '"description":"...","prerequisites":[],"citations":["exact-chunk-id"]}],'
        '"concepts":[{"id":"concept-1","label":"...","explanation":"...",'
        '"prerequisites":[],"citations":["exact-chunk-id"]}],'
        '"glossary":[{"term":"...","definition":"...",'
        '"citations":["exact-chunk-id"]}],"misconceptions":["..."]}. '
        "Produce exactly 3 objectives, 4 concepts, 3 glossary items, and 2 "
        "misconceptions. Keep the synopsis under 60 words and every other text "
        "field under 25 words. Use exactly one citation per item. "
        "Prerequisites may reference only earlier IDs in the same list. Every item "
        "requires at least one exact supplied chunk ID. Use only source content."
    )
    return (
        "Analyze the supplied excerpts from one study document. Do not use outside "
        "knowledge.\n\n"
        f"DOCUMENT_ID: {document_id.value}\n\nSOURCES:\n{sources}\n\n"
        f"OUTPUT CONTRACT (MANDATORY):\n{contract}"
    )


def document_brief_repair_prompt(
    original_prompt: str,
    raw_output: str,
    error: str,
) -> str:
    """Request one bounded repair for an invalid brief."""
    return (
        f"{original_prompt}\n\nINVALID RESPONSE:\n{raw_output}\n\n"
        f"VALIDATION ERROR: {error}\n"
        "Repair the response now. Follow the OUTPUT CONTRACT exactly. Return only "
        "the JSON object, with no code fence, Markdown, explanation, or extra keys."
    )


def parse_document_brief(
    raw_output: str,
    document_id: DocumentId,
    chunks: tuple[DocumentChunk, ...],
) -> DocumentBrief:
    """Parse and validate one cited document brief."""
    payload = _json_object(raw_output)
    nested = payload.get("document_brief")
    if isinstance(nested, dict):
        payload = {str(key): value for key, value in nested.items()}
    required = {"synopsis", "objectives", "concepts", "glossary", "misconceptions"}
    missing = required - set(payload)
    if missing:
        names = ", ".join(sorted(missing))
        received = ", ".join(sorted(payload)) or "none"
        raise ValueError(
            f"Document brief is missing required fields: {names}. "
            f"Received fields: {received}."
        )
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    synopsis = _text(payload.get("synopsis"), "synopsis")
    objectives = _objectives(payload.get("objectives"), document_id, chunks_by_id)
    concepts = _concepts(payload.get("concepts"), document_id, chunks_by_id)
    glossary = _glossary(payload.get("glossary"), document_id, chunks_by_id)
    misconceptions = _misconceptions(payload.get("misconceptions"))
    if not objectives or not concepts:
        raise ValueError("Document brief requires objectives and concepts.")
    return DocumentBrief(
        document_id=document_id,
        synopsis=synopsis,
        objectives=objectives,
        concepts=concepts,
        glossary=glossary,
        misconceptions=misconceptions,
    )


def _representative_chunks(
    chunks: tuple[DocumentChunk, ...],
    maximum_characters: int = 6500,
) -> tuple[DocumentChunk, ...]:
    if sum(len(chunk.content) for chunk in chunks) <= maximum_characters:
        return chunks
    selected: list[DocumentChunk] = []
    step = max(1, len(chunks) // 12)
    size = 0
    for chunk in chunks[::step]:
        if size + len(chunk.content) > maximum_characters:
            break
        selected.append(chunk)
        size += len(chunk.content)
    return tuple(selected)


def _json_object(raw_output: str) -> dict[str, object]:
    start = raw_output.find("{")
    end = raw_output.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Response does not contain a JSON object.")
    try:
        payload = json.loads(raw_output[start : end + 1])
    except json.JSONDecodeError as error:
        raise ValueError("Response is not valid JSON.") from error
    if not isinstance(payload, dict):
        raise ValueError("Response must be a JSON object.")
    return {str(key): value for key, value in payload.items()}


def _objectives(
    value: object,
    document_id: DocumentId,
    chunks: Mapping[str, DocumentChunk],
) -> tuple[LearningObjective, ...]:
    items = _object_list(value, "objectives")
    results: list[LearningObjective] = []
    seen: set[str] = set()
    for item in items:
        identifier = _text(item.get("id"), "objective id")
        if identifier in seen:
            raise ValueError(f"Duplicate objective ID: {identifier}.")
        prerequisites = _text_list(item.get("prerequisites"), "prerequisites")
        if any(prerequisite not in seen for prerequisite in prerequisites):
            raise ValueError(
                "Objective prerequisites must reference earlier objectives."
            )
        results.append(
            LearningObjective(
                identifier=identifier,
                title=_text(item.get("title"), "objective title"),
                description=_text(item.get("description"), "objective description"),
                prerequisite_ids=prerequisites,
                citations=_references(
                    item.get("citations"), document_id, chunks, "objective"
                ),
            )
        )
        seen.add(identifier)
    return tuple(results)


def _concepts(
    value: object,
    document_id: DocumentId,
    chunks: Mapping[str, DocumentChunk],
) -> tuple[ConceptNode, ...]:
    items = _object_list(value, "concepts")
    results: list[ConceptNode] = []
    seen: set[str] = set()
    for item in items:
        identifier = _text(item.get("id"), "concept id")
        if identifier in seen:
            raise ValueError(f"Duplicate concept ID: {identifier}.")
        prerequisites = _text_list(item.get("prerequisites"), "prerequisites")
        if any(prerequisite not in seen for prerequisite in prerequisites):
            raise ValueError("Concept prerequisites must reference earlier concepts.")
        results.append(
            ConceptNode(
                identifier=identifier,
                label=_text(item.get("label"), "concept label"),
                explanation=_text(item.get("explanation"), "concept explanation"),
                prerequisite_ids=prerequisites,
                citations=_references(
                    item.get("citations"), document_id, chunks, "concept"
                ),
            )
        )
        seen.add(identifier)
    return tuple(results)


def _glossary(
    value: object,
    document_id: DocumentId,
    chunks: Mapping[str, DocumentChunk],
) -> tuple[GlossaryTerm, ...]:
    return tuple(
        GlossaryTerm(
            term=_text(item.get("term"), "glossary term"),
            definition=_text(item.get("definition"), "glossary definition"),
            citations=_references(
                item.get("citations"), document_id, chunks, "glossary item"
            ),
        )
        for item in _object_list(value, "glossary")
    )


def _references(
    value: object,
    document_id: DocumentId,
    chunks: Mapping[str, DocumentChunk],
    field: str,
) -> tuple[SourceReference, ...]:
    chunk_ids = _text_list(value, f"{field} citations")
    if not chunk_ids:
        raise ValueError(f"Every {field} requires a citation.")
    references: list[SourceReference] = []
    for chunk_id in chunk_ids:
        chunk = chunks.get(chunk_id)
        if chunk is None:
            raise ValueError(f"Unknown citation chunk: {chunk_id}.")
        references.append(
            SourceReference(
                document_id=document_id,
                chunk_id=chunk.chunk_id,
                page_number=chunk.page_number,
                excerpt=chunk.content[:500],
            )
        )
    return tuple(references)


def _object_list(value: object, field: str) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"'{field}' must be an array of objects.")
    return tuple(
        {str(key): item_value for key, item_value in item.items()} for item in value
    )


def _text_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"'{field}' must be an array of non-blank strings.")
    return tuple(item.strip() for item in value)


def _misconceptions(value: object) -> tuple[str, ...]:
    """Accept concise strings or a small model's labeled misconception objects."""
    if not isinstance(value, list):
        raise ValueError("'misconceptions' must be an array.")
    results: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            results.append(item.strip())
            continue
        if isinstance(item, dict):
            term = item.get("term")
            definition = item.get("definition")
            if (
                isinstance(term, str)
                and term.strip()
                and isinstance(definition, str)
                and definition.strip()
            ):
                results.append(f"{term.strip()}: {definition.strip()}")
                continue
        raise ValueError("Every misconception must contain non-blank text.")
    return tuple(results)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field}' must be non-blank text.")
    return value.strip()
