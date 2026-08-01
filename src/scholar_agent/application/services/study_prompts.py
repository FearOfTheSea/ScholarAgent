"""Prompt construction for local study use cases."""

from scholar_agent.application.dtos.retrieval import DocumentChunk, RetrievedChunk
from scholar_agent.domain.entities.study_session import SourceReference


def answer_question_prompt(question: str, chunks: tuple[RetrievedChunk, ...]) -> str:
    """Create a grounded question-answering prompt."""
    sources = "\n\n".join(_citation_block(chunk) for chunk in chunks)
    return (
        "Answer only from the source excerpts below. If the excerpts do not "
        "support an answer, say that the selected document does not provide "
        "enough information. Cite claims with the supplied [source] labels.\n\n"
        f"Question: {question}\n\nSources:\n{sources}"
    )


def summarize_prompt(source_text: str) -> str:
    """Create a concise grounded-summary prompt."""
    return (
        "Write a concise study summary using only the source text below. "
        "Preserve important definitions, claims, and evidence. Return only JSON "
        'with string field "summary" and a non-empty "citations" array containing '
        "exact supplied chunk IDs.\n\n"
        f"Source text:\n{source_text}"
    )


def combine_summaries_prompt(
    summaries: tuple[str, ...],
    citations: tuple[RetrievedChunk | DocumentChunk | SourceReference, ...] = (),
) -> str:
    """Create a prompt that combines partial document summaries."""
    joined_summaries = "\n\n".join(summaries)
    source_hint = ""
    if citations:
        source_hint = (
            " Cite only these exact chunk IDs and return them in the citations "
            "array:\n" + "\n".join(chunk.chunk_id for chunk in citations) + "\n"
        )
    return (
        "Combine these partial summaries into one coherent study summary. "
        "Do not introduce information that is not present. Return only JSON with "
        "string summary and a citations array of exact supplied chunk IDs.\n\n"
        f"Partial summaries:\n{joined_summaries}\n\n{source_hint}"
    )


def quiz_prompt(source_text: str, question_count: int) -> str:
    """Create a structured quiz-generation prompt."""
    return (
        f"Create exactly {question_count} study questions from the source text.\n"
        'Return ONLY a JSON array of objects with string keys "prompt", "answer", '
        'and "citations". citations must contain exact supplied chunk IDs.\n'
        "Do not include any conversational text or explanation outside the JSON.\n\n"
        "Format example:\n"
        "[\n"
        "  {\n"
        '    "prompt": "Question prompt here",\n'
        '    "answer": "Question answer here",\n'
        '    "citations": ["exact-chunk-id"]\n'
        "  }\n"
        "]\n\n"
        f"Source text:\n{source_text}"
    )


def flashcards_prompt(source_text: str, card_count: int) -> str:
    """Create a structured flashcard-generation prompt."""
    return (
        f"Create exactly {card_count} study flashcards from the source text.\n"
        'Return ONLY a JSON array of objects with string keys "front", "back", '
        'and "citations". citations must contain exact supplied chunk IDs.\n'
        "Do not include any conversational text or explanation outside the JSON.\n\n"
        "Format example:\n"
        "[\n"
        "  {\n"
        '    "front": "Question or term here",\n'
        '    "back": "Answer or definition here",\n'
        '    "citations": ["exact-chunk-id"]\n'
        "  }\n"
        "]\n\n"
        f"Source text:\n{source_text}"
    )


def chunks_to_source_text(
    chunks: tuple[DocumentChunk | RetrievedChunk, ...],
    maximum_length: int | None = 6000,
) -> str:
    """Join chunk content without exceeding a local-model prompt budget."""
    content_parts: list[str] = []
    current_length = 0
    for chunk in chunks:
        page = chunk.page_number if chunk.page_number is not None else "unknown"
        content = f"[{chunk.chunk_id}|page={page}]\n{chunk.content}"
        if maximum_length is None:
            content_parts.append(content)
            current_length += len(content)
            continue
        remaining_length = maximum_length - current_length
        if remaining_length <= 0:
            break
        content_parts.append(content[:remaining_length])
        current_length += len(content_parts[-1])
    return "\n\n".join(content_parts)


def split_source_text(source_text: str, maximum_length: int = 6000) -> tuple[str, ...]:
    """Split source text into prompt-sized segments at paragraph boundaries."""
    paragraphs = tuple(part for part in source_text.split("\n\n") if part.strip())
    segments: list[str] = []
    current_parts: list[str] = []
    current_length = 0
    for paragraph in paragraphs:
        if current_parts and current_length + len(paragraph) > maximum_length:
            segments.append("\n\n".join(current_parts))
            current_parts = []
            current_length = 0
        current_parts.append(paragraph)
        current_length += len(paragraph)
    if current_parts:
        segments.append("\n\n".join(current_parts))
    return tuple(segments)


def split_source_chunks(
    chunks: tuple[DocumentChunk, ...], maximum_length: int = 6000
) -> tuple[tuple[DocumentChunk, ...], ...]:
    """Split source chunks without losing the labels used for citations."""
    segments: list[tuple[DocumentChunk, ...]] = []
    current: list[DocumentChunk] = []
    current_length = 0
    for chunk in chunks:
        page = chunk.page_number if chunk.page_number is not None else "unknown"
        labeled_length = len(f"[{chunk.chunk_id}|page={page}]\n{chunk.content}")
        if current and current_length + labeled_length > maximum_length:
            segments.append(tuple(current))
            current = []
            current_length = 0
        current.append(chunk)
        current_length += labeled_length
    if current:
        segments.append(tuple(current))
    return tuple(segments)


def _citation_block(chunk: RetrievedChunk) -> str:
    page = chunk.page_number if chunk.page_number is not None else "unknown page"
    label = f"{chunk.document_id.value}:p{page}:{chunk.chunk_id}"
    return f"[{label}]\n{chunk.content}"
