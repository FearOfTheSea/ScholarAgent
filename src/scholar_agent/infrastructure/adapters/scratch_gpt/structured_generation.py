"""Deterministic structured decoding for contracts too strict for a tiny GPT."""

import json
import re


def structured_response(prompt: str) -> str | None:
    """Return a contract-safe response for recognized structured prompts."""
    if prompt.startswith("Choose a bounded study mission"):
        return _mission_plan_response(prompt)
    if prompt.startswith("Write a concise study summary"):
        return _summary_response(prompt)
    if prompt.startswith("Combine these partial summaries"):
        return _combined_summary_response(prompt)
    if prompt.startswith("Explain one learning objective"):
        return _explanation_response(prompt)
    if prompt.startswith("You are the constrained planner"):
        return _planner_response(prompt)
    if prompt.startswith("Create exactly") and '"prompt"' in prompt:
        return _quiz_response(prompt)
    if prompt.startswith("Create exactly") and '"front"' in prompt:
        return _flashcard_response(prompt)
    if "OUTPUT CONTRACT (MANDATORY):" in prompt and "DOCUMENT_ID:" in prompt:
        return _document_brief_response(prompt)
    if prompt.startswith("Assess the learner response using only the "):
        return _assessment_response(prompt)
    if prompt.startswith("Verify whether every factual claim in RESPONSE"):
        return _verification_response(prompt)
    return None


def _planner_response(prompt: str) -> str:
    request = prompt.rsplit("LEARNER REQUEST:", 1)[-1].strip()
    lowered = request.casefold()
    if any(
        phrase in lowered
        for phrase in ("compare", "another document", "other pdf", "web", "internet")
    ):
        return json.dumps(
            {
                "actions": [],
                "message": (
                    "This local study agent supports one selected document at a time."
                ),
            }
        )

    requested: list[tuple[int, dict[str, object]]] = []
    if _requests_answer(request):
        requested.append(
            (
                _position(lowered, ("answer", "what", "why", "how")),
                {
                    "tool_name": "answer_question",
                    "arguments": {"question": _question_text(request)},
                },
            )
        )
    if any(word in lowered for word in ("summarize", "summary", "overview")):
        requested.append(
            (
                _position(lowered, ("summarize", "summary", "overview")),
                {"tool_name": "summarize_document", "arguments": {}},
            )
        )
    if any(word in lowered for word in ("quiz", "question")) and not (
        request.rstrip().endswith("?") and "quiz" not in lowered
    ):
        count = _requested_count(
            lowered,
            (
                r"(\d+)\s*[- ]?question",
                r"(\d+)\s*quiz",
            ),
        )
        arguments = {"question_count": count} if count is not None else {}
        requested.append(
            (
                _position(lowered, ("quiz", "questions")),
                {"tool_name": "generate_quiz", "arguments": arguments},
            )
        )
    if any(word in lowered for word in ("flashcard", "study card", "cards")):
        count = _requested_count(
            lowered,
            (
                r"(\d+)\s*flashcard",
                r"(\d+)\s*study card",
                r"(\d+)\s*card",
            ),
        )
        arguments = {"card_count": count} if count is not None else {}
        requested.append(
            (
                _position(lowered, ("flashcard", "study card", "cards")),
                {"tool_name": "generate_flashcards", "arguments": arguments},
            )
        )
    if not requested and any(
        phrase in lowered for phrase in ("prepare me", "study plan", "exam")
    ):
        requested = [
            (0, {"tool_name": "summarize_document", "arguments": {}}),
            (1, {"tool_name": "generate_flashcards", "arguments": {}}),
            (2, {"tool_name": "generate_quiz", "arguments": {}}),
        ]
    actions = [action for _, action in sorted(requested, key=lambda item: item[0])]
    message = None
    if not actions:
        message = (
            "Ask for a cited answer, summary, quiz, or flashcards from this document."
        )
    return json.dumps({"actions": actions, "message": message})


def _mission_plan_response(prompt: str) -> str:
    objectives_text = prompt.split("OBJECTIVES:", 1)[-1].strip()
    try:
        objectives = json.loads(objectives_text)
    except json.JSONDecodeError as error:
        raise ValueError("Mission prompt contains invalid objective JSON.") from error
    if not isinstance(objectives, list) or not objectives:
        raise ValueError("Mission prompt contains no objectives.")
    first = objectives[0]
    if not isinstance(first, dict) or not isinstance(first.get("id"), str):
        raise ValueError("Mission prompt contains an invalid objective.")
    return json.dumps(
        {
            "focus": (
                "Build a prerequisite-first understanding of the selected document."
            ),
            "objective_ids": [first["id"]],
        }
    )


def _summary_response(prompt: str) -> str:
    source = prompt.split("Source text:", 1)[-1]
    return json.dumps(
        {
            "summary": " ".join(_source_sentences(prompt)[:5]),
            "citations": _source_chunk_ids(source),
        }
    )


def _combined_summary_response(prompt: str) -> str:
    summaries = prompt.split("Partial summaries:", 1)[-1].split(
        "Cite only these exact chunk IDs", 1
    )[0]
    citation_text = prompt.split("Cite only these exact chunk IDs", 1)[-1]
    citation_text = citation_text.split("array:", 1)[-1]
    return json.dumps(
        {
            "summary": " ".join(_sentences(summaries)[:5]),
            "citations": _source_chunk_ids(citation_text),
        }
    )


def _explanation_response(prompt: str) -> str:
    objective = _between(prompt, "OBJECTIVE_ID:", "\nSTYLE:")
    source = prompt.split("SOURCES:", 1)[-1]
    sentences = _sentences(source)
    return json.dumps(
        {
            "explanation": " ".join(sentences[:3]),
            "check_question": f"How would you explain {objective} in your own words?",
            "citations": _source_chunk_ids(source),
        }
    )


def _quiz_response(prompt: str) -> str:
    count = _leading_count(prompt)
    sentences = _source_sentences(prompt)
    citations = _source_chunk_ids(prompt.split("Source text:", 1)[-1])
    return json.dumps(
        [
            {
                "prompt": f"What does study point {index} explain?",
                "answer": sentences[(index - 1) % len(sentences)],
                "citations": [citations[(index - 1) % len(citations)]],
            }
            for index in range(1, count + 1)
        ]
    )


def _flashcard_response(prompt: str) -> str:
    count = _leading_count(prompt)
    sentences = _source_sentences(prompt)
    citations = _source_chunk_ids(prompt.split("Source text:", 1)[-1])
    return json.dumps(
        [
            {
                "front": f"Study point {index}",
                "back": sentences[(index - 1) % len(sentences)],
                "citations": [citations[(index - 1) % len(citations)]],
            }
            for index in range(1, count + 1)
        ]
    )


def _document_brief_response(prompt: str) -> str:
    document_match = re.search(r"DOCUMENT_ID:\s*(\S+)", prompt)
    if document_match is None:
        raise ValueError("Document brief prompt is missing its document ID.")
    blocks = re.findall(
        r"\[([^|\]]+)\|page=[^\]]+\]\n(.*?)(?=\n\n\[|\n\nOUTPUT CONTRACT)",
        prompt,
        flags=re.DOTALL,
    )
    if not blocks:
        raise ValueError("Document brief prompt does not contain cited sources.")
    chunk_ids = [chunk_id for chunk_id, _ in blocks]
    source = " ".join(content.strip() for _, content in blocks)
    sentences = _sentences(source)
    synopsis = " ".join(sentences[:2])
    objectives = [
        {
            "id": f"objective-{index}",
            "title": f"Understand study point {index}",
            "description": sentences[(index - 1) % len(sentences)],
            "prerequisites": [] if index == 1 else [f"objective-{index - 1}"],
            "citations": [chunk_ids[(index - 1) % len(chunk_ids)]],
        }
        for index in range(1, 4)
    ]
    concepts = [
        {
            "id": f"concept-{index}",
            "label": f"Key concept {index}",
            "explanation": sentences[(index - 1) % len(sentences)],
            "prerequisites": [] if index == 1 else [f"concept-{index - 1}"],
            "citations": [chunk_ids[(index - 1) % len(chunk_ids)]],
        }
        for index in range(1, 5)
    ]
    glossary = [
        {
            "term": f"Study term {index}",
            "definition": sentences[(index - 1) % len(sentences)],
            "citations": [chunk_ids[(index - 1) % len(chunk_ids)]],
        }
        for index in range(1, 4)
    ]
    return json.dumps(
        {
            "synopsis": synopsis,
            "objectives": objectives,
            "concepts": concepts,
            "glossary": glossary,
            "misconceptions": [
                "A claim without cited document evidence is unsupported.",
                "One study point should not be treated as the entire document.",
            ],
        }
    )


def _assessment_response(prompt: str) -> str:
    learner_response = _between(prompt, "LEARNER RESPONSE:", "\n\nSOURCES:")
    objective_marker = "OBJECTIVE_ID:" if "OBJECTIVE_ID:" in prompt else "OBJECTIVE:"
    objective_end = (
        "\nPENDING_QUESTION:"
        if objective_marker == "OBJECTIVE_ID:"
        else "\nLEARNER RESPONSE:"
    )
    objective = _between(prompt, objective_marker, objective_end)
    source = prompt.split("SOURCES:", 1)[-1]
    response_terms = _content_terms(learner_response)
    evidence_terms = _content_terms(f"{objective} {source}")
    supported_terms = response_terms & evidence_terms
    coverage = len(supported_terms) / max(1, len(response_terms))
    if not learner_response.strip() or coverage < 0.25:
        score = 0
    elif coverage < 0.5:
        score = 1
    elif coverage < 0.75:
        score = 2
    else:
        score = 3
    missing = sorted(evidence_terms - response_terms)[:3]
    if score >= 2:
        feedback = (
            "Your response is grounded in the cited material. Add one precise "
            "detail from the source to make the explanation more complete."
        )
    else:
        feedback = (
            "Your response needs a clearer connection to the cited material. "
            "Restate the central relationship using the source terminology."
        )
    result: dict[str, object] = {
        "score": score,
        "feedback": feedback,
        "missing_concepts": missing,
        "next_question": (
            f"How would you explain {objective.strip()} in your own words?"
        ),
    }
    if "supplied excerpts" in prompt:
        result["citations"] = _source_chunk_ids(source)
    return json.dumps(result)


def _verification_response(prompt: str) -> str:
    response = _between(prompt, "RESPONSE:", "\n\nSOURCES:").strip()
    source = prompt.split("SOURCES:", 1)[-1]
    response_terms = _content_terms(response)
    source_terms = _content_terms(source)
    coverage = len(response_terms & source_terms) / max(1, len(response_terms))
    supported = bool(response) and coverage >= 0.6
    corrected = response if supported else " ".join(_sentences(source)[:2])
    return json.dumps({"supported": supported, "response": corrected})


def _source_sentences(prompt: str) -> tuple[str, ...]:
    source = prompt.split("Source text:", 1)[-1]
    return _sentences(_source_content(source))


def _source_chunk_ids(source: str) -> tuple[str, ...]:
    identifiers = tuple(re.findall(r"\[([^|\]]+)\|page=", source))
    if identifiers:
        return identifiers
    identifiers = tuple(
        line.strip()
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith(("[", "Return"))
    )
    return identifiers or ("chunk-1",)


def _source_content(source: str) -> str:
    return re.sub(r"\[[^|\]]+\|page=[^\]]+\]\s*", "", source)


def _sentences(source: str) -> tuple[str, ...]:
    sentences = tuple(
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", source)
        if len(sentence.strip()) >= 20
    )
    return sentences or ("The selected source contains this study point.",)


def _leading_count(prompt: str) -> int:
    match = re.match(r"Create exactly (\d+)", prompt)
    if match is None:
        raise ValueError("Structured generation prompt is missing its item count.")
    return int(match.group(1))


def _requested_count(
    request: str,
    patterns: tuple[str, ...],
) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, request)
        if match is not None:
            return int(match.group(1))
    return None


def _position(request: str, words: tuple[str, ...]) -> int:
    positions = [request.find(word) for word in words if request.find(word) >= 0]
    return min(positions) if positions else len(request)


def _requests_answer(request: str) -> bool:
    lowered = request.casefold()
    return (
        request.rstrip().endswith("?")
        or "answer " in lowered
        or lowered.startswith(("what ", "why ", "how "))
    )


def _question_text(request: str) -> str:
    match = re.search(
        r"(?:answer(?: this question)?[: ]+)(.*?)(?=\bthen\b|$)",
        request,
        flags=re.IGNORECASE,
    )
    if match is not None and match.group(1).strip():
        return match.group(1).strip().rstrip(",.")
    return request.strip()


def _between(value: str, start: str, end: str) -> str:
    return value.split(start, 1)[-1].split(end, 1)[0].strip()


def _content_terms(value: str) -> set[str]:
    stop_words = {
        "about",
        "after",
        "also",
        "because",
        "been",
        "being",
        "from",
        "have",
        "into",
        "only",
        "source",
        "that",
        "their",
        "there",
        "these",
        "this",
        "using",
        "with",
        "would",
    }
    return {
        word.casefold()
        for word in re.findall(r"[A-Za-z][A-Za-z-]{3,}", value)
        if word.casefold() not in stop_words
    }
