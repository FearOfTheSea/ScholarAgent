"""Evaluate a trained ScholarGPT checkpoint against application contracts."""

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from scholar_agent.application.dtos.retrieval import DocumentChunk
from scholar_agent.application.services.document_brief_parser import (
    document_brief_prompt,
    parse_document_brief,
)
from scholar_agent.application.services.structured_output import parse_items
from scholar_agent.application.services.study_prompts import (
    flashcards_prompt,
    quiz_prompt,
)
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.infrastructure.adapters.scratch_gpt.scratch_gpt_adapter import (
    ScratchGPTAdapter,
)
from scholar_agent.infrastructure.adapters.study_agent_planning import (
    build_planner_prompt,
    parse_study_plan,
)
from scholar_agent.infrastructure.tools.capabilities import STUDY_CAPABILITIES


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One generated response and contract validator."""

    name: str
    prompt: str
    maximum_tokens: int
    validate: Callable[[str], None]


def evaluate(checkpoint: Path) -> int:
    """Run held-out contract checks and return the number that pass."""
    cases = _cases()
    passed = 0
    for case in cases:
        output = "<none>"
        adapter = ScratchGPTAdapter(
            context_length=1024,
            maximum_tokens=case.maximum_tokens,
            checkpoint_path=checkpoint,
        )
        try:
            output = adapter.generate(case.prompt)
            case.validate(output)
        except (RuntimeError, ValueError, AssertionError) as error:
            print(f"FAIL {case.name}: {error}")
            print(f"OUTPUT: {output}")
        else:
            passed += 1
            print(f"PASS {case.name}")
    print(f"ScholarGPT contract score: {passed}/{len(cases)}")
    return passed


def _cases() -> tuple[EvaluationCase, ...]:
    source = (
        "Bayesian inference updates prior beliefs with observed evidence to form "
        "a posterior distribution. It represents uncertainty explicitly. A "
        "posterior is not determined by evidence alone because the prior matters."
    )
    document_id = DocumentId("held-out-bayesian-document")
    chunk = DocumentChunk(
        document_id=document_id,
        content=source,
        page_number=7,
        section=None,
        chunk_id="99999999-8888-7777-6666-555555555555-0",
        ordinal=0,
    )
    planner_prompt = build_planner_prompt(
        "Summarize this document and make 2 flashcards.",
        STUDY_CAPABILITIES,
        5,
    )
    return (
        EvaluationCase(
            "planner",
            planner_prompt,
            220,
            _validate_planner,
        ),
        EvaluationCase(
            "quiz",
            quiz_prompt(source, 2),
            280,
            lambda output: _validate_items(output, "prompt", "answer", 2),
        ),
        EvaluationCase(
            "flashcards",
            flashcards_prompt(source, 2),
            280,
            lambda output: _validate_items(output, "front", "back", 2),
        ),
        EvaluationCase(
            "document_brief",
            document_brief_prompt(document_id, (chunk,)),
            760,
            lambda output: _validate_brief(output, document_id, chunk),
        ),
        EvaluationCase(
            "assessment",
            _assessment_prompt(source),
            240,
            _validate_assessment,
        ),
        EvaluationCase(
            "verification",
            _verification_prompt(source),
            160,
            _validate_verification,
        ),
    )


def _validate_planner(output: str) -> None:
    actions, _ = parse_study_plan(output, STUDY_CAPABILITIES, 5)
    assert [action["tool_name"] for action in actions] == [
        "summarize_document",
        "generate_flashcards",
    ]
    assert actions[1]["arguments"]["card_count"] == 2


def _validate_items(
    output: str,
    first_key: str,
    second_key: str,
    count: int,
) -> None:
    items = parse_items(output, first_key, second_key)
    assert len(items) == count
    assert all(first and second for first, second in items)


def _validate_brief(
    output: str,
    document_id: DocumentId,
    chunk: DocumentChunk,
) -> None:
    brief = parse_document_brief(output, document_id, (chunk,))
    assert brief.objectives
    assert brief.concepts
    assert all(
        citation.document_id == document_id
        for objective in brief.objectives
        for citation in objective.citations
    )


def _validate_assessment(output: str) -> None:
    payload = _json_object(output)
    assert set(payload) == {
        "score",
        "feedback",
        "missing_concepts",
        "next_question",
    }
    score = payload["score"]
    assert isinstance(score, int) and not isinstance(score, bool)
    assert 0 <= score <= 3
    assert isinstance(payload["feedback"], str) and payload["feedback"]
    assert isinstance(payload["missing_concepts"], list)
    assert isinstance(payload["next_question"], str) and payload["next_question"]


def _validate_verification(output: str) -> None:
    payload = _json_object(output)
    assert set(payload) == {"supported", "response"}
    assert isinstance(payload["supported"], bool)
    assert isinstance(payload["response"], str) and payload["response"]


def _json_object(output: str) -> dict[str, object]:
    start = output.find("{")
    end = output.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Output does not contain a JSON object.")
    payload = json.loads(output[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Output JSON must be an object.")
    return {str(key): value for key, value in payload.items()}


def _assessment_prompt(source: str) -> str:
    return (
        "Assess the learner response using only the sources. Return JSON only: "
        '{"score":0,"feedback":"...","missing_concepts":["..."],'
        '"next_question":"..."}. score is 0-3: 0 unsupported/incorrect, '
        "1 partial, 2 mostly correct, 3 correct and complete. Feedback must be "
        "specific and next_question must not reveal its answer.\n\n"
        "LEVEL: intermediate\n"
        "OBJECTIVE: Explain Bayesian inference.\n"
        "LEARNER RESPONSE: It combines prior beliefs with observed evidence.\n\n"
        f"SOURCES:\n[held-out-chunk|page=7]\n{source}"
    )


def _verification_prompt(source: str) -> str:
    return (
        "Verify whether every factual claim in RESPONSE is supported by SOURCES. "
        "Return JSON only with supported (boolean) and response (a corrected, "
        "fully supported response). Keep the original when already supported.\n"
        "RESPONSE:\nBayesian inference updates beliefs with evidence.\n\n"
        f"SOURCES:\n[held-out-chunk|page=7]\n{source}"
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/scholar_gpt.pt"),
    )
    parser.add_argument("--require-score", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _arguments()
    score = evaluate(arguments.checkpoint)
    if score < arguments.require_score:
        raise SystemExit(1)
