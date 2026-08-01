"""Tests for ScholarGPT checkpointing and response-only training behavior."""

import json
from pathlib import Path

import torch

from scholar_agent.application.services.study_prompts import (
    quiz_prompt,
    summarize_prompt,
)
from scholar_agent.infrastructure.adapters.scratch_gpt.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from scholar_agent.infrastructure.adapters.scratch_gpt.gpt_model import (
    GPTConfig,
    GPTModel,
)
from scholar_agent.infrastructure.adapters.scratch_gpt.scratch_gpt_adapter import (
    _contains_complete_json,
    _extract_grounded_response,
    _fit_prompt_to_context,
    _generation_budget,
)
from scholar_agent.infrastructure.adapters.scratch_gpt.structured_generation import (
    structured_response,
)
from scholar_agent.infrastructure.adapters.study_agent_planning import (
    build_planner_prompt,
    parse_study_plan,
)
from scholar_agent.infrastructure.tools.capabilities import STUDY_CAPABILITIES
from scripts.train_scholar_gpt import (
    InstructionExample,
    _encode_example,
    build_instruction_examples,
)


class CharacterTokenizer:
    """Small deterministic tokenizer for loss-mask tests."""

    eos_token_id = 0

    @staticmethod
    def encode(value: str) -> list[int]:
        return [ord(character) for character in value]


TINY_CONFIG: GPTConfig = {
    "vocab_size": 128,
    "context_length": 16,
    "emb_dim": 8,
    "n_heads": 2,
    "n_layers": 1,
    "bias": True,
}


def test_training_examples_cover_every_structured_contract() -> None:
    examples = build_instruction_examples()

    assert {item.category for item in examples} == {
        "planner",
        "quiz",
        "flashcards",
        "document_brief",
        "assessment",
        "mission_plan",
        "explanation",
        "mission_assessment",
        "verification",
    }
    assert any(item.validation for item in examples)
    assert any(not item.validation for item in examples)


def test_response_only_labels_mask_prompt_tokens() -> None:
    example = InstructionExample("Prompt", "OK", "test")

    input_ids, labels = _encode_example(
        example,
        CharacterTokenizer(),  # type: ignore[arg-type]
        maximum_length=64,
    )

    first_response_token = int((labels != -100).nonzero()[0].item())
    assert torch.all(labels[:first_response_token] == -100)
    assert torch.equal(input_ids[first_response_token:], labels[first_response_token:])


def test_versioned_checkpoint_round_trip(tmp_path: Path) -> None:
    model = GPTModel(TINY_CONFIG)
    checkpoint_path = tmp_path / "scholar_gpt.pt"
    save_checkpoint(
        checkpoint_path,
        TINY_CONFIG,
        model.state_dict(),
        {"validation_loss": 1.25},
    )

    loaded = load_checkpoint(
        checkpoint_path,
        legacy_config=TINY_CONFIG,
        device=torch.device("cpu"),
    )

    assert loaded.config == TINY_CONFIG
    assert loaded.metadata["validation_loss"] == 1.25
    assert set(loaded.state_dict) == set(model.state_dict())


def test_model_ties_input_and_output_embeddings() -> None:
    model = GPTModel(TINY_CONFIG)

    assert model.tok_emb.weight.data_ptr() == model.out_head.weight.data_ptr()


def test_context_fitting_keeps_instruction_and_recent_evidence() -> None:
    token_ids = list(range(20))

    fitted = _fit_prompt_to_context(token_ids, 8)

    assert fitted == [0, 1, 2, 3, 16, 17, 18, 19]


def test_generation_budget_preserves_prompt_when_setting_exceeds_context() -> None:
    assert _generation_budget(1024, 1024) == 512
    assert _generation_budget(1024, 120) == 120


def test_extractive_fallback_does_not_intercept_structured_generation() -> None:
    source = "This sufficiently long sentence contains grounded source evidence."

    assert _extract_grounded_response(quiz_prompt(source, 2)) is None
    assert _extract_grounded_response(summarize_prompt(source)) == source


def test_json_completion_detects_balanced_value_and_ignores_braces_in_text() -> None:
    assert _contains_complete_json('prefix {"value": "a } brace", "items": [1]}')
    assert not _contains_complete_json('{"value": "unfinished"')


def test_structured_planner_preserves_requested_actions_and_counts() -> None:
    prompt = build_planner_prompt(
        "Summarize the PDF, then make exactly 3 flashcards and 2 quiz questions.",
        STUDY_CAPABILITIES,
        5,
    )

    output = structured_response(prompt)

    assert output is not None
    actions, message = parse_study_plan(output, STUDY_CAPABILITIES, 5)
    assert message is None
    assert [action["tool_name"] for action in actions] == [
        "summarize_document",
        "generate_flashcards",
        "generate_quiz",
    ]
    assert actions[1]["arguments"] == {"card_count": 3}
    assert actions[2]["arguments"] == {"question_count": 2}


def test_structured_tutor_contracts_return_grounded_valid_json() -> None:
    source = (
        "A loss function measures prediction error. Training minimizes the loss "
        "by adjusting model parameters."
    )
    assessment_prompt = (
        "Assess the learner response using only the sources. Return JSON only.\n"
        "OBJECTIVE: Explain the loss function.\n"
        "LEARNER RESPONSE: It measures prediction error.\n\n"
        f"SOURCES:\n[chunk|page=1]\n{source}"
    )
    verification_prompt = (
        "Verify whether every factual claim in RESPONSE is supported by SOURCES. "
        "Return JSON only with supported (boolean) and response (a corrected, "
        "fully supported response).\n"
        "RESPONSE:\nA loss function measures prediction error.\n\n"
        f"SOURCES:\n[chunk|page=1]\n{source}"
    )

    assessment = structured_response(assessment_prompt)
    verification = structured_response(verification_prompt)

    assert assessment is not None
    assert verification is not None
    assert set(json.loads(assessment)) == {
        "score",
        "feedback",
        "missing_concepts",
        "next_question",
    }
    assert json.loads(verification) == {
        "supported": True,
        "response": "A loss function measures prediction error.",
    }
