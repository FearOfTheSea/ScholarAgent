"""Instruction-tune the custom GPT-2 model for ScholarAgent's local contracts."""

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from scholar_agent.application.dtos.retrieval import DocumentChunk
from scholar_agent.application.services.document_brief_parser import (
    document_brief_prompt,
)
from scholar_agent.application.services.mission_planning import (
    build_mission_plan_prompt,
)
from scholar_agent.application.services.mission_prompts import (
    assess_response_prompt,
    explain_concept_prompt,
)
from scholar_agent.application.services.study_prompts import (
    chunks_to_source_text,
    flashcards_prompt,
    quiz_prompt,
)
from scholar_agent.domain.entities.study_session import (
    DocumentBrief,
    LearnerLevel,
    LearningObjective,
    StudyMode,
)
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.domain.value_objects.source_reference import SourceReference
from scholar_agent.infrastructure.adapters.scratch_gpt.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from scholar_agent.infrastructure.adapters.scratch_gpt.gpt_model import (
    GPT2_124M_CONFIG,
    GPTModel,
    load_gpt2_weights,
)
from scholar_agent.infrastructure.adapters.study_agent_planning import (
    build_planner_prompt,
)
from scholar_agent.infrastructure.tools.capabilities import (
    STUDY_CAPABILITIES,
)

PROMPT_SUFFIX = "\n\nResponse:"
CATEGORY_REPETITIONS = {
    "planner": 5,
    "quiz": 1,
    "flashcards": 3,
    "document_brief": 3,
    "assessment": 1,
    "mission_plan": 5,
    "explanation": 2,
    "mission_assessment": 1,
    "verification": 1,
}


@dataclass(frozen=True, slots=True)
class Topic:
    """Small grounded knowledge unit used to create varied instructions."""

    name: str
    definition: str
    purpose: str
    misconception: str


@dataclass(frozen=True, slots=True)
class InstructionExample:
    """One prompt, target response, and deterministic split."""

    prompt: str
    response: str
    category: str
    validation: bool = False


TOPICS = (
    Topic(
        "linear regression",
        "Linear regression predicts a numeric target with a weighted input sum.",
        "It estimates relationships and supports numeric prediction.",
        "It does not require the observed data to be perfectly linear.",
    ),
    Topic(
        "gradient descent",
        "Gradient descent updates parameters in the direction that reduces loss.",
        "It finds useful parameters when a direct solution is impractical.",
        "A larger learning rate does not always converge faster.",
    ),
    Topic(
        "cost function",
        "A cost function measures disagreement between predictions and targets.",
        "It gives training a numerical objective to minimize.",
        "Low training cost alone does not guarantee good generalization.",
    ),
    Topic(
        "learning rate",
        "The learning rate controls the size of each optimization update.",
        "It balances training speed against stability.",
        "One learning rate is not optimal for every problem.",
    ),
    Topic(
        "normal equation",
        "The normal equation directly solves ordinary linear regression parameters.",
        "It avoids iterative optimization for suitable small feature sets.",
        "It is not always cheaper than gradient descent on large feature sets.",
    ),
    Topic(
        "supervised learning",
        "Supervised learning fits a mapping from labeled inputs to targets.",
        "It supports prediction when representative labeled examples exist.",
        "Labels do not automatically make a dataset unbiased.",
    ),
    Topic(
        "retrieval-augmented generation",
        "RAG retrieves source evidence before generating a grounded response.",
        "It connects answers to selected knowledge without retraining the model.",
        "Retrieval does not guarantee that every generated claim is supported.",
    ),
    Topic(
        "clean architecture",
        "Clean architecture keeps policies independent from external frameworks.",
        "It makes providers replaceable while preserving business behavior.",
        "Adding more interfaces does not automatically improve architecture.",
    ),
    Topic(
        "embeddings",
        "Embeddings represent content as vectors that preserve useful similarity.",
        "They enable semantic retrieval beyond exact keyword matching.",
        "Nearby vectors are not proof that two claims are equivalent.",
    ),
    Topic(
        "vector search",
        "Vector search ranks stored embeddings against a query embedding.",
        "It retrieves semantically related evidence efficiently.",
        "The highest similarity result is not always relevant enough to answer.",
    ),
    Topic(
        "causal attention",
        "Causal attention prevents each token from reading future tokens.",
        "It enables autoregressive next-token generation.",
        "The causal mask does not remove the need for positional information.",
    ),
    Topic(
        "overfitting",
        (
            "Overfitting occurs when a model learns training details that do not "
            "generalize."
        ),
        "Recognizing it guides regularization and validation choices.",
        "A very low training loss does not rule out overfitting.",
    ),
)


class ResponseOnlyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Tokenize examples while masking every prompt token from the loss."""

    def __init__(
        self,
        examples: tuple[InstructionExample, ...],
        tokenizer: GPT2Tokenizer,
        maximum_length: int,
    ) -> None:
        self._items = tuple(
            _encode_example(example, tokenizer, maximum_length) for example in examples
        )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._items[index]


def build_instruction_examples() -> tuple[InstructionExample, ...]:
    """Create deterministic grounded, structured, and planning examples."""
    examples: list[InstructionExample] = []
    for index, topic in enumerate(TOPICS):
        validation = index >= len(TOPICS) - 2
        document_id = DocumentId(f"training-document-{index}")
        chunk_id = f"11111111-2222-3333-4444-{index:012d}-0"
        source = f"{topic.definition} {topic.purpose} {topic.misconception}"
        chunk = DocumentChunk(
            document_id=document_id,
            content=source,
            page_number=index + 1,
            section=None,
            chunk_id=chunk_id,
            ordinal=0,
        )
        examples.extend(
            (
                InstructionExample(
                    quiz_prompt(chunks_to_source_text((chunk,)), 2),
                    json.dumps(
                        [
                            {
                                "prompt": f"What is {topic.name}?",
                                "answer": topic.definition,
                                "citations": [chunk_id],
                            },
                            {
                                "prompt": f"Why is {topic.name} useful?",
                                "answer": topic.purpose,
                                "citations": [chunk_id],
                            },
                        ]
                    ),
                    "quiz",
                    validation,
                ),
                InstructionExample(
                    flashcards_prompt(chunks_to_source_text((chunk,)), 2),
                    json.dumps(
                        [
                            {
                                "front": topic.name.title(),
                                "back": topic.definition,
                                "citations": [chunk_id],
                            },
                            {
                                "front": f"Purpose of {topic.name}",
                                "back": topic.purpose,
                                "citations": [chunk_id],
                            },
                        ]
                    ),
                    "flashcards",
                    validation,
                ),
                InstructionExample(
                    document_brief_prompt(document_id, (chunk,)),
                    _brief_response(topic, chunk_id),
                    "document_brief",
                    validation,
                ),
                InstructionExample(
                    _assessment_prompt(topic, source, correct=True),
                    json.dumps(
                        {
                            "score": 3,
                            "feedback": (
                                f"The response correctly explains {topic.name}."
                            ),
                            "missing_concepts": [],
                            "next_question": (
                                f"What is one limitation of {topic.name}?"
                            ),
                        }
                    ),
                    "assessment",
                    validation,
                ),
                InstructionExample(
                    _verification_prompt(topic.definition, source),
                    json.dumps({"supported": True, "response": topic.definition}),
                    "verification",
                    validation,
                ),
            )
        )
        examples.extend(_mission_examples(topic, document_id, chunk_id, source))
    examples.extend(_planner_examples())
    return tuple(examples)


def _mission_examples(
    topic: Topic, document_id: DocumentId, chunk_id: str, source: str
) -> tuple[InstructionExample, ...]:
    reference = SourceReference(document_id, chunk_id, 1, source)
    brief = DocumentBrief(
        document_id=document_id,
        synopsis=f"A cited overview of {topic.name}.",
        objectives=(
            LearningObjective(
                identifier="objective-1",
                title=f"Explain {topic.name}",
                description=topic.definition,
                prerequisite_ids=(),
                citations=(reference,),
            ),
        ),
        concepts=(),
        glossary=(),
        misconceptions=(),
    )
    cited_source = f"[{chunk_id}|page=1]\n{source}"
    return (
        InstructionExample(
            build_mission_plan_prompt(
                f"Understand {topic.name}",
                LearnerLevel.INTERMEDIATE,
                StudyMode.GUIDED,
                20,
                brief,
            ),
            json.dumps(
                {
                    "focus": f"Understand {topic.name}.",
                    "objective_ids": ["objective-1"],
                }
            ),
            "mission_plan",
        ),
        InstructionExample(
            explain_concept_prompt("objective-1", None, "concise", cited_source),
            json.dumps(
                {
                    "explanation": topic.definition,
                    "check_question": f"What is one use of {topic.name}?",
                    "citations": [chunk_id],
                }
            ),
            "explanation",
        ),
        InstructionExample(
            assess_response_prompt(
                "objective-1",
                f"Explain {topic.name}.",
                topic.definition,
                cited_source,
            ),
            json.dumps(
                {
                    "score": 3,
                    "feedback": f"The response explains {topic.name} accurately.",
                    "missing_concepts": [],
                    "next_question": f"How is {topic.name} useful?",
                    "citations": [chunk_id],
                }
            ),
            "mission_assessment",
        ),
    )


def _brief_response(topic: Topic, chunk_id: str) -> str:
    objectives = [
        {
            "id": f"objective-{number}",
            "title": f"{verb} {topic.name}",
            "description": text,
            "prerequisites": [] if number == 1 else [f"objective-{number - 1}"],
            "citations": [chunk_id],
        }
        for number, (verb, text) in enumerate(
            (
                ("Define", topic.definition),
                ("Apply", topic.purpose),
                ("Critique", topic.misconception),
            ),
            start=1,
        )
    ]
    concepts = [
        {
            "id": f"concept-{number}",
            "label": label,
            "explanation": explanation,
            "prerequisites": [] if number == 1 else [f"concept-{number - 1}"],
            "citations": [chunk_id],
        }
        for number, (label, explanation) in enumerate(
            (
                (topic.name.title(), topic.definition),
                ("Purpose", topic.purpose),
                ("Limitation", topic.misconception),
                ("Evidence", topic.definition),
            ),
            start=1,
        )
    ]
    glossary = [
        {
            "term": term,
            "definition": definition,
            "citations": [chunk_id],
        }
        for term, definition in (
            (topic.name.title(), topic.definition),
            ("Purpose", topic.purpose),
            ("Misconception", topic.misconception),
        )
    ]
    return json.dumps(
        {
            "synopsis": f"This document explains {topic.name} and its use.",
            "objectives": objectives,
            "concepts": concepts,
            "glossary": glossary,
            "misconceptions": [topic.misconception, "Evidence still matters."],
        }
    )


def _assessment_prompt(topic: Topic, source: str, correct: bool) -> str:
    response = (
        topic.definition
        if correct
        else f"{topic.name.title()} means that no evidence is required."
    )
    return (
        "Assess the learner response using only the sources. Return JSON only: "
        '{"score":0,"feedback":"...","missing_concepts":["..."],'
        '"next_question":"..."}. score is 0-3: 0 unsupported/incorrect, '
        "1 partial, 2 mostly correct, 3 correct and complete. Feedback must be "
        "specific and next_question must not reveal its answer.\n\n"
        "LEVEL: intermediate\n"
        f"OBJECTIVE: Explain {topic.name}: {topic.definition}\n"
        f"LEARNER RESPONSE: {response}\n\n"
        f"SOURCES:\n[training-chunk|page=1]\n{source}"
    )


def _verification_prompt(response: str, source: str) -> str:
    return (
        "Verify whether every factual claim in RESPONSE is supported by SOURCES. "
        "Return JSON only with supported (boolean) and response (a corrected, "
        "fully supported response). Keep the original when already supported.\n"
        f"RESPONSE:\n{response}\n\n"
        f"SOURCES:\n[training-chunk|page=1]\n{source}"
    )


def _planner_examples() -> tuple[InstructionExample, ...]:
    requests_and_actions: tuple[
        tuple[str, list[dict[str, object]], str | None], ...
    ] = (
        (
            "What is the central claim?",
            [
                {
                    "tool_name": "answer_question",
                    "arguments": {"question": "What is the central claim?"},
                }
            ],
            None,
        ),
        (
            "Summarize this PDF.",
            [{"tool_name": "summarize_document", "arguments": {}}],
            None,
        ),
        (
            "Create 4 quiz questions.",
            [
                {
                    "tool_name": "generate_quiz",
                    "arguments": {"question_count": 4},
                }
            ],
            None,
        ),
        (
            "Make 6 flashcards.",
            [
                {
                    "tool_name": "generate_flashcards",
                    "arguments": {"card_count": 6},
                }
            ],
            None,
        ),
        (
            "Summarize this and make a 3-question quiz.",
            [
                {"tool_name": "summarize_document", "arguments": {}},
                {
                    "tool_name": "generate_quiz",
                    "arguments": {"question_count": 3},
                },
            ],
            None,
        ),
        (
            "Answer what gradient descent is, then make 5 flashcards.",
            [
                {
                    "tool_name": "answer_question",
                    "arguments": {"question": "What is gradient descent?"},
                },
                {
                    "tool_name": "generate_flashcards",
                    "arguments": {"card_count": 5},
                },
            ],
            None,
        ),
        (
            "Give me an overview and create 8 study cards.",
            [
                {"tool_name": "summarize_document", "arguments": {}},
                {
                    "tool_name": "generate_flashcards",
                    "arguments": {"card_count": 8},
                },
            ],
            None,
        ),
        (
            "Create a 2-question quiz and 3 flashcards.",
            [
                {
                    "tool_name": "generate_quiz",
                    "arguments": {"question_count": 2},
                },
                {
                    "tool_name": "generate_flashcards",
                    "arguments": {"card_count": 3},
                },
            ],
            None,
        ),
        (
            "Answer what loss means, summarize, make 2 questions and 3 cards.",
            [
                {
                    "tool_name": "answer_question",
                    "arguments": {"question": "What does loss mean?"},
                },
                {"tool_name": "summarize_document", "arguments": {}},
                {
                    "tool_name": "generate_quiz",
                    "arguments": {"question_count": 2},
                },
                {
                    "tool_name": "generate_flashcards",
                    "arguments": {"card_count": 3},
                },
            ],
            None,
        ),
        (
            "Prepare me for an exam.",
            [
                {"tool_name": "summarize_document", "arguments": {}},
                {"tool_name": "generate_flashcards", "arguments": {}},
                {"tool_name": "generate_quiz", "arguments": {}},
            ],
            None,
        ),
        (
            "Compare this PDF with another document.",
            [],
            "Comparison is not supported; select one document.",
        ),
    )
    return tuple(
        InstructionExample(
            prompt=build_planner_prompt(request, STUDY_CAPABILITIES, 5),
            response=json.dumps({"actions": actions, "message": message}),
            category="planner",
            validation=index >= len(requests_and_actions) - 2,
        )
        for index, (request, actions, message) in enumerate(requests_and_actions)
    )


def _encode_example(
    example: InstructionExample,
    tokenizer: GPT2Tokenizer,
    maximum_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise RuntimeError("GPT-2 tokenizer requires an EOS token.")
    prompt_ids = tokenizer.encode(f"{example.prompt.rstrip()}{PROMPT_SUFFIX}")
    response_ids = tokenizer.encode(example.response) + [eos_token_id]
    if len(response_ids) >= maximum_length:
        raise ValueError(
            f"{example.category} response exceeds maximum training length."
        )
    prompt_budget = maximum_length - len(response_ids)
    prompt_ids = _fit_both_ends(prompt_ids, prompt_budget)
    input_ids = torch.tensor(prompt_ids + response_ids, dtype=torch.long)
    labels = torch.tensor(
        [-100] * len(prompt_ids) + response_ids,
        dtype=torch.long,
    )
    return input_ids, labels


def _fit_both_ends(token_ids: list[int], limit: int) -> list[int]:
    if len(token_ids) <= limit:
        return token_ids
    head_length = max(1, limit // 2)
    tail_length = limit - head_length
    if tail_length == 0:
        return token_ids[:head_length]
    return token_ids[:head_length] + token_ids[-tail_length:]


def _freeze_early_blocks(model: GPTModel, train_last_blocks: int) -> None:
    block_count = len(model.trf_blocks)
    frozen_count = max(0, block_count - train_last_blocks)
    for index, module in enumerate(model.trf_blocks):
        if index < frozen_count:
            for parameter in module.parameters():
                parameter.requires_grad = False


def _response_loss(
    model: GPTModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_function: nn.CrossEntropyLoss,
) -> torch.Tensor:
    logits = model(input_ids)
    return loss_function(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
    )


def _validation_loss(
    model: GPTModel,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_function: nn.CrossEntropyLoss,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for input_ids, labels in loader:
            loss = _response_loss(
                model,
                input_ids.to(device),
                labels.to(device),
                loss_function,
            )
            losses.append(float(loss.item()))
    return sum(losses) / len(losses)


def train(args: argparse.Namespace) -> None:
    """Run bounded response-only instruction tuning and save the best model."""
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = _select_device(args.device)
    print(f"Using device: {device}")
    tokenizer = GPT2Tokenizer.from_pretrained(
        "gpt2",
        local_files_only=args.offline,
    )
    examples = build_instruction_examples()
    training_examples = tuple(
        item
        for item in examples
        if not item.validation
        for _ in range(CATEGORY_REPETITIONS[item.category])
    )
    validation_examples = tuple(item for item in examples if item.validation)
    training_data = ResponseOnlyDataset(
        training_examples,
        tokenizer,
        args.maximum_length,
    )
    validation_data = ResponseOnlyDataset(
        validation_examples,
        tokenizer,
        args.maximum_length,
    )
    generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(
        training_data,
        batch_size=1,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(validation_data, batch_size=1)

    if args.resume is not None:
        checkpoint = load_checkpoint(
            Path(args.resume),
            legacy_config=GPT2_124M_CONFIG,
            device=device,
        )
        if checkpoint.config != GPT2_124M_CONFIG:
            raise ValueError("Resume checkpoint is not the GPT-2 124M architecture.")
        model = GPTModel(checkpoint.config)
        model.load_state_dict(checkpoint.state_dict)
    else:
        pretrained = GPT2LMHeadModel.from_pretrained(
            "gpt2",
            local_files_only=args.offline,
        )
        model = GPTModel(GPT2_124M_CONFIG)
        load_gpt2_weights(model, pretrained)
        del pretrained
    _freeze_early_blocks(model, args.train_last_blocks)
    model.to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    updates_per_epoch = math.ceil(len(training_loader) / args.gradient_accumulation)
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_updates = max(1, total_updates // 10)

    def learning_rate_multiplier(update: int) -> float:
        if update < warmup_updates:
            return (update + 1) / warmup_updates
        remaining = max(0, total_updates - update)
        return remaining / max(1, total_updates - warmup_updates)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        learning_rate_multiplier,
    )
    started = time.monotonic()
    deadline = started + args.maximum_minutes * 60
    best_validation_loss = float("inf")
    patience_reference_loss = float("inf")
    epochs_without_improvement = 0
    update = 0
    optimizer.zero_grad(set_to_none=True)

    print(
        f"Training {len(training_examples)} examples; validating "
        f"{len(validation_examples)}; "
        f"{sum(parameter.numel() for parameter in trainable_parameters):,} "
        "trainable parameters."
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (input_ids, labels) in enumerate(training_loader, start=1):
            loss = _response_loss(
                model,
                input_ids.to(device),
                labels.to(device),
                loss_function,
            )
            (loss / args.gradient_accumulation).backward()
            running_loss += float(loss.item())
            should_update = step % args.gradient_accumulation == 0 or step == len(
                training_loader
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
            if time.monotonic() >= deadline:
                break

        validation_loss = _validation_loss(
            model,
            validation_loader,
            loss_function,
            device,
        )
        training_loss = running_loss / max(1, step)
        elapsed_minutes = (time.monotonic() - started) / 60
        print(
            f"epoch={epoch} train_loss={training_loss:.4f} "
            f"validation_loss={validation_loss:.4f} "
            f"minutes={elapsed_minutes:.2f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(
                Path(args.output),
                GPT2_124M_CONFIG,
                model.state_dict(),
                {
                    "base_model": "gpt2",
                    "training_examples": len(training_examples),
                    "validation_examples": len(validation_examples),
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "response_only_loss": True,
                    "train_last_blocks": args.train_last_blocks,
                    "elapsed_minutes": elapsed_minutes,
                    "resumed_from": args.resume,
                },
            )
        if validation_loss < patience_reference_loss - args.minimum_improvement:
            patience_reference_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience or time.monotonic() >= deadline:
            break
    print(
        f"Best validation loss: {best_validation_loss:.4f}; checkpoint: {args.output}"
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/scholar_gpt.pt")
    parser.add_argument("--resume")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device; auto selects CUDA when available.",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--maximum-minutes", type=float, default=42)
    parser.add_argument("--maximum-length", type=int, default=768)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--train-last-blocks", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--minimum-improvement", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def _select_device(requested: str) -> torch.device:
    """Resolve the requested training device with an actionable CUDA error."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Install a CUDA-enabled "
            "PyTorch build and verify the NVIDIA driver."
        )
    return torch.device(requested)


if __name__ == "__main__":
    train(_arguments())
