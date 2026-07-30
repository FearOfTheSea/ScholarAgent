"""Prompt construction and strict parsing for study-agent plans."""

import json
from typing import TypedDict, cast

from scholar_agent.application.dtos.agent import StudyTask
from scholar_agent.application.output_ports.tool_executor import (
    StudyToolDefinition,
    ToolArgumentDefinition,
    ToolArgumentKind,
)


class PlannedAction(TypedDict):
    """One fully validated tool invocation."""

    tool_name: str
    arguments: dict[str, object]


def build_planner_prompt(
    learner_prompt: str,
    definitions: tuple[StudyToolDefinition, ...],
    quiz_count_default: int,
) -> str:
    """Create the constrained routing prompt from registered capabilities."""
    capabilities = [
        _capability_payload(definition, quiz_count_default)
        for definition in definitions
    ]
    return (
        "You are the constrained planner for a local PDF study application.\n"
        "Choose only from the capabilities supplied below. Select every capability "
        "the learner explicitly requests. For a broad study goal, infer a useful "
        "combination of supported capabilities. Each capability may appear at most "
        "once. Never select comparison or any multi-document operation. For an "
        "answer action, extract the actual study question. Preserve an explicitly "
        "requested quiz or flashcard count even when it is large; application "
        "policy applies limits later. Omit optional counts when the learner does "
        "not state one. If the request is unrelated, unsupported, or asks for a "
        "comparison, return no actions and a short helpful message.\n\n"
        "Return only one JSON object with this exact shape:\n"
        '{"actions":[{"tool_name":"name","arguments":{}}],"message":null}\n\n'
        f"CAPABILITIES:\n{json.dumps(capabilities, indent=2)}\n\n"
        f"LEARNER REQUEST:\n{learner_prompt}"
    )


def build_repair_prompt(original_prompt: str, raw_plan: str, error: str) -> str:
    """Ask the local model to repair one invalid structured plan."""
    return (
        f"{original_prompt}\n\n"
        "Your previous response was invalid. Repair it and return only the required "
        "JSON object. Do not add Markdown or commentary.\n"
        f"VALIDATION ERROR: {error}\n"
        f"INVALID RESPONSE: {raw_plan}"
    )


def parse_study_plan(
    raw_plan: str,
    definitions: tuple[StudyToolDefinition, ...],
    quiz_count_default: int,
) -> tuple[list[PlannedAction], str | None]:
    """Parse and fully validate a model-proposed plan without executing it."""
    payload = _json_payload(raw_plan)
    raw_actions = payload.get("actions")
    if not isinstance(raw_actions, list):
        raise ValueError("'actions' must be a JSON array.")
    raw_message = payload.get("message")
    if raw_message is not None and (
        not isinstance(raw_message, str) or not raw_message.strip()
    ):
        raise ValueError("'message' must be non-blank text or null.")

    definitions_by_name = {
        definition.task.value: definition for definition in definitions
    }
    seen: set[str] = set()
    actions: list[PlannedAction] = []
    for raw_action in raw_actions:
        action = _validate_action(
            raw_action,
            definitions_by_name,
            seen,
            quiz_count_default,
        )
        seen.add(action["tool_name"])
        actions.append(action)

    message = raw_message.strip() if isinstance(raw_message, str) else None
    if not actions and message is None:
        message = (
            "Ask me to answer a question, summarize the selected PDF, generate a "
            "quiz, or generate flashcards."
        )
    return actions, message


def _capability_payload(
    definition: StudyToolDefinition,
    quiz_count_default: int,
) -> dict[str, object]:
    arguments: list[dict[str, object]] = []
    for argument in definition.arguments:
        default = argument.default
        if (
            definition.task is StudyTask.GENERATE_QUIZ
            and argument.name == "question_count"
        ):
            default = quiz_count_default
        arguments.append(
            {
                "name": argument.name,
                "type": argument.kind.value,
                "required": argument.required,
                "default_when_omitted": default,
            }
        )
    return {
        "tool_name": definition.task.value,
        "description": definition.description,
        "arguments": arguments,
    }


def _json_payload(raw_plan: str) -> dict[str, object]:
    start = raw_plan.find("{")
    end = raw_plan.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("The planner response does not contain a JSON object.")
    try:
        payload = json.loads(raw_plan[start : end + 1])
    except json.JSONDecodeError as error:
        raise ValueError("The planner response is not valid JSON.") from error
    if not isinstance(payload, dict):
        raise ValueError("The planner response must be a JSON object.")
    if set(payload) != {"actions", "message"}:
        raise ValueError(
            "The planner object must contain only 'actions' and 'message'."
        )
    return cast(dict[str, object], payload)


def _validate_action(
    raw_action: object,
    definitions: dict[str, StudyToolDefinition],
    seen: set[str],
    quiz_count_default: int,
) -> PlannedAction:
    if not isinstance(raw_action, dict):
        raise ValueError("Every action must be a JSON object.")
    if set(raw_action) != {"tool_name", "arguments"}:
        raise ValueError("Every action must contain only 'tool_name' and 'arguments'.")
    tool_name = raw_action.get("tool_name")
    if not isinstance(tool_name, str) or tool_name not in definitions:
        raise ValueError(f"Unsupported tool: {tool_name}.")
    if tool_name in seen:
        raise ValueError(f"Duplicate tool: {tool_name}.")
    raw_arguments = raw_action.get("arguments")
    if not isinstance(raw_arguments, dict):
        raise ValueError(f"Arguments for {tool_name} must be a JSON object.")
    return {
        "tool_name": tool_name,
        "arguments": _validate_arguments(
            tool_name,
            raw_arguments,
            definitions[tool_name].arguments,
            quiz_count_default,
        ),
    }


def _validate_arguments(
    tool_name: str,
    raw_arguments: dict[object, object],
    definitions: tuple[ToolArgumentDefinition, ...],
    quiz_count_default: int,
) -> dict[str, object]:
    if not all(isinstance(key, str) for key in raw_arguments):
        raise ValueError(f"Argument names for {tool_name} must be strings.")
    allowed_names = {definition.name for definition in definitions}
    unknown_names = set(cast(dict[str, object], raw_arguments)) - allowed_names
    if unknown_names:
        names = ", ".join(sorted(unknown_names))
        raise ValueError(f"Unsupported arguments for {tool_name}: {names}.")

    validated: dict[str, object] = {}
    for definition in definitions:
        value = raw_arguments.get(definition.name)
        if value is None:
            _apply_default(
                validated,
                tool_name,
                definition,
                quiz_count_default,
            )
        else:
            validated[definition.name] = _validate_argument_value(
                definition,
                value,
            )
    return validated


def _apply_default(
    validated: dict[str, object],
    tool_name: str,
    definition: ToolArgumentDefinition,
    quiz_count_default: int,
) -> None:
    if definition.required:
        raise ValueError(f"'{definition.name}' is required for {tool_name}.")
    default = definition.default
    if (
        tool_name == StudyTask.GENERATE_QUIZ.value
        and definition.name == "question_count"
    ):
        default = quiz_count_default
    if default is not None:
        validated[definition.name] = default


def _validate_argument_value(
    definition: ToolArgumentDefinition,
    value: object,
) -> object:
    if definition.kind is ToolArgumentKind.TEXT:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"'{definition.name}' must be non-blank text.")
        return value.strip()
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"'{definition.name}' must be a positive integer.")
    return value
