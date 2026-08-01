"""Bounded mission-plan construction and prerequisite policy."""

import json
from collections.abc import Iterable

from scholar_agent.application.output_ports.llm_provider import ILLMProvider
from scholar_agent.domain.entities.study_session import (
    DocumentBrief,
    LearnerLevel,
    LearningObjective,
    MilestoneKind,
    MilestoneStatus,
    SourceReference,
    StudyMilestone,
    StudyMode,
    StudyPlan,
)


class MissionPlanner:
    """Select a small prerequisite-valid objective set for one mission."""

    def __init__(self, llm_provider: ILLMProvider, maximum_objectives: int = 6) -> None:
        self._llm_provider = llm_provider
        self._maximum_objectives = maximum_objectives

    def plan(
        self,
        goal: str,
        level: LearnerLevel,
        mode: StudyMode,
        target_minutes: int,
        brief: DocumentBrief,
    ) -> StudyPlan:
        """Plan with one repair and a deterministic earliest-valid fallback."""
        capacity = max(
            1,
            min(
                len(brief.objectives),
                target_minutes // 10,
                self._maximum_objectives,
            ),
        )
        prompt = build_mission_plan_prompt(goal, level, mode, target_minutes, brief)
        raw = self._llm_provider.generate(prompt)
        try:
            focus, requested_ids = parse_mission_plan(raw, brief)
        except ValueError as first_error:
            repaired = self._llm_provider.generate(
                build_mission_plan_repair_prompt(prompt, raw, str(first_error))
            )
            try:
                focus, requested_ids = parse_mission_plan(repaired, brief)
            except ValueError:
                focus = goal.strip()
                requested_ids = ()
        objective_ids = _fit_prerequisite_closure(brief, requested_ids, capacity)
        if not objective_ids:
            objective_ids = tuple(
                objective.identifier for objective in brief.objectives[:capacity]
            )
        citations = _unique_references(
            reference
            for objective in brief.objectives
            if objective.identifier in objective_ids
            for reference in objective.citations
        )
        return StudyPlan(focus=focus, objective_ids=objective_ids, citations=citations)


def build_mission_plan_prompt(
    goal: str,
    level: LearnerLevel,
    mode: StudyMode,
    target_minutes: int,
    brief: DocumentBrief,
) -> str:
    """Build the exact planner contract from the cited brief."""
    objectives = [
        {
            "id": objective.identifier,
            "title": objective.title,
            "description": objective.description,
            "prerequisites": list(objective.prerequisite_ids),
            "citations": [reference.chunk_id for reference in objective.citations],
        }
        for objective in brief.objectives
    ]
    return (
        "Choose a bounded study mission from this cited document map. Return only "
        'JSON with exactly "focus" and "objective_ids". focus is a short user-facing '
        "description. objective_ids must contain only supplied objective IDs.\n"
        f"GOAL: {goal}\nLEVEL: {level.value}\nMODE: {mode.value}\n"
        f"TARGET_MINUTES: {target_minutes}\nBRIEF_SYNOPSIS: {brief.synopsis}\n"
        f"OBJECTIVES: {json.dumps(objectives)}"
    )


def build_mission_plan_repair_prompt(
    original_prompt: str, raw_output: str, error: str
) -> str:
    """Build one bounded planner repair request."""
    return (
        f"{original_prompt}\nINVALID RESPONSE: {raw_output}\n"
        f"VALIDATION ERROR: {error}\nReturn only the required JSON object."
    )


def parse_mission_plan(
    raw_output: str, brief: DocumentBrief
) -> tuple[str, tuple[str, ...]]:
    """Strictly parse and validate a model-selected mission plan."""
    start = raw_output.find("{")
    end = raw_output.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Mission plan is not a JSON object.")
    try:
        payload = json.loads(raw_output[start : end + 1])
    except json.JSONDecodeError as error:
        raise ValueError("Mission plan is invalid JSON.") from error
    if not isinstance(payload, dict) or set(payload) != {"focus", "objective_ids"}:
        raise ValueError("Mission plan must contain exactly focus and objective_ids.")
    focus = payload.get("focus")
    objective_ids = payload.get("objective_ids")
    if not isinstance(focus, str) or not focus.strip():
        raise ValueError("Mission plan focus must be non-blank text.")
    if not isinstance(objective_ids, list) or not all(
        isinstance(item, str) and item.strip() for item in objective_ids
    ):
        raise ValueError("Mission plan objective_ids must be a string array.")
    identifiers = tuple(item.strip() for item in objective_ids)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Mission plan cannot contain duplicate objectives.")
    allowed = {objective.identifier for objective in brief.objectives}
    unknown = set(identifiers) - allowed
    if unknown:
        raise ValueError(
            f"Mission plan contains unknown objectives: {sorted(unknown)}."
        )
    return focus.strip(), identifiers


def _fit_prerequisite_closure(
    brief: DocumentBrief, requested_ids: tuple[str, ...], capacity: int
) -> tuple[str, ...]:
    by_id = {objective.identifier: objective for objective in brief.objectives}
    selected: set[str] = set()
    for requested_id in requested_ids:
        closure = _closure(requested_id, by_id)
        new_ids = closure - selected
        if len(selected) + len(new_ids) > capacity:
            continue
        selected.update(new_ids)
    return tuple(
        objective.identifier
        for objective in brief.objectives
        if objective.identifier in selected
    )


def _closure(identifier: str, by_id: dict[str, LearningObjective]) -> set[str]:
    objective = by_id[identifier]
    prerequisites = objective.prerequisite_ids
    result: set[str] = {identifier}
    for prerequisite in prerequisites:
        result.update(_closure(prerequisite, by_id))
    return result


def _unique_references(
    references: Iterable[SourceReference],
) -> tuple[SourceReference, ...]:
    result: list[SourceReference] = []
    seen: set[tuple[str, str]] = set()
    for reference in references:
        key = (reference.document_id.value, reference.chunk_id)
        if key not in seen:
            result.append(reference)
            seen.add(key)
    return tuple(result)


def build_mission_milestones(
    brief: DocumentBrief, plan: StudyPlan, mode: StudyMode
) -> tuple[StudyMilestone, ...]:
    """Create a small mode-specific sequence of bounded milestones."""
    by_id = {objective.identifier: objective for objective in brief.objectives}
    milestones: list[StudyMilestone] = [
        StudyMilestone(
            identifier="milestone-orient",
            kind=MilestoneKind.ORIENT,
            title="Orient in the selected document",
            objective_id=None,
            capability="build_document_map",
            status=MilestoneStatus.ACTIVE,
            citations=plan.citations,
        )
    ]
    if mode is StudyMode.EXAM:
        milestones.append(
            StudyMilestone(
                identifier="milestone-diagnostic",
                kind=MilestoneKind.PRACTICE,
                title="Diagnostic check",
                objective_id=None,
                capability="generate_quiz",
                citations=plan.citations,
            )
        )
    if mode is StudyMode.CRAM:
        milestones.extend(
            (
                StudyMilestone(
                    identifier="milestone-summary",
                    kind=MilestoneKind.LEARN,
                    title="Cited overview",
                    objective_id=None,
                    capability="summarize_document",
                    citations=plan.citations,
                ),
                StudyMilestone(
                    identifier="milestone-flashcards",
                    kind=MilestoneKind.LEARN,
                    title="Cited flashcards",
                    objective_id=None,
                    capability="generate_flashcards",
                    citations=plan.citations,
                ),
            )
        )
    elif mode is StudyMode.GUIDED:
        milestones.append(
            StudyMilestone(
                identifier="milestone-overview",
                kind=MilestoneKind.ORIENT,
                title="Cited overview",
                objective_id=None,
                capability="summarize_document",
                status=MilestoneStatus.PENDING,
                citations=plan.citations,
            )
        )
    for objective_id in plan.objective_ids:
        objective = by_id[objective_id]
        milestones.extend(
            (
                StudyMilestone(
                    identifier=f"milestone-learn-{objective_id}",
                    kind=MilestoneKind.LEARN,
                    title=objective.title,
                    objective_id=objective_id,
                    capability="explain_concept",
                    citations=objective.citations,
                ),
                StudyMilestone(
                    identifier=f"milestone-practice-{objective_id}",
                    kind=MilestoneKind.PRACTICE,
                    title=f"Practice {objective.title}",
                    objective_id=objective_id,
                    capability="assess_learner_response",
                    citations=objective.citations,
                ),
            )
        )
    if mode is StudyMode.EXAM:
        milestones.extend(
            (
                StudyMilestone(
                    identifier="milestone-weak-summary",
                    kind=MilestoneKind.REVIEW,
                    title="Weak-point summary",
                    objective_id=None,
                    capability="summarize_document",
                    citations=plan.citations,
                ),
                StudyMilestone(
                    identifier="milestone-weak-flashcards",
                    kind=MilestoneKind.REVIEW,
                    title="Weak-point flashcards",
                    objective_id=None,
                    capability="generate_flashcards",
                    citations=plan.citations,
                ),
            )
        )
    milestones.append(
        StudyMilestone(
            identifier="milestone-review",
            kind=MilestoneKind.REVIEW,
            title="Final review quiz",
            objective_id=None,
            capability="generate_quiz",
            citations=plan.citations,
        )
    )
    return tuple(milestones)
