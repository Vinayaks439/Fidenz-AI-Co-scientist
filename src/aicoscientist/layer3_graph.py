"""Layer 3 — In-Silico Validation as a LangGraph with a bounded closed loop.

    design_experiment -> run_validation -> reflect -> {refine -> design_experiment | finalize}
    finalize -> link_and_persist -> END

This is the "loop engineering" at the heart of Layer 3 (per arXiv:2510.27130): the
Reflection agent can route back to a new experiment design (bounded by MAX_VALIDATION_ITERS)
before the result is linked into the knowledge graph and persisted. The graph is built by
a factory that closes over the knowledge graph and artifact store so those (non-serializable)
objects stay out of graph state.
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .knowledge_graph import KnowledgeGraph
from .models import (
    Layer3Output,
    OfficialHypothesis,
    Reflection,
    ValidationPlan,
    ValidationResult,
)
from .persistence import ArtifactStore
from .validation.designer import ExperimentDesigner
from .validation.reflection import ReflectionAgent
from .validation.registry import get_validator

logger = logging.getLogger(__name__)


class ValidationState(TypedDict, total=False):
    run_id: str
    offline: bool
    hypothesis: str
    official: dict[str, Any]
    concept_names: list[str]

    iteration: int
    max_iters: int

    plan: dict[str, Any]
    result: dict[str, Any]
    critique: dict[str, Any] | None
    screening: dict[str, Any] | None

    history: Annotated[list[dict], operator.add]
    reflections: Annotated[list[dict], operator.add]

    output: dict[str, Any]


def build_layer3_graph(kg: KnowledgeGraph, store: ArtifactStore):
    """Compile the Layer 3 graph bound to a specific KG + artifact store."""

    def design_experiment(state: ValidationState) -> dict:
        designer = ExperimentDesigner(offline=state.get("offline", False))
        official = OfficialHypothesis.model_validate(state["official"])
        critique = state.get("critique")
        critique_obj = Reflection.model_validate(critique) if critique else None
        plan = designer.design(
            official,
            concept_names=state.get("concept_names", []),
            prior_critique=critique_obj,
            iteration=state.get("iteration", 0),
        )
        return {"plan": plan.model_dump()}

    def run_validation(state: ValidationState) -> dict:
        plan = ValidationPlan.model_validate(state["plan"])
        validator = get_validator(plan.domain)
        result = validator.run(
            run_id=state["run_id"],
            hypothesis=state["hypothesis"],
            plan=plan,
            datasets_dir=store.datasets_dir,
            logs_dir=store.logs_dir,
        )
        dump = result.model_dump()
        return {"result": dump, "history": [dump]}

    def reflect(state: ValidationState) -> dict:
        reflector = ReflectionAgent(offline=state.get("offline", False))
        result = ValidationResult.model_validate(state["result"])
        iteration = state.get("iteration", 0)
        max_iters = state.get("max_iters", 2)
        refl = reflector.review(result, iteration, max_iters)
        refine = refl.decision == "refine"
        return {
            "reflections": [refl.model_dump()],
            "critique": refl.model_dump() if refine else None,
            "iteration": iteration + 1,
        }

    def route_after_reflect(state: ValidationState) -> str:
        if state.get("critique") and state.get("iteration", 0) < state.get("max_iters", 2):
            return "design_experiment"
        return "link_and_persist"

    def link_and_persist(state: ValidationState) -> dict:
        result = ValidationResult.model_validate(state["result"])
        history = [ValidationResult.model_validate(r) for r in state.get("history", [])]
        reflections = [Reflection.model_validate(r) for r in state.get("reflections", [])]
        official = OfficialHypothesis.model_validate(state["official"])

        source_ids = list(result.artifact_paths.values())
        node_id = kg.add_validation_result(
            result_id=f"{state['run_id']}-{result.plan.domain}",
            domain=result.plan.domain,
            verdict=result.verdict.value,
            confidence=result.confidence,
            related_concept_ids=official.state_graph.related_concept_ids,
            source_ids=source_ids,
        )

        output = Layer3Output(
            run_id=state["run_id"],
            hypothesis_statement=state["hypothesis"],
            result=result,
            history=history,
            reflections=reflections,
            iterations=len(history),
            screening=state.get("screening"),
        )
        store.save_layer3(output, kg)
        logger.info(
            "Layer 3 complete: %s (%d iteration(s)); KG node %s",
            result.verdict.value,
            len(history),
            node_id,
        )
        return {"output": output.model_dump()}

    builder = StateGraph(ValidationState)
    builder.add_node("design_experiment", design_experiment)
    builder.add_node("run_validation", run_validation)
    builder.add_node("reflect", reflect)
    builder.add_node("link_and_persist", link_and_persist)

    builder.add_edge(START, "design_experiment")
    builder.add_edge("design_experiment", "run_validation")
    builder.add_edge("run_validation", "reflect")
    builder.add_conditional_edges(
        "reflect",
        route_after_reflect,
        ["design_experiment", "link_and_persist"],
    )
    builder.add_edge("link_and_persist", END)

    return builder.compile()


def build_layer3_screening_graph(kg: KnowledgeGraph, store: ArtifactStore):
    """Compile the screening-funnel Layer 3 graph (SCREENING_MODE=funnel).

        run_screening -> reflect -> {refine -> rerun_winner -> reflect | finalize}
        finalize -> link_and_persist -> END

    The funnel itself (pool -> Tier-0 rank -> shared-slab MLIP batch -> top-k
    full-fidelity -> recommendation) runs inside ``run_screening``. Reflection is
    scoped to what it can honestly judge here: the WINNER's evidence quality. On
    "refine" the winner is re-run with a larger surface ensemble (true refinement,
    not candidate hopping -- the funnel already explored the candidate space).
    """
    from .validation.screening import run_screening_campaign

    def run_screening(state: ValidationState) -> dict:
        official = OfficialHypothesis.model_validate(state["official"])
        screening, winner_result = run_screening_campaign(
            run_id=state["run_id"],
            official=official,
            concept_names=state.get("concept_names", []),
            offline=state.get("offline", False),
            datasets_dir=store.datasets_dir,
            logs_dir=store.logs_dir,
        )
        dump = winner_result.model_dump()
        return {
            "screening": screening,
            "plan": winner_result.plan.model_dump(),
            "result": dump,
            "history": [dump],
        }

    def rerun_winner(state: ValidationState) -> dict:
        plan = ValidationPlan.model_validate(state["plan"])
        ds = dict(plan.data_spec)
        ds["ensemble_n"] = int(ds.get("ensemble_n", 5)) + 2
        plan = plan.model_copy(update={
            "data_spec": ds,
            "iteration": plan.iteration + 1,
            "seed": (plan.seed or 42) + 1,
            "reasoning_trace": list(plan.reasoning_trace) + [
                f"Reflection refine: winner re-run with ensemble_n={ds['ensemble_n']} "
                "for tighter statistics.",
            ],
        })
        validator = get_validator(plan.domain)
        result = validator.run(
            run_id=state["run_id"],
            hypothesis=state["hypothesis"],
            plan=plan,
            datasets_dir=store.datasets_dir,
            logs_dir=store.logs_dir,
        )
        dump = result.model_dump()
        return {"plan": plan.model_dump(), "result": dump, "history": [dump]}

    def reflect(state: ValidationState) -> dict:
        reflector = ReflectionAgent(offline=state.get("offline", False))
        result = ValidationResult.model_validate(state["result"])
        iteration = state.get("iteration", 0)
        max_iters = state.get("max_iters", 2)
        refl = reflector.review(result, iteration, max_iters)
        refine = refl.decision == "refine"
        return {
            "reflections": [refl.model_dump()],
            "critique": refl.model_dump() if refine else None,
            "iteration": iteration + 1,
        }

    def route_after_reflect(state: ValidationState) -> str:
        if state.get("critique") and state.get("iteration", 0) < state.get("max_iters", 2):
            return "rerun_winner"
        return "link_and_persist"

    def link_and_persist(state: ValidationState) -> dict:
        result = ValidationResult.model_validate(state["result"])
        history = [ValidationResult.model_validate(r) for r in state.get("history", [])]
        reflections = [Reflection.model_validate(r) for r in state.get("reflections", [])]
        official = OfficialHypothesis.model_validate(state["official"])

        source_ids = list(result.artifact_paths.values())
        node_id = kg.add_validation_result(
            result_id=f"{state['run_id']}-{result.plan.domain}",
            domain=result.plan.domain,
            verdict=result.verdict.value,
            confidence=result.confidence,
            related_concept_ids=official.state_graph.related_concept_ids,
            source_ids=source_ids,
        )

        output = Layer3Output(
            run_id=state["run_id"],
            hypothesis_statement=state["hypothesis"],
            result=result,
            history=history,
            reflections=reflections,
            iterations=len(history),
            screening=state.get("screening"),
        )
        store.save_layer3(output, kg)
        screening = state.get("screening") or {}
        logger.info(
            "Layer 3 screening funnel complete: winner '%s', verdict %s "
            "(%d candidate(s) screened); KG node %s",
            screening.get("winner"), result.verdict.value,
            len(screening.get("rows", [])), node_id,
        )
        return {"output": output.model_dump()}

    builder = StateGraph(ValidationState)
    builder.add_node("run_screening", run_screening)
    builder.add_node("rerun_winner", rerun_winner)
    builder.add_node("reflect", reflect)
    builder.add_node("link_and_persist", link_and_persist)

    builder.add_edge(START, "run_screening")
    builder.add_edge("run_screening", "reflect")
    builder.add_conditional_edges(
        "reflect",
        route_after_reflect,
        ["rerun_winner", "link_and_persist"],
    )
    builder.add_edge("rerun_winner", "reflect")
    builder.add_edge("link_and_persist", END)

    return builder.compile()
