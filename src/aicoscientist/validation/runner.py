"""Supervisor entry point for Layer 3.

Loads a run's approved hypothesis and knowledge graph, drives the bounded
design -> validate -> reflect -> refine loop via the Layer 3 LangGraph, links the result
back into the KG, and persists all artifacts. This is the perception/computation/action/
memory orchestration described in arXiv:2510.27130.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..config import get_settings
from ..knowledge_graph import KnowledgeGraph
from ..models import Concept, Layer3Output, OfficialHypothesis, Relation
from ..persistence import ArtifactStore

logger = logging.getLogger(__name__)


class ValidationDataError(RuntimeError):
    """Raised when the inputs required for Layer 3 are missing."""


def _run_dir(run_id: str) -> Path:
    return get_settings().artifacts_path / run_id


def load_official_hypothesis(run_id: str) -> OfficialHypothesis:
    path = _run_dir(run_id) / "official_hypothesis.json"
    if not path.exists():
        raise ValidationDataError(
            f"No official_hypothesis.json for run '{run_id}'. Run Layers 1-2 first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return OfficialHypothesis.model_validate(data)


def load_knowledge_graph(run_id: str) -> KnowledgeGraph:
    """Rebuild the KG from layer1_output.json (falls back to knowledge_graph.json)."""
    base = _run_dir(run_id)
    l1 = base / "layer1_output.json"
    if l1.exists():
        data = json.loads(l1.read_text(encoding="utf-8"))
        concepts = [Concept.model_validate(c) for c in data.get("concepts", [])]
        relations = [Relation.model_validate(r) for r in data.get("relations", [])]
        return KnowledgeGraph.from_concepts_relations(concepts, relations)

    kg_path = base / "knowledge_graph.json"
    if kg_path.exists():
        data = json.loads(kg_path.read_text(encoding="utf-8"))
        concepts = [Concept.model_validate(c) for c in data.get("concepts", [])]
        relations = [Relation.model_validate(r) for r in data.get("relations", [])]
        return KnowledgeGraph.from_concepts_relations(concepts, relations)

    logger.warning("no knowledge graph found for run %s; proceeding with empty KG", run_id)
    return KnowledgeGraph()


def run_validation(run_id: str, offline: bool = False) -> Layer3Output:
    """Execute Layer 3 for a completed run and return its output."""
    # lazy imports avoid a cycle
    from ..layer3_graph import build_layer3_graph, build_layer3_screening_graph

    settings = get_settings()
    official = load_official_hypothesis(run_id)
    kg = load_knowledge_graph(run_id)
    store = ArtifactStore(run_id)

    if settings.screening_mode.strip().lower() == "funnel":
        logger.info(
            "Layer 3 screening funnel: pool=%d, shortlist=%d, top_k=%d, tier=%d",
            settings.screen_pool_size, settings.screen_shortlist_m,
            settings.screen_top_k, settings.compute_tier,
        )
        graph = build_layer3_screening_graph(kg, store)
    else:
        graph = build_layer3_graph(kg, store)
    initial = {
        "run_id": run_id,
        "offline": offline,
        "hypothesis": official.statement,
        "official": official.model_dump(),
        "concept_names": [c.name for c in kg.concepts()],
        "iteration": 0,
        "max_iters": max(1, settings.max_validation_iters),
    }
    final = graph.invoke(initial)
    return Layer3Output.model_validate(final["output"])
