"""Typed artifacts shared across Layers 1 and 2.

These pydantic models are the contract between the source clients, the agents, the
knowledge graph, and the persistence layer. Keeping them strongly typed lets the LLM
return structured output that flows through the pipeline without ad-hoc dicts.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(text: str) -> str:
    """Stable, normalized identifier for a concept name (used for dedup)."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "unknown"


# ──────────────────────────── Citations ────────────────────────────


class Citation(BaseModel):
    """A normalized reference to a scholarly source."""

    id: str = Field(description="Stable provenance id, e.g. 'arxiv:2401.00001'")
    source: str = Field(description="Originating API: arxiv|openalex|crossref|pubmed|semantic_scholar|mock")
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    abstract: str | None = None
    citation_count: int | None = None

    @staticmethod
    def make_id(source: str, native_id: str) -> str:
        native = native_id.strip() or hashlib.sha1(f"{source}".encode()).hexdigest()[:10]
        return f"{source}:{native}"

    def short(self) -> str:
        author = self.authors[0] + " et al." if self.authors else "Unknown"
        year = f" ({self.year})" if self.year else ""
        return f"{author}{year}. {self.title}".strip()


# ──────────────────────────── Knowledge graph ────────────────────────────


class Concept(BaseModel):
    """A node in the knowledge graph."""

    id: str = Field(description="slug of the concept name")
    name: str
    type: str = Field(default="concept", description="entity type, e.g. gene, drug, method, disease, concept")
    description: str = ""
    domains: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list, description="Citation ids supporting this concept")

    @classmethod
    def new(cls, name: str, **kwargs) -> "Concept":
        return cls(id=slugify(name), name=name.strip(), **kwargs)


class Relation(BaseModel):
    """A directed, typed edge between two concepts."""

    source_id: str = Field(description="slug of source concept")
    target_id: str = Field(description="slug of target concept")
    relation: str = Field(default="related_to", description="relationship type, e.g. inhibits, causes, treats, associated_with")
    description: str = ""
    source_ids: list[str] = Field(default_factory=list, description="Citation ids supporting this relation")


class DomainSubgraph(BaseModel):
    """One research swarm's evolving state graph for a single domain."""

    domain: str
    keywords: list[str] = Field(default_factory=list)
    concepts: list[Concept] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    notes: str = ""


# ──────────────────────────── Hypotheses ────────────────────────────


class Evidence(BaseModel):
    """A single piece of supporting or contradicting evidence."""

    statement: str
    stance: Literal["supporting", "contradicting"] = "supporting"
    source_ids: list[str] = Field(default_factory=list)
    strength: float = Field(default=0.5, ge=0.0, le=1.0)


class HypothesisStateGraph(BaseModel):
    """The dedicated state graph attached to each hypothesis."""

    supporting_evidence: list[Evidence] = Field(default_factory=list)
    contradicting_evidence: list[Evidence] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    related_concept_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    references: list[str] = Field(default_factory=list, description="Citation ids")


class RankingScores(BaseModel):
    """The components of a hypothesis ranking."""

    evidence_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty: float = Field(default=0.0, ge=0.0, le=1.0)
    consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    composite: float = Field(default=0.0, ge=0.0, le=1.0)


class Hypothesis(BaseModel):
    """A competing scientific hypothesis with its state graph and reasoning trace."""

    id: str
    statement: str
    rationale: str = ""
    novelty_assessment: str = ""
    reasoning_trace: list[str] = Field(default_factory=list)
    state_graph: HypothesisStateGraph = Field(default_factory=HypothesisStateGraph)
    scores: RankingScores = Field(default_factory=RankingScores)

    @staticmethod
    def make_id(index: int) -> str:
        return f"H{index + 1:02d}"


# ──────────────────────────── Aggregate outputs ────────────────────────────


class KnowledgeGraphMetadata(BaseModel):
    num_concepts: int = 0
    num_relations: int = 0
    num_citations: int = 0
    domains: list[str] = Field(default_factory=list)
    density: float = 0.0
    concept_types: dict[str, int] = Field(default_factory=dict)
    generated_at: str = Field(default_factory=_now)


class ResearchProvenance(BaseModel):
    """Audit trail of how Layer 1 reached its conclusions."""

    idea: str
    run_id: str
    domains: list[str] = Field(default_factory=list)
    sources_queried: list[str] = Field(default_factory=list)
    reasoning_trace: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=_now)


class Layer1Output(BaseModel):
    """Complete persistent output of the Deep Research Engine."""

    run_id: str
    idea: str
    concepts: list[Concept] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    kg_metadata: KnowledgeGraphMetadata = Field(default_factory=KnowledgeGraphMetadata)
    provenance: ResearchProvenance | None = None

    def top(self, n: int = 5) -> list[Hypothesis]:
        return sorted(self.hypotheses, key=lambda h: h.scores.composite, reverse=True)[:n]


# ──────────────────────────── Layer 2 ────────────────────────────


class ResearcherDecision(BaseModel):
    """Structured capture of the human-in-the-loop choice."""

    action: Literal["select", "modify", "merge", "new", "quit"]
    selected_ids: list[str] = Field(default_factory=list)
    statement: str | None = Field(
        default=None, description="Edited/merged/new hypothesis statement"
    )
    notes: str = ""
    decided_at: str = Field(default_factory=_now)


class ASALDSpec(BaseModel):
    """Structured area-selective-ALD intervention committed by Layer 2.

    This is the ``Hypothesis`` object of the reference in-silico protocol: it fixes the
    growth / non-growth surfaces, the inhibitor + precursor scheme, and the target film,
    thickness, and selectivity threshold that flow into the surface builder, the
    selection agent, the reactivity engine, and the Layer-4 manuscript.
    """

    growth_surface: str = "a-SiO2"          # GS -- stays reactive
    non_growth_surface: str = "a-SiN"       # NGS -- passivated by the inhibitor
    inhibitor: str = "acetic acid"          # small-molecule inhibitor (SMI)
    precursor: str = "BDEAS"                # ALD precursor for the target film
    target_film: str = "SiOx"
    target_thickness_nm: float = 10.0
    target_selectivity: float = 0.90
    provenance_refs: list[str] = Field(
        default_factory=list, description="DOIs from the Layer-1 KG grounding this scheme"
    )


class OfficialHypothesis(BaseModel):
    """The official research hypothesis produced by Layer 2."""

    run_id: str
    statement: str
    origin: ResearcherDecision
    state_graph: HypothesisStateGraph = Field(default_factory=HypothesisStateGraph)
    source_hypothesis_ids: list[str] = Field(default_factory=list)
    asald: ASALDSpec | None = Field(
        default=None,
        description="Structured AS-ALD intervention parsed/derived from the committed hypothesis.",
    )
    finalized_at: str = Field(default_factory=_now)


# ──────────────────────────── Layer 3 — In-Silico Validation ────────────────────────────


class ValidationVerdict(str, Enum):
    """Outcome of computationally testing a hypothesis."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class SuccessCriterion(BaseModel):
    """A single quantitative pass/fail condition for a validation."""

    metric: str = Field(description="Name of the metric this criterion checks")
    operator: Literal["<", "<=", ">", ">=", "==", "!="] = ">"
    threshold: float = 0.0
    description: str = ""

    def evaluate(self, value: float) -> bool:
        ops = {
            "<": value < self.threshold,
            "<=": value <= self.threshold,
            ">": value > self.threshold,
            ">=": value >= self.threshold,
            "==": value == self.threshold,
            "!=": value != self.threshold,
        }
        return bool(ops[self.operator])


class ValidationMetric(BaseModel):
    """A single quantitative result produced by a validator."""

    name: str
    value: float
    unit: str = ""
    threshold: float | None = None
    passed: bool | None = None
    note: str = ""


class SurfaceFidelityReport(BaseModel):
    """One generated slab's site-density gate report (Deliverable #1)."""

    material: str
    site_density_per_nm2: float
    acceptance_band: list[float] = Field(default_factory=list)
    n_sites: int = 0
    area_nm2: float = 0.0
    passed: bool = False
    seed: int | None = None
    note: str = ""


class ValidationPlan(BaseModel):
    """The in-silico experiment designed for a hypothesis (ReAct output).

    For the AS-ALD co-scientist the plan carries the committed (inhibitor, precursor)
    pair and its literature adsorption-energy priors inside ``data_spec``; the
    ``surface_reactivity`` engine consumes it to run the ADR-009 protocol.
    """

    domain: str = Field(
        default="surface_reactivity",
        description="Validation domain (AS-ALD co-scientist routes to surface_reactivity)",
    )
    method: str = Field(default="", description="Concrete method/test to run")
    rationale: str = ""
    reasoning_trace: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    data_spec: dict[str, Any] = Field(
        default_factory=dict, description="Spec for the synthetic dataset/model to generate"
    )
    success_criteria: list[SuccessCriterion] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    seed: int = 42
    iteration: int = 0


class ValidationResult(BaseModel):
    """Quantitative evidence produced by a validator agent."""

    run_id: str
    hypothesis_statement: str
    plan: ValidationPlan
    metrics: list[ValidationMetric] = Field(default_factory=list)
    verdict: ValidationVerdict = ValidationVerdict.INCONCLUSIVE
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    narrative: str = ""
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    generated_at: str = Field(default_factory=_now)


class Reflection(BaseModel):
    """A Reflection-agent critique deciding whether to refine the experiment."""

    decision: Literal["accept", "refine"] = "accept"
    critique: str = ""
    suggested_adjustments: list[str] = Field(default_factory=list)


METHODOLOGY_CITATIONS = [
    "Parsons & Clark, Area-selective deposition, Chem. Mater. 2020, 32, 4920",
    "Tezsevin et al., Aniline SMI DFT+RSA, Langmuir 2023, 10.1021/acs.langmuir.2c03214",
    "Area-Selective ALD of Al2O3 with MSA inhibitor, 10.1021/acs.chemmater.4c02902",
    "Dehydroxylated amorphous silica slab models, PCCP 2025, 10.1039/D5CP01570G",
    "Fine-tuning foundation MLIPs (barrier underestimation), arXiv:2502.15582",
    "Seal et al., agentic Supervisor/Swarm/ReAct/Reflection, arXiv:2510.27130",
]


class Layer3Output(BaseModel):
    """Complete persistent output of the In-Silico Validation layer."""

    run_id: str
    hypothesis_statement: str
    result: ValidationResult
    history: list[ValidationResult] = Field(default_factory=list)
    reflections: list[Reflection] = Field(default_factory=list)
    iterations: int = 1
    screening: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Screening-funnel campaign summary (pool, per-candidate table, winner, "
            "recommendation) when SCREENING_MODE=funnel; None in legacy single mode."
        ),
    )
    agentic_pattern: str = "Supervisor + Swarm + ReAct + Reflection"
    methodology_citations: list[str] = Field(default_factory=lambda: list(METHODOLOGY_CITATIONS))
    generated_at: str = Field(default_factory=_now)
