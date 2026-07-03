"""Hypothesis generation agent.

Generates multiple competing scientific hypotheses grounded in the unified knowledge
graph and citation repository. Each hypothesis is returned with its own Hypothesis
State Graph: supporting evidence, contradicting evidence, assumptions, confidence, and
references (citation ids).
"""

from __future__ import annotations

import logging

from ..config import get_settings
from ..knowledge_graph import KnowledgeGraph
from ..llm import structured_call
from ..models import (
    Citation,
    Evidence,
    Hypothesis,
    HypothesisStateGraph,
    slugify,
)
from .schemas import HypothesisDrafts

logger = logging.getLogger(__name__)

_HYPO_SYSTEM = (
    "You are a hypothesis-generation agent for the Fidenz AI Co-scientist, an AS-ALD "
    "SCREENING co-scientist. Using the provided knowledge-graph concepts/relations and "
    "citations, propose multiple DISTINCT, competing, testable CAMPAIGN hypotheses for "
    "area-selective atomic layer deposition. Each hypothesis is a SCREENING hypothesis, "
    "NOT a bet on one molecule: it identifies the site-selective mechanism (which "
    "non-growth-surface sites the inhibitor must chemisorb -- e.g. a-SiN -NH2/-NH -- "
    "while leaving the growth-surface -OH free for the precursor) and proposes an "
    "inhibitor CLASS / functional-group family, named with a representative LEAD "
    "molecule, that an agentic funnel should select from a pool of ~40 candidates. Frame "
    "each as a screen, e.g. 'Among small-molecule inhibitors, a chlorosilane/silylamine "
    "family (lead: ETS) that chemisorbs a-SiN -NH2/-NH while sparing a-SiO2 -OH will be "
    "selected by the agentic funnel -- which screens the full candidate library (library "
    "+ literature-mined + AI-proposed) and recommends the best-selectivity member -- to "
    "enable BDEAS-based SiOx growth on a-SiO2 to >=90% selectivity at 10 nm'. For each "
    "hypothesis give supporting and contradicting evidence (tied to citation ids when "
    "possible), key assumptions, the related concepts it builds on, a brief reasoning "
    "trace, a novelty assessment, and a calibrated confidence in [0,1]. Hypotheses should "
    "differ in inhibitor-class chemistry, precursor, or target-surface pairing."
)


class HypothesisAgent:
    def __init__(self, offline: bool = False) -> None:
        self.offline = offline

    def generate(
        self, idea: str, kg: KnowledgeGraph, citations: list[Citation]
    ) -> list[Hypothesis]:
        settings = get_settings()
        n = settings.num_hypotheses
        if self.offline:
            return self._generate_heuristic(idea, kg, citations, n)

        user = self._build_prompt(idea, kg, citations, n)
        try:
            drafts = structured_call(HypothesisDrafts, _HYPO_SYSTEM, user)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM hypothesis generation failed: %s", exc)
            return self._generate_heuristic(idea, kg, citations, n)

        valid_concepts = {c.id for c in kg.concepts()}
        valid_refs = {c.id for c in citations}
        hypotheses: list[Hypothesis] = []
        for i, d in enumerate(drafts.hypotheses[:n]):
            if not d.statement.strip():
                continue
            related = [
                slugify(c) for c in d.related_concepts if slugify(c) in valid_concepts
            ]
            refs = [r for r in d.references if r in valid_refs]
            if not refs:
                refs = self._refs_for_concepts(related, kg)[:5]
            state = HypothesisStateGraph(
                supporting_evidence=[
                    Evidence(
                        statement=e.statement,
                        stance="supporting",
                        source_ids=[s for s in e.source_ids if s in valid_refs],
                        strength=e.strength,
                    )
                    for e in d.supporting_evidence
                ],
                contradicting_evidence=[
                    Evidence(
                        statement=e.statement,
                        stance="contradicting",
                        source_ids=[s for s in e.source_ids if s in valid_refs],
                        strength=e.strength,
                    )
                    for e in d.contradicting_evidence
                ],
                assumptions=d.assumptions,
                related_concept_ids=related,
                confidence=d.confidence,
                references=refs,
            )
            hypotheses.append(
                Hypothesis(
                    id=Hypothesis.make_id(i),
                    statement=d.statement.strip(),
                    rationale=d.rationale,
                    novelty_assessment=d.novelty_assessment,
                    reasoning_trace=d.reasoning_trace,
                    state_graph=state,
                )
            )
        return hypotheses or self._generate_heuristic(idea, kg, citations, n)

    # ──────────────────────── offline heuristic ────────────────────────

    def _generate_heuristic(
        self, idea: str, kg: KnowledgeGraph, citations: list[Citation], n: int
    ) -> list[Hypothesis]:
        concepts = kg.concepts()
        centrality = kg.degree_centrality()
        concepts.sort(key=lambda c: centrality.get(c.id, 0.0), reverse=True)
        if not concepts:
            concepts = []

        hypotheses: list[Hypothesis] = []
        templates = [
            "Among small-molecule inhibitors, a {a}-family candidate (screened by the "
            "agentic funnel from a ~40-candidate pool) selectively passivates the "
            "non-growth surface while {b} grows the target film on the growth surface, "
            "achieving >=90% selectivity at 10 nm for {idea}.",
            "An agentic screen of inhibitor candidates (lead: {a}) will identify the "
            "member whose chemisorption on the non-growth surface best blocks {b} "
            "precursor adsorption to reach the selectivity target for {idea}.",
            "{a}-class inhibitors paired with {b} form a candidate set whose differential "
            "adsorption, ranked by the funnel, drives area-selective growth in {idea}.",
            "The selectivity in {idea} is governed by the differential blocking coverage "
            "of {a} between the growth and non-growth surfaces rather than by {b} alone.",
            "{a} is a viable non-growth-surface passivant enabling {b}-based selective "
            "deposition in {idea}.",
        ]
        for i in range(min(n, max(1, len(concepts)))):
            a = concepts[i % len(concepts)] if concepts else None
            b = concepts[(i + 1) % len(concepts)] if len(concepts) > 1 else a
            a_name = a.name if a else idea
            b_name = b.name if b else "related factors"
            statement = templates[i % len(templates)].format(a=a_name, b=b_name, idea=idea)
            refs = sorted(set((a.source_ids if a else []) + (b.source_ids if b else [])))[:5]
            support = [
                Evidence(
                    statement=f"Literature links {a_name} to {idea}.",
                    stance="supporting",
                    source_ids=refs[:3],
                    strength=0.6,
                )
            ]
            contra = [
                Evidence(
                    statement=f"Some sources do not isolate {a_name}'s specific effect.",
                    stance="contradicting",
                    source_ids=refs[3:4],
                    strength=0.4,
                )
            ]
            related = [c.id for c in (a, b) if c]
            confidence = round(0.4 + 0.5 * centrality.get(a.id, 0.0) if a else 0.4, 3)
            hypotheses.append(
                Hypothesis(
                    id=Hypothesis.make_id(i),
                    statement=statement,
                    rationale=f"Derived from co-occurrence of {a_name} and {b_name} in the knowledge graph.",
                    novelty_assessment="Heuristic novelty estimate (offline mode).",
                    reasoning_trace=[
                        f"Selected central concept {a_name}.",
                        f"Paired with {b_name}.",
                        "Instantiated hypothesis template.",
                    ],
                    state_graph=HypothesisStateGraph(
                        supporting_evidence=support,
                        contradicting_evidence=contra,
                        assumptions=[f"{a_name} is measurable in the relevant system."],
                        related_concept_ids=related,
                        confidence=min(0.95, confidence),
                        references=refs,
                    ),
                )
            )
        return hypotheses

    # ──────────────────────── helpers ────────────────────────

    def _build_prompt(
        self, idea: str, kg: KnowledgeGraph, citations: list[Citation], n: int
    ) -> str:
        concepts = kg.concepts()[:40]
        relations = kg.relations()[:40]
        concept_lines = "\n".join(
            f"- {c.id}: {c.name} ({c.type})" for c in concepts
        )
        relation_lines = "\n".join(
            f"- {r.source_id} --{r.relation}--> {r.target_id}" for r in relations
        )
        cite_lines = "\n".join(
            f"- {c.id}: {c.short()}" for c in citations[:30]
        )
        return (
            f"Research idea: {idea}\n\n"
            f"Knowledge graph concepts:\n{concept_lines or '(none)'}\n\n"
            f"Knowledge graph relations:\n{relation_lines or '(none)'}\n\n"
            f"Available citations (use these ids as references):\n{cite_lines or '(none)'}\n\n"
            f"Generate up to {n} competing hypotheses."
        )

    @staticmethod
    def _refs_for_concepts(concept_ids: list[str], kg: KnowledgeGraph) -> list[str]:
        refs: list[str] = []
        by_id = {c.id: c for c in kg.concepts()}
        for cid in concept_ids:
            c = by_id.get(cid)
            if c:
                refs.extend(c.source_ids)
        # Stable de-dupe.
        seen, out = set(), []
        for r in refs:
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out
