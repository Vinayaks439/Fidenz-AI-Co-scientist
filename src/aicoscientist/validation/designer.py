"""Agentic inhibitor/precursor selection agent (ADR-005, Deliverable #2).

The Layer-3 ReAct designer, specialized for AS-ALD. It:

1. Retrieves candidate inhibitors/precursors from the Layer-1 knowledge graph.
2. Ranks them against a human-editable ``selection_criteria.md`` (volatility,
   functional-group <-> site compatibility, chemisorb-vs-physisorb differential, sterics,
   removability), grounded in the committed :class:`ASALDSpec`.
3. Encodes the chosen ``(inhibitor, precursor)`` pair and its literature adsorption-energy
   priors into a runnable ``ValidationPlan`` for the ``surface_reactivity`` engine.
4. On a Reflection ``refine`` it advances to the next-ranked candidate pair (the swarm-style
   "keep exploring" behavior), bounded by ``MAX_VALIDATION_ITERS``.

Without an LLM key it uses a deterministic ranking so offline runs are fully reproducible.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ..config import get_settings
from ..models import (
    ASALDSpec,
    OfficialHypothesis,
    Reflection,
    SuccessCriterion,
    ValidationPlan,
)

logger = logging.getLogger(__name__)

VALID_DOMAINS = {"surface_reactivity"}

_VOL = {"low": 0.0, "medium": 0.5, "high": 1.0}

# Built-in fallback library, literature-informed. NGS numbers for a-SiN are estimates
# (published DFT uses metal/oxide NGS, e.g. aniline chemisorbs -3.59 eV on Ru, -2.17 eV
# on Co); GS (SiO2) physisorption numbers are taken from the cited DFT where available
# (aniline -0.57 eV, Langmuir 2023 10.1021/acs.langmuir.2c03214).
_DEFAULT_LIBRARY = {
    "inhibitors": {
        "acetic acid": {"dE_ngs": -1.00, "dE_gs": -0.20, "functional_group": "carboxylic acid", "volatility": "high", "removability": "high"},
        "pivalic acid": {"dE_ngs": -0.95, "dE_gs": -0.22, "functional_group": "carboxylic acid", "volatility": "high", "removability": "high"},
        "ethylbutyric acid": {"dE_ngs": -0.98, "dE_gs": -0.24, "functional_group": "carboxylic acid", "volatility": "high", "removability": "high"},
        "methanesulfonic acid": {"dE_ngs": -1.15, "dE_gs": -0.25, "functional_group": "sulfonic acid", "volatility": "medium", "removability": "medium"},
        "aniline": {"dE_ngs": -0.90, "dE_gs": -0.57, "functional_group": "aromatic amine", "volatility": "high", "removability": "high"},
        "octadecylphosphonic acid": {"dE_ngs": -1.30, "dE_gs": -0.30, "functional_group": "phosphonic acid", "volatility": "low", "removability": "low"},
        "dmatms": {
            "dE_ngs": -0.80, "dE_gs": -0.85, "functional_group": "aminosilane",
            "volatility": "high", "removability": "high",
            "site_reactivity": {
                "SiO2": {"OH": {"deltaEr_eV": -0.85, "Ea_eV": 0.48},
                         "O_bridge": {"deltaEr_eV": 0.64, "Ea_eV": 1.50}},
                "SiN": {"NH2": {"deltaEr_eV": -0.80, "Ea_eV": 1.34},
                        "NH_bridge": {"deltaEr_eV": -0.70, "Ea_eV": 1.54}},
            },
        },
        "ets": {
            "dE_ngs": -0.95, "dE_gs": -0.30, "functional_group": "chlorosilane",
            "volatility": "high", "removability": "high",
            "site_reactivity": {
                "SiO2": {"OH": {"deltaEr_eV": -0.30, "Ea_eV": 1.10},
                         "O_bridge": {"deltaEr_eV": 0.74, "Ea_eV": 1.46}},
                "SiN": {"NH2": {"deltaEr_eV": -0.95, "Ea_eV": 0.79},
                        "NH_bridge": {"deltaEr_eV": -0.85, "Ea_eV": 0.80}},
            },
        },
    },
    "precursors": {
        "BDEAS": {"target_film": "SiOx"},
        "DIPAS": {"target_film": "SiOx"},
        "HCDS": {"target_film": "SiOx"},
        "TDMAT": {"target_film": "TiN"},
        "TMA": {"target_film": "Al2O3"},
    },
}


_MERGE_KEYS = ("dE_ngs", "dE_gs", "functional_group", "volatility", "removability",
               "site_reactivity", "smiles")

# Precursor preferred reactive sites (Kim et al. 2026 screening protocol).
PRECURSOR_SITE_PREFS: dict[str, list[str]] = {
    "BDEAS": ["OH"],
    "DIPAS": ["OH"],
    "HCDS": ["OH"],
    "TMA": ["OH", "O_bridge"],
    "DMAI": ["OH", "O_bridge"],
    "TDMAT": ["NH2"],
}


def _load_manual_library(path: str) -> dict:
    """Parse the ```json``` candidate block from selection_criteria.md (human override)."""
    p = Path(path)
    if not p.exists():
        logger.info("selection_criteria.md not found at %s; no manual overrides", path)
        return {}
    text = p.read_text(encoding="utf-8")
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        logger.info("no json candidate block in %s; no manual overrides", path)
        return {}
    try:
        lib = json.loads(m.group(1))
        if isinstance(lib, dict):
            return lib
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to parse candidate block (%s); ignoring manual overrides", exc)
    return {}


def _load_kg_candidates(run_id: str) -> dict:
    """Load the literature-mined candidate library written by Layer 1 (kg_candidates.json)."""
    settings = get_settings()
    p = Path(settings.artifacts_path) / run_id / "kg_candidates.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("inhibitors", {}) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to read kg_candidates.json (%s)", exc)
        return {}


def _merge_libraries(manual: dict, kg: dict, run_id: str) -> tuple[dict, dict]:
    """Merge inhibitor priors with precedence set by PRIORS_SOURCE.

    Base is the built-in default; KG-mined and manual layers overlay it. In ``auto`` mode
    KG-mined values win; in ``manual`` mode selection_criteria.md wins. Returns the merged
    ``{inhibitors, precursors}`` library plus a ``provenance`` map recording, per inhibitor,
    which layer set ``dE_ngs`` and whether that value is extrapolated from another surface.
    """
    settings = get_settings()
    manual_inh = (manual or {}).get("inhibitors", {})
    default_inh = _DEFAULT_LIBRARY["inhibitors"]

    # Precedence: later entries override earlier ones for the same key.
    if settings.priors_source.lower() == "manual":
        layers = [("builtin", default_inh), ("kg-mined", kg), ("manual", manual_inh)]
    else:  # auto (default): literature (KG) wins over the shipped manual defaults
        layers = [("builtin", default_inh), ("manual", manual_inh), ("kg-mined", kg)]

    names = set().union(*(set(layer.keys()) for _, layer in layers))
    merged: dict[str, dict] = {}
    provenance: dict[str, dict] = {}
    for name in names:
        entry: dict = {}
        prov = {"dE_ngs_source": "builtin", "ngs_extrapolated": False, "source_ids": []}
        for src_name, layer in layers:
            src = layer.get(name)
            if not src:
                continue
            for k in _MERGE_KEYS:
                v = src.get(k)
                if v is not None:
                    entry[k] = v
                    if k == "dE_ngs":
                        prov["dE_ngs_source"] = src_name
                        prov["ngs_extrapolated"] = bool(src.get("ngs_extrapolated", False))
                        prov["source_ids"] = src.get("source_ids", prov["source_ids"])
        entry.setdefault("dE_ngs", -1.0)
        entry.setdefault("dE_gs", -0.2)
        merged[name] = entry
        provenance[name] = prov

    precursors = (manual or {}).get("precursors") or _DEFAULT_LIBRARY["precursors"]
    return {"inhibitors": merged, "precursors": precursors}, provenance


class ExperimentDesigner:
    """AS-ALD selection agent (kept named ExperimentDesigner for the Layer-3 graph)."""

    def __init__(self, offline: bool = False) -> None:
        self.offline = offline

    def design(
        self,
        official: OfficialHypothesis,
        concept_names: list[str] | None = None,
        prior_critique: Reflection | None = None,
        iteration: int = 0,
    ) -> ValidationPlan:
        settings = get_settings()
        concept_names = concept_names or []
        spec = official.asald or ASALDSpec()

        manual = _load_manual_library(settings.selection_criteria_path)
        kg = _load_kg_candidates(official.run_id)
        library, provenance = _merge_libraries(manual, kg, official.run_id)

        # Phase 2: let the generative agent invent NOVEL candidates and merge them in
        # (ranked below the known library; picked on later refine iterations). Tagged
        # ai-proposed + extrapolated so they can never be "supported" on priors alone.
        n_proposed = 0
        if getattr(settings, "use_inhibitor_proposer", False):
            n_proposed = self._merge_proposed(
                library, provenance, spec, concept_names
            )

        ranked = self._rank_inhibitors(library, spec, concept_names)
        # Choose the inhibitor + compute tier for this iteration. With the AI planner the
        # LLM co-scientist decides both from the deep research + prior reflection (cheap
        # Tier-0 screen vs. expensive real-MLIP confirmation); otherwise fall back to the
        # deterministic rank-``iteration`` schedule at the global compute tier.
        chosen_tier = settings.compute_tier
        planner_rationale = None
        if getattr(settings, "use_ai_planner", False):
            inhibitor, props, chosen_tier, planner_rationale = self._plan_iteration(
                spec, ranked, iteration, concept_names, prior_critique, settings
            )
        else:
            idx = min(iteration, len(ranked) - 1)
            inhibitor, props = ranked[idx]
        idx = next((i for i, (n, _) in enumerate(ranked) if n == inhibitor), 0)
        prov = provenance.get(inhibitor, {"dE_ngs_source": "builtin",
                                          "ngs_extrapolated": False, "source_ids": []})

        precursor, target_film = self._choose_precursor(library, spec)

        n_kg = len(kg)
        n_manual = len((manual or {}).get("inhibitors", {}))
        trace = [
            f"Prior sources merged (mode={settings.priors_source}): {n_kg} KG-mined, "
            f"{n_manual} manual, {len(_DEFAULT_LIBRARY['inhibitors'])} built-in defaults.",
            f"Retrieved {len(ranked)} inhibitor candidates"
            + (f" ({n_proposed} AI-proposed novel)" if n_proposed else "")
            + "; ranked by differential adsorption + volatility + removability.",
            f"Selected '{inhibitor}' (rank {idx + 1}); dE_ngs from '{prov['dE_ngs_source']}'"
            + (" [extrapolated from another NGS material]" if prov["ngs_extrapolated"] else "")
            + ".",
            f"Paired with precursor '{precursor}' for target film {target_film}.",
        ]
        if prov["source_ids"]:
            trace.append(f"dE_ngs supported by citations: {', '.join(prov['source_ids'])}.")
        if planner_rationale:
            trace.append(
                f"AI planner chose Tier-{chosen_tier} "
                f"({'real MLIP confirmation' if chosen_tier >= 1 else 'cheap Tier-0 screen'}): "
                f"{planner_rationale}"
            )
        if prior_critique and prior_critique.decision == "refine":
            trace.append(f"Refinement: {prior_critique.critique}")

        plan = ValidationPlan(
            domain="surface_reactivity",
            method=(
                f"AS-ALD differential-reactivity protocol (ADR-009): inhibitor "
                f"adsorption screen on {spec.non_growth_surface} (NGS) vs "
                f"{spec.growth_surface} (GS) over a gated surface ensemble, "
                f"blocking coverage -> nucleation delay -> S(N)."
            ),
            rationale=(
                f"'{inhibitor}' ({props.get('functional_group', 'n/a')}) is predicted to "
                f"chemisorb on the NGS and physisorb on the GS, giving a large differential "
                f"blocking coverage."
            ),
            reasoning_trace=trace,
            assumptions=[
                "Adsorption-energy priors are literature/xTB values; Tier-1 MLIP recomputes "
                "them within a single calculator/head/dtype.",
                "Only chemisorbed, purge-surviving inhibitor blocks the precursor.",
                "Selectivity is reported as mean +/- std over the surface ensemble.",
            ],
            seed=42 + iteration,
            iteration=iteration,
            data_spec={
                "inhibitor": inhibitor,
                "precursor": precursor,
                "growth_surface": spec.growth_surface,
                "non_growth_surface": spec.non_growth_surface,
                "target_film": target_film,
                "target_thickness_nm": spec.target_thickness_nm,
                "target_selectivity": spec.target_selectivity,
                "inhibitor_smiles": props.get("smiles"),
                "dE_ngs_eV": props["dE_ngs"],
                "dE_gs_eV": props["dE_gs"],
                "dE_prior_std": 0.08,
                "literature_dE_ngs_eV": props["dE_ngs"],
                "temperature_K": settings.ald_temperature_k,
                "dose_ratio": 1.0,
                "ensemble_n": settings.surface_ensemble_n,
                "compute_tier": chosen_tier,
                "provenance_refs": spec.provenance_refs,
                "prior_source": prov["dE_ngs_source"],
                "prior_extrapolated": prov["ngs_extrapolated"],
                "prior_source_ids": prov["source_ids"],
                "site_reactivity": props.get("site_reactivity"),
                "literature_Ea_ngs_eV": _literature_ea_ngs(props),
            },
            success_criteria=[
                SuccessCriterion(
                    metric="S_at_target", operator=">=", threshold=spec.target_selectivity,
                    description=f"selectivity meets the {spec.target_selectivity:.0%} target",
                ),
                SuccessCriterion(
                    metric="differential_blocking", operator=">=", threshold=0.5,
                    description="NGS is blocked substantially more than GS",
                ),
                SuccessCriterion(
                    metric="dE_ngs_mean_eV", operator="<", threshold=-0.7,
                    description="chemisorption on the non-growth surface",
                ),
            ],
        )
        return plan

    def _plan_iteration(self, spec, ranked, iteration, concept_names, prior_critique,
                        settings):
        """Ask the AI planner for (inhibitor, props, compute_tier, rationale).

        Robust to planner failure: on any error, falls back to the deterministic
        rank-``iteration`` pick at the global compute tier.
        """
        from ..agents.experiment_planner import ExperimentPlanner

        try:
            planner = ExperimentPlanner(offline=self.offline)
            decision = planner.plan(
                spec=spec,
                ranked=ranked,
                iteration=iteration,
                max_iters=max(1, settings.max_validation_iters),
                max_tier=settings.ai_planner_max_tier,
                default_tier=settings.compute_tier,
                concept_names=concept_names,
                prior_critique=prior_critique,
            )
            props = dict(ranked)[decision.inhibitor]
            return decision.inhibitor, props, int(decision.compute_tier), decision.rationale
        except Exception as exc:  # noqa: BLE001 -- never let planning break the loop
            logger.warning("AI planner unavailable (%s); deterministic pick", exc)
            idx = min(iteration, len(ranked) - 1)
            name, props = ranked[idx]
            return name, props, settings.compute_tier, None

    def _merge_proposed(self, library, provenance, spec, concept_names) -> int:
        """Generate novel candidates and merge them into ``library`` in-place.

        Returns the number of novel candidates added. Existing library names are never
        overwritten; provenance records them as ai-proposed / extrapolated.
        """
        try:
            from ..agents.inhibitor_proposer import InhibitorProposer, to_library_entries
        except Exception as exc:  # noqa: BLE001
            logger.warning("inhibitor proposer unavailable (%s)", exc)
            return 0

        settings = get_settings()
        inhibitors = library.setdefault("inhibitors", {})
        existing = set(inhibitors.keys())
        n = getattr(settings, "n_proposed_inhibitors", 3)
        try:
            proposer = InhibitorProposer(offline=self.offline)
            candidates = proposer.propose(
                spec, concept_names=concept_names, existing_names=existing,
                n=n, citations=list(spec.provenance_refs or []),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("inhibitor proposer failed (%s)", exc)
            return 0

        added = 0
        for name, entry in to_library_entries(candidates).items():
            if name in inhibitors:
                continue
            inhibitors[name] = entry
            provenance[name] = {
                "dE_ngs_source": "ai-proposed",
                "ngs_extrapolated": True,
                "source_ids": entry.get("citations", []),
            }
            added += 1
        if added:
            logger.info("proposer added %d novel inhibitor candidate(s)", added)
        return added

    # ──────────────────────── ranking ────────────────────────

    def _rank_inhibitors(
        self, library: dict, spec: ASALDSpec, concept_names: list[str]
    ) -> list[tuple[str, dict]]:
        inhibitors = library.get("inhibitors", {})
        kg_text = " ".join(concept_names).lower()

        def score(name: str, props: dict) -> float:
            differential = props["dE_gs"] - props["dE_ngs"]     # want large positive
            vol = _VOL.get(str(props.get("volatility", "medium")).lower(), 0.5)
            rem = _VOL.get(str(props.get("removability", "medium")).lower(), 0.5)
            s = differential + 0.2 * vol + 0.2 * rem
            if name.lower() in kg_text:      # grounded in the literature KG -> preferred
                s += 0.3
            # Site-matched screening (Kim et al. 2026): inhibitor should passivate
            # the sites the precursor uses on the NGS.
            s += _site_match_score(props, spec.precursor)
            if props.get("provenance") == "ai-proposed":
                s -= 0.5
            if name.lower() == spec.inhibitor.lower():  # honor the committed choice first
                s += 100.0
            return s

        ranked = sorted(inhibitors.items(), key=lambda kv: score(*kv), reverse=True)
        if not ranked:  # ensure at least the committed inhibitor is runnable
            ranked = [(spec.inhibitor, {"dE_ngs": -1.0, "dE_gs": -0.2,
                                        "functional_group": "n/a"})]
        return ranked

    @staticmethod
    def _choose_precursor(library: dict, spec: ASALDSpec) -> tuple[str, str]:
        precursors = library.get("precursors", {})
        if spec.precursor in precursors:
            return spec.precursor, precursors[spec.precursor].get("target_film", spec.target_film)
        # else pick a precursor matching the target film, defaulting to the committed one.
        for name, props in precursors.items():
            if props.get("target_film", "").lower() == spec.target_film.lower():
                return name, props["target_film"]
        return spec.precursor, spec.target_film


def _literature_ea_ngs(props: dict) -> float | None:
    """Extract a representative NGS Ea from site-resolved priors if present."""
    sr = props.get("site_reactivity") or {}
    sin = sr.get("SiN") or {}
    for st in ("NH2", "NH_bridge", "OH"):
        ea = (sin.get(st) or {}).get("Ea_eV")
        if ea is not None:
            return float(ea)
    return None


def _site_match_score(props: dict, precursor: str) -> float:
    """Bonus when inhibitor passivates precursor-preferred NGS sites (Kim et al. 2026)."""
    prefs = PRECURSOR_SITE_PREFS.get(precursor.upper(), ["NH2"])
    sr = (props.get("site_reactivity") or {}).get("SiN") or {}
    if not sr:
        return 0.0
    score = 0.0
    for st in prefs:
        p = sr.get(st) or sr.get(st.replace("_bridge", ""))
        if not p:
            continue
        delta = p.get("deltaEr_eV", 0.0)
        if delta < 0:
            score += 0.25
        elif delta > 0:
            score -= 0.15  # endothermic at a precursor site -> penalise
    return score
