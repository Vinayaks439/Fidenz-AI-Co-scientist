"""Derive a structured :class:`ASALDSpec` from a committed hypothesis statement.

Layer 2 commits a free-text intervention hypothesis; Layer 3 needs the structured
(growth surface, non-growth surface, inhibitor, precursor, target film, thickness,
selectivity) tuple. This module maps the statement (plus KG concepts and provenance
DOIs) onto that tuple using surface-chemistry vocabulary, falling back to the ADR's
verified worked example so the pipeline always has a runnable spec.
"""

from __future__ import annotations

import re

from .models import ASALDSpec

# Known AS-ALD inhibitor vocabulary -> canonical library key. Matched on word boundaries
# so short tokens (e.g. "ets") do not hit substrings ("targets"). Includes the Kim et al.
# 2026 silylamine / chlorosilane inhibitors (DMATMS, ETS) and their spelled-out aliases,
# which the earlier list silently dropped -- so a committed "ETS" hypothesis actually maps
# onto the ETS spec instead of falling through to an unrelated molecule.
_INHIBITOR_ALIASES = {
    "octadecylphosphonic acid": "octadecylphosphonic acid",
    "methanesulfonic acid": "methanesulfonic acid",
    "phosphonic acid": "octadecylphosphonic acid",
    "pivalic acid": "pivalic acid",
    "ethylbutyric acid": "ethylbutyric acid",
    "acetic acid": "acetic acid",
    "carboxylic acid": "acetic acid",
    "aniline": "aniline",
    "dimethylamino-trimethylsilane": "dmatms",
    "dmatms": "dmatms",
    "ethyltrichlorosilane": "ets",
    "ethoxysilane": "ets",
    "ets": "ets",
}
_PRECURSORS = ["bdeas", "dipas", "hcds", "tdmat", "dmai", "tma"]

# Growth / non-growth surface synonyms.
_GROWTH = ["a-sio2", "sio2", "silica", "silicon oxide", "oxide growth"]
_NONGROWTH = ["a-sin", "sin", "sinx", "silicon nitride", "nitride", "-nh"]

_FILMS = {
    "siox": "SiOx", "sio2": "SiOx", "al2o3": "Al2O3", "tin": "TiN",
    "zro2": "ZrO2", "tio2": "TiO2", "hfo2": "HfO2",
}


def _canonical_inhibitor(text: str) -> str | None:
    # Match by earliest word-boundary occurrence, not list order, so the molecule the
    # author actually named wins over incidental mentions elsewhere in the corpus, and
    # short tokens like "ets" match "ETS" but not "targets".
    best: tuple[int, str] | None = None
    for alias, canonical in _INHIBITOR_ALIASES.items():
        m = re.search(r"\b" + re.escape(alias) + r"\b", text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), canonical)
    return best[1] if best else None


def _canonical_precursor(text: str) -> str | None:
    best: tuple[int, str] | None = None
    for name in _PRECURSORS:
        idx = text.find(name)
        if idx >= 0 and (best is None or idx < best[0]):
            best = (idx, name)
    return best[1].upper() if best else None


def _target_thickness_nm(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*nm", text)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:angstrom|å|a)\b", text)
    if m:
        return float(m.group(1)) / 10.0
    return None


def _target_selectivity(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        return float(m.group(1)) / 100.0
    m = re.search(r"selectivit\w*\s*(?:of|>=|>|=|~)?\s*(0?\.\d+)", text)
    if m:
        return float(m.group(1))
    return None


def _target_film(text: str) -> str | None:
    for key, val in _FILMS.items():
        if key in text:
            return val
    return None


def derive_asald_spec(
    statement: str,
    concept_names: list[str] | None = None,
    provenance_refs: list[str] | None = None,
) -> ASALDSpec:
    """Best-effort structured AS-ALD spec, defaulting to the verified worked example."""
    stmt = statement.lower()
    corpus = (statement + " " + " ".join(concept_names or [])).lower()
    defaults = ASALDSpec()  # the ADR worked example (acetic acid / BDEAS, a-SiO2/a-SiN)

    # Prefer the molecule named in the committed statement; fall back to KG concepts.
    inhibitor = _canonical_inhibitor(stmt) or _canonical_inhibitor(corpus) or defaults.inhibitor
    precursor = _canonical_precursor(stmt) or _canonical_precursor(corpus) or defaults.precursor

    growth = defaults.growth_surface
    non_growth = defaults.non_growth_surface
    if any(g in corpus for g in _GROWTH):
        growth = "a-SiO2"
    if any(n in corpus for n in _NONGROWTH):
        non_growth = "a-SiN"

    return ASALDSpec(
        growth_surface=growth,
        non_growth_surface=non_growth,
        inhibitor=inhibitor,
        precursor=precursor,
        target_film=_target_film(corpus) or defaults.target_film,
        target_thickness_nm=_target_thickness_nm(corpus) or defaults.target_thickness_nm,
        target_selectivity=_target_selectivity(corpus) or defaults.target_selectivity,
        provenance_refs=list(provenance_refs or []),
    )
