"""LLM validation summarizer (ADR-007 support, runs at the end of Layer 3).

After the closed-loop validation completes, this module asks the configured LLM
(Gemini for ``google_genai`` runs) to write a detailed scientific summary of

* the amorphous base-layer model generation (slab source, passivation, per-site-type
  densities vs the Kim 2026 acceptance bands, fidelity-gate outcomes),
* the inhibitor/precursor selection reasoning (the designer's ranked screen), and
* every quantitative result with units (adsorption energies in eV, blocking coverages,
  selectivity at the target thickness in nm, calibration deltas, temperatures in K)
  plus the compute provenance (which MLIP/engine produced each number).

The summary is written to ``artifacts/<run_id>/validation_summary.md`` and is consumed
by the Layer-4 manuscript swarm as grounded prose context. It NEVER introduces new
numbers -- the prompt forbids it and the offline fallback is fully deterministic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..config import get_settings

logger = logging.getLogger(__name__)


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _payload(run_dir: Path) -> dict:
    """Compact, numbers-complete digest of the validation artifacts."""
    rich = _load(run_dir / "asald_results.json")
    fidelity = _load(run_dir / "surface_fidelity.json")
    plan = _load(run_dir / "validation_plan.json")
    ads = rich.get("inhibitor_adsorption", {})
    return {
        "hypothesis": rich.get("hypothesis", {}),
        "plan_method": plan.get("method"),
        "plan_reasoning_trace": plan.get("reasoning_trace", []),
        "plan_assumptions": plan.get("assumptions", []),
        "surface_fidelity": fidelity,
        "adsorption": {
            k: v for k, v in ads.items() if k not in ("configs_ngs", "configs_gs")
        },
        "adsorption_configs": {
            "ngs": ads.get("configs_ngs", []),
            "gs": ads.get("configs_gs", []),
        },
        "selectivity": rich.get("selectivity", {}),
        "calibration_vs_literature": rich.get("calibration_vs_literature"),
        "precursor_barrier": rich.get("precursor_barrier"),
        "verdict": rich.get("verdict"),
        "provenance": rich.get("provenance", {}),
    }


_SYSTEM = (
    "You are the results-summarization agent of an AS-ALD (area-selective atomic layer "
    "deposition) in-silico co-scientist. You are given the COMPLETE quantitative record "
    "of one validation run as JSON. Write a detailed, publication-grade markdown summary "
    "(600-1000 words) with these sections:\n"
    "## Amorphous surface model generation -- how the a-SiO2 (growth) and a-SiN "
    "(non-growth) slabs were built (slab source in provenance), the per-site-type "
    "densities (OH, O-bridge, NH2, NH-bridge, in nm^-2) against their acceptance bands, "
    "and the fidelity-gate outcomes.\n"
    "## Inhibitor/precursor selection -- which candidates were screened, why the chosen "
    "inhibitor/precursor pair was selected (use plan_reasoning_trace), and the site-"
    "matching logic.\n"
    "## Quantitative energetics and selectivity -- EVERY number with its unit: adsorption "
    "energies dE (eV) on NGS vs GS with the chemisorption (< -0.7 eV) and physisorption "
    "(> -0.3 eV) interpretation, blocking coverages, differential blocking, selectivity "
    "S at the target thickness (nm) with its mean +/- std, calibration-vs-literature "
    "error (eV) and validity flag, process temperature (K).\n"
    "## Compute provenance -- which engine/model computed the energies (e.g. the MACE-MP "
    "foundation machine-learning interatomic potential on cpu/cuda, or Tier-0 literature "
    "priors), ensemble size, seed.\n"
    "STRICT RULES: use ONLY numbers present in the JSON; never invent, extrapolate, or "
    "round beyond what is given; if a value is missing say 'not recorded'. If the "
    "calibration flag is 'review' or an energy sits at a clamp bound (-3.0 or +1.0 eV), "
    "state plainly that the energetics are flagged and why that limits the conclusion."
)


def _fallback_markdown(p: dict) -> str:
    """Deterministic summary used offline or when the LLM call fails."""
    hyp, ads, sel = p["hypothesis"], p["adsorption"], p["selectivity"]
    prov, calib = p["provenance"], p.get("calibration_vs_literature") or {}
    fid = p.get("surface_fidelity", {})
    lines = [
        "# Validation summary (deterministic fallback)",
        "",
        "## Amorphous surface model generation",
    ]
    for key, label in (("growth_surface", "Growth surface (GS)"),
                       ("non_growth_surface", "Non-growth surface (NGS)")):
        g = fid.get(key, {})
        lines.append(
            f"- **{label}** `{g.get('material','?')}`: mean terminal-site density "
            f"{g.get('site_density_mean','n/a')} nm^-2 (target "
            f"{g.get('target_density_per_nm2','n/a')} nm^-2, acceptance band "
            f"{g.get('acceptance_band','n/a')}); {g.get('n_passed','?')}/"
            f"{g.get('n_surfaces','?')} slabs passed the fidelity gate."
        )
        for rep in g.get("reports", []):
            sd = rep.get("site_densities", {})
            lines.append(
                f"  - seed {rep.get('seed')}: "
                + ", ".join(f"{k} = {v} nm^-2" for k, v in sd.items() if v)
                + f"; gate {'PASS' if rep.get('passed') else 'FAIL'}"
            )
    lines += [
        "",
        "## Inhibitor/precursor selection",
        f"- Committed hypothesis: {hyp.get('statement','n/a')}",
        f"- Tested pair: inhibitor **{hyp.get('inhibitor','n/a')}** / precursor "
        f"**{hyp.get('precursor','n/a')}** for target film {hyp.get('target_film','n/a')}.",
    ]
    lines += [f"- {step}" for step in p.get("plan_reasoning_trace", [])]
    lines += [
        "",
        "## Quantitative energetics and selectivity",
        f"- dE_ads(NGS) = {ads.get('dE_ngs_mean_eV','n/a')} +/- "
        f"{ads.get('dE_ngs_std_eV','n/a')} eV (chemisorption expected < -0.7 eV).",
        f"- dE_ads(GS) = {ads.get('dE_gs_mean_eV','n/a')} +/- "
        f"{ads.get('dE_gs_std_eV','n/a')} eV (physisorption expected > -0.3 eV).",
        f"- Blocking coverage: NGS {ads.get('blocking_ngs_mean','n/a')}, GS "
        f"{ads.get('blocking_gs_mean','n/a')}; differential blocking "
        f"{ads.get('differential_blocking','n/a')} (the selectivity driver).",
        f"- Selectivity S = {sel.get('S_at_target_mean','n/a')} +/- "
        f"{sel.get('S_at_target_std','n/a')} at {sel.get('target_thickness_nm','n/a')} nm "
        f"(target {sel.get('target','n/a')}); verdict **{p.get('verdict','n/a')}**.",
        f"- Calibration vs literature: predicted {calib.get('predicted_dE_ngs_eV','n/a')} eV "
        f"vs literature {calib.get('literature_dE_ngs_eV','n/a')} eV, |error| = "
        f"{calib.get('abs_error_eV','n/a')} eV, flag = {calib.get('validity_flag','n/a')}.",
        "",
        "## Compute provenance",
        f"- Engine {prov.get('engine','n/a')} (compute tier {prov.get('compute_tier','n/a')}, "
        f"MLIP {prov.get('mlip_model','n/a')} on {prov.get('mlip_device','n/a')}), "
        f"T = {prov.get('temperature_K','n/a')} K, ensemble N = "
        f"{prov.get('ensemble_n','n/a')}, seed {prov.get('seed','n/a')}.",
    ]
    if calib.get("validity_flag") == "review":
        lines.append(
            "\n> **Flag:** the MLIP-vs-literature calibration error exceeds the 0.3 eV "
            "acceptance threshold; the absolute energetics of this run should be treated "
            "as unreliable pending a denser adsorption search."
        )
    return "\n".join(lines)


def write_validation_summary(run_id: str, offline: bool = False) -> Path | None:
    """Write ``validation_summary.md`` for the run; returns the path (None on failure)."""
    settings = get_settings()
    run_dir = settings.artifacts_path / run_id
    if not (run_dir / "asald_results.json").exists():
        logger.warning("no asald_results.json for %s; skipping validation summary", run_id)
        return None
    payload = _payload(run_dir)

    text: str | None = None
    if not offline:
        try:
            from ..llm import get_llm

            resp = get_llm().invoke([
                ("system", _SYSTEM),
                ("human", json.dumps(payload, indent=1)),
            ])
            text = resp.content if hasattr(resp, "content") else str(resp)
            if isinstance(text, list):  # some providers return content blocks
                text = "\n".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in text
                )
            if not text or len(text.strip()) < 200:  # implausibly short -> fallback
                text = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM validation summary failed (%s); using fallback", exc)
            text = None
    if text is None:
        text = _fallback_markdown(payload)

    out = run_dir / "validation_summary.md"
    out.write_text(text, encoding="utf-8")
    logger.info("wrote validation summary to %s", out)
    return out
