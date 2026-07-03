"""Layer 4 - Agentic LaTeX paper stitcher (ADR-007, capstone).

Assembles a full-length IEEE-style manuscript from the Layer-1->3 artifacts of a
completed run:

* Hypothesis, energetics, selectivity, provenance  <- ``asald_results.json``
* Surface fidelity (per-site densities vs bands)    <- ``surface_fidelity.json``
* Designer method/assumptions/reasoning trace       <- ``validation_plan.json``
* LLM validation digest (Gemini et al.)             <- ``validation_summary.md``
* Real-DOI bibliography                             <- ``citation_repository.json``
* Slab geometries for atomic-model figures          <- ``datasets/*.extxyz``

A LangGraph swarm of per-section writer agents (``swarm.py``) drafts the prose; the
figure agents render the full plot + atomic-model suite; tables and captions are built
deterministically from the artifacts so no number is ever LLM-generated. The compiler
agent builds the PDF via tectonic/latexmk/pdflatex, degrading to ``.tex`` source when
no toolchain is present. Nothing is invented.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import get_settings
from . import figures, sections, swarm
from .compiler import compile_pdf

logger = logging.getLogger(__name__)

_TEMPLATE = Path(__file__).parent / "template.tex"


class PaperDataError(RuntimeError):
    """Raised when the artifacts required to stitch a manuscript are missing."""


@dataclass
class PaperResult:
    run_id: str
    tex_path: Path
    pdf_path: Path | None
    figure_path: Path | None
    verdict: str
    figures: list[Path] = field(default_factory=list)


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


# ─────────────────────────── payload for the swarm ───────────────────────────


# Domain vocabulary that marks a mined citation as on-topic for AS-ALD / surface chemistry.
# Layer-1 mining can return off-domain papers for a generic term (e.g. "termination" hit
# pile-driving, quantum-loop verification, and theology papers); those pad the reference
# list with nonsense. Curated anchors (seed + hypothesis provenance) are always kept; the
# rest must match at least one of these to be cited.
_CITATION_DOMAIN_TERMS = (
    "atomic layer deposition", "area-selective", "area selective", "selective deposition",
    "as-ald", "asd of", "(ald", "ald)", " ald ", "ald of", "thin film",
    "passivat", "inhibitor", "precursor", "nucleation",
    "self-assembled monolayer", "silane", "silanol", "siloxane", "chlorosilane",
    "aminosilane", "silylamine", "silica", "silicon nitride", "silicon oxide",
    "sio2", "si3n4", "sinx", "siox", "dielectric",
    "hydroxylat", "amorphous silica", "amorphous silicon", "amorphous surface",
    "adsorption energ", "molecular adsorption", "chemisorption", "physisorption",
    "mace", "interatomic potential", "machine-learning potential",
    "machine learning potential", "foundation model", "melt-quench", "melt quench",
)


def _sanitize_cites(text: str, valid_keys: set) -> str:
    """Remove \\cite keys not present in the bibliography (they render as '[?]').

    A \\cite with only-invalid keys is dropped entirely; a mixed one keeps its valid
    keys. Prevents undefined-citation '[?]' marks from the LLM inventing keys."""
    if not text:
        return text

    def repl(m):
        keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
        keep = [k for k in keys if k in valid_keys or _safe_key_local(k) in valid_keys]
        return ("\\cite{" + ",".join(keep) + "}") if keep else ""

    return re.sub(r"\\cite\{([^}]*)\}", repl, text)


def _safe_key_local(cid: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", cid) or "ref"


def _strip_adr(text: str) -> str:
    """Remove internal 'ADR-NNN' design-record tags from rendered paper text.

    These are internal engineering references that should not appear in a
    publication. Handles parentheticals '(ADR-008)', '(ADR-004/009)', inline
    'ADR-009', and 'see ADR-007', then tidies the punctuation/space scars."""
    if not text:
        return text
    # Parentheticals that are purely an ADR tag (+ optional 'see').
    text = re.sub(r"\s*\((?:see\s+)?ADR-[0-9][0-9/,\s-]*\)", "", text)
    # Inline mentions, with an optional leading 'see '.
    text = re.sub(r"\s*(?:see\s+)?ADR-[0-9][0-9/-]*", "", text)
    # Tidy scars: space-before-punct, doubled punctuation/space.
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def _sanitize_refs(body: str) -> str:
    """Drop \\ref/\\eqref to labels that are not \\label-defined in the body.

    Safety net for LLM drift after we stop rendering some figures: an undefined
    \\ref prints '??' in the PDF. We strip the whole cross-reference clause
    (optional leading 'Fig.~'/'Table~'/'see ' and wrapping parens), then tidy the
    punctuation/whitespace scars, so the sentence still reads."""
    if not body:
        return body
    defined = set(re.findall(r"\\label\{([^}]+)\}", body))

    # 1) Parentheticals that are ENTIRELY cross-refs, at least one of them dangling:
    #    "(Fig.~\ref{a}, Fig.~\ref{b})" -> "" when every kept label would be gone.
    def _paren(m):
        inner = m.group(1)
        labels = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", inner)
        # only collapse if the parenthetical is purely refs (+ connective words)
        residue = re.sub(r"(?:Figs?\.?|Figures?|Tables?|Tab\.?|Secs?\.?|"
                         r"Sections?|see|and|,|~|\s|\\(?:ref|eqref)\{[^}]+\})+",
                         "", inner)
        if residue:
            return m.group(0)  # has real prose -> leave it
        kept = [l for l in labels if l in defined]
        if not kept:
            return ""
        rebuilt = ", ".join("Fig.~\\ref{" + l + "}" for l in kept)
        return "(" + rebuilt + ")"

    body = re.sub(r"\(([^()]*\\(?:ref|eqref)\{[^}]+\}[^()]*)\)", _paren, body)

    # 2) Inline dangling refs: strip an optional float word + the \ref itself.
    def _inline(m):
        return m.group(0) if m.group("lab") in defined else ""

    body = re.sub(
        r"(?:Figs?\.?|Figures?|Tables?|Tab\.?|Secs?\.?|Sections?)?~?\s*"
        r"\\(?:ref|eqref)\{(?P<lab>[^}]+)\}",
        _inline, body,
    )

    # 3) Tidy scars: doubled spaces, space-before-punct, empty parens, ", and".
    body = re.sub(r"\(\s*\)", "", body)
    body = re.sub(r"\s+([,.;:])", r"\1", body)
    body = re.sub(r"([,;]\s*){2,}", ", ", body)
    body = re.sub(r"\band\s*([,.])", r"\1", body)
    body = re.sub(r"[ \t]{2,}", " ", body)
    return body


def _is_on_domain(c: dict) -> bool:
    text = " ".join(str(c.get(k, "")) for k in ("title", "venue", "abstract")).lower()
    kws = c.get("keywords") or c.get("concepts") or []
    if isinstance(kws, list):
        text += " " + " ".join(str(k) for k in kws).lower()
    return any(t in text for t in _CITATION_DOMAIN_TERMS)


def _select_citations(citations: list[dict], hypothesis: dict, limit: int = 12) -> list[dict]:
    """Curated anchors + hypothesis provenance first, then on-domain mined refs, capped.

    Off-domain mined papers are dropped so the bibliography stays AS-ALD / surface-chemistry
    (the earlier version padded to the cap with whatever was mined, incl. unrelated
    'termination' hits)."""
    refs = set(hypothesis.get("provenance_refs", []))
    anchors = [c for c in citations
               if c.get("id") in refs or str(c.get("id", "")).startswith("seed_asald")]
    rest = [c for c in citations if c not in anchors]
    on_domain = [c for c in rest if _is_on_domain(c)]
    dropped = len(rest) - len(on_domain)
    if dropped:
        logger.info("citation filter: dropped %d off-domain mined reference(s)", dropped)
    seen, out = set(), []
    for c in anchors + on_domain:
        cid = c.get("id")
        if cid in seen:
            continue
        seen.add(cid)
        out.append(c)
        if len(out) >= limit:
            break
    return out


def _build_payload(run_id: str, rich: dict, fidelity: dict, plan: dict,
                   summary_md: str, cited: list[dict],
                   screening: dict | None = None) -> dict:
    ads = rich.get("inhibitor_adsorption", {})
    return {
        "run_id": run_id,
        "hypothesis": rich.get("hypothesis", {}),
        "verdict": rich.get("verdict"),
        "adsorption": {k: v for k, v in ads.items()
                       if k not in ("configs_ngs", "configs_gs")},
        "selectivity": rich.get("selectivity", {}),
        "calibration_vs_literature": rich.get("calibration_vs_literature"),
        "precursor_barrier": rich.get("precursor_barrier"),
        "provenance": rich.get("provenance", {}),
        "surface_fidelity": fidelity,
        "plan_method": plan.get("method"),
        "plan_assumptions": plan.get("assumptions", []),
        "plan_reasoning_trace": plan.get("reasoning_trace", []),
        "validation_summary_md": summary_md,
        "architecture_digest": sections.ARCH_DIGEST,
        # Screening-funnel campaign (winner deep-dive numbers are the rich dict above;
        # this carries the full comparative table + recommendation for the writers).
        "screening": screening,
        "citation_keys": [
            {"key": sections._safe_key(c.get("id", "ref")),
             "title": c.get("title", ""), "year": c.get("year")}
            for c in cited
        ],
    }


# ─────────────────────────── figure floats + captions ───────────────────────────


def _fig_block(name: str, label: str, caption: str, width: float = 1.0,
               star: bool = False) -> str:
    env = "figure*" if star else "figure"
    return (
        f"\\begin{{{env}}}[!t]\\centering\n"
        f"\\includegraphics[width={width:.2f}\\linewidth]{{{name}}}\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{{env}}}"
    )


def _render_figures(run_dir: Path, paper_dir: Path, rich: dict,
                    fidelity: dict,
                    screening: dict | None = None) -> tuple[dict[str, str], list[Path]]:
    """Render the figure suite; return ({section_key: latex_floats}, written_paths)."""
    hyp = rich.get("hypothesis", {})
    prov = rich.get("provenance", {})
    sel = rich.get("selectivity", {})
    gs = hyp.get("growth_surface", "a-SiO2")
    ngs = hyp.get("non_growth_surface", "a-SiN")
    inh = hyp.get("inhibitor", "inhibitor")
    esc = sections.latex_escape

    written: list[Path] = []
    blocks: dict[str, list[str]] = {"methods_surfaces": [], "methods_selection": [],
                                    "results": []}

    p = figures.slab_figure(run_dir, rich, paper_dir / "slabs.png")
    if p:
        written.append(p)
        blocks["methods_surfaces"].append(_fig_block(
            p.name, "fig:slabs",
            f"Atomic models of the amorphous surface slabs used in this run: "
            f"{esc(gs)} growth surface (top row) and {esc(ngs)} non-growth surface "
            "(bottom row), each in top view (left, growth face toward the reader) and "
            "side view (right, vacuum gap above the surface). Color code: Si beige, O "
            "red, N blue, H white. These are the exact gated geometries (or their "
            "deterministic seed-identical rebuilds) on which the Tier-1 adsorption "
            "search placed the inhibitor; terminal $-$OH/$-$NH$_2$ caps and bridge "
            "sites are visible on the exposed face.", star=True))

    p = figures.site_density_figure(fidelity, paper_dir / "site_densities.png")
    if p:
        written.append(p)
        blocks["methods_surfaces"].append(_fig_block(
            p.name, "fig:sites",
            "Measured per-site-type surface densities (bars, nm$^{-2}$) of the "
            "generated slabs against the Kim 2026 acceptance bands (green shading) "
            "used by the fidelity gate; dashed line marks the crystalline reference "
            "density, illustrating the $\\sim$35\\% terminal-site deficit that makes "
            "amorphous models mandatory. Left panel: growth surface; right panel: "
            "non-growth surface.", star=True))

    p = figures.molecule_figure(rich, paper_dir / "inhibitor.png")
    if p:
        written.append(p)
        blocks["methods_selection"].append(_fig_block(
            p.name, "fig:molecule",
            f"Three-dimensional conformer of the tested inhibitor, {esc(inh)}, as "
            "built by the validation engine (ETKDGv3 embedding with force-field "
            "cleanup) and used as the adsorbate in the Tier-1 search; two orthogonal "
            "views. The head group anchors to the surface site while the tail provides "
            "the steric blocking footprint used by the RSA coverage cap."))

    p = figures.energetics_figure(rich, paper_dir / "energetics.png")
    if p:
        written.append(p)
        blocks["results"].append(_fig_block(
            p.name, "fig:energetics",
            f"Mean inhibitor adsorption energies of {esc(inh)} on the non-growth "
            f"({esc(ngs)}, red) and growth ({esc(gs)}, green) surfaces with ensemble "
            "error bars, computed by "
            f"{esc(str(prov.get('engine', 'the reactivity engine')))}. Dashed and "
            "dotted lines mark the chemisorption ($<-0.7$ eV) and physisorption "
            "($>-0.3$ eV) regime thresholds; the star marks the literature anchor "
            "value on the NGS used for calibration. The chemisorb-on-NGS / "
            "physisorb-on-GS contrast is the mechanism of area selectivity."))

    p = figures.growth_curves_figure(rich, paper_dir / "growth_curves.png")
    if p:
        written.append(p)
        blocks["results"].append(_fig_block(
            p.name, "fig:growth",
            "Simulated film thickness versus ALD cycle on the growth surface (green) "
            "and non-growth surface (red), from the blocking-coverage "
            "nucleation-delay model. The horizontal line marks the "
            f"{sel.get('target_thickness_nm', 'n/a')} nm evaluation thickness; the "
            "lag of the red curve is the nucleation delay purchased by the inhibitor, "
            "and its eventual rise is the residual defect nucleation that makes "
            "selectivity finite."))

    if screening:
        p = figures.screening_funnel_figure(screening, paper_dir / "screening.png")
        if p:
            written.append(p)
            n_pool = (screening.get("config") or {}).get("pool_size", "N")
            blocks["results"].append(_fig_block(
                p.name, "fig:screening",
                f"Screening-funnel outcome over the {n_pool}-candidate inhibitor "
                "pool: computed area selectivity $S$ at the target thickness "
                "(ensemble mean $\\pm\\sigma$) for every candidate that reached the "
                "MLIP batch screen (light blue) or the full-fidelity top-$k$ re-run "
                "(dark blue), all scored on identical seed-shared gated slab "
                "ensembles. The recommended winner (orange) is the candidate the "
                "recommendation agent selected; the dashed line marks the committed "
                "selectivity target. Candidates eliminated at the Tier-0 prior rank "
                "carry no computed $S$ and appear only in "
                "Table~\\ref{tab:screening}."))

    p = figures.selectivity_figure(rich, paper_dir / "selectivity.png")
    if p:
        written.append(p)
        blocks["results"].append(_fig_block(
            p.name, "fig:selectivity",
            "Area selectivity $S$ versus growth-surface oxide thickness with the "
            f"{sections.pct(sel.get('target', 0.9))} target at "
            f"{sel.get('target_thickness_nm', 'n/a')} nm (dashed) and the "
            "surface-ensemble $\\pm\\sigma$ band (shading). $S$ is evaluated at the "
            "cycle where the growth-surface film reaches each thickness; the decay "
            "past the nucleation-delay breakthrough is the expected qualitative "
            "signature of area-selective growth."))

    return {k: "\n\n".join(v) for k, v in blocks.items() if v}, written


# ─────────────────────────── assembly ───────────────────────────

_SECTION_ORDER = [
    ("introduction", "Introduction", None),
    ("architecture", "System Architecture", "sec:architecture"),
    ("methods_surfaces", "Methods: Amorphous Surface Builder", "sec:surfaces"),
    ("methods_selection", "Methods: Agentic Inhibitor/Precursor Selection",
     "sec:selection"),
    ("methods_protocol", "Methods: In-Silico Testing Protocol", "sec:protocol"),
    ("results", "Results", "sec:results"),
    ("discussion", "Discussion", None),
    ("limitations", "Limitations", None),
    ("conclusion", "Conclusion", None),
    ("reproducibility", "Reproducibility and Provenance", "sec:reproducibility"),
]


def _assemble_body(drafts: dict[str, str], fig_blocks: dict[str, str],
                   rich: dict, fidelity: dict, run_id: str,
                   screening: dict | None = None,
                   hyp_table: str = "") -> str:
    tables_for = {
        "architecture": hyp_table,
        "methods_surfaces": sections.fidelity_table(fidelity),
        "results": "\n\n".join(t for t in (
            sections.screening_table(screening) if screening else "",
            sections.adsorption_table(rich),
            sections.calibration_table(rich)) if t),
        "reproducibility": sections.provenance_table(rich, run_id),
    }
    parts = []
    for key, title, label in _SECTION_ORDER:
        head = f"\\section{{{title}}}"
        if label:
            head += f"\\label{{{label}}}"
        chunk = [head, drafts.get(key, "")]
        if tables_for.get(key):
            chunk.append(tables_for[key])
        if fig_blocks.get(key):
            chunk.append(fig_blocks[key])
        parts.append("\n".join(c for c in chunk if c))
    return "\n\n".join(parts)


def stitch_paper(run_id: str, offline: bool = False) -> PaperResult:
    settings = get_settings()
    run_dir = settings.artifacts_path / run_id
    rich_path = run_dir / "asald_results.json"
    if not rich_path.exists():
        raise PaperDataError(
            f"No asald_results.json for run '{run_id}'. Run Layer 3 validation first."
        )
    rich = _load_json(rich_path)
    fidelity = _load_json(run_dir / "surface_fidelity.json")
    plan = _load_json(run_dir / "validation_plan.json")
    citations = _load_json(run_dir / "citation_repository.json").get("citations", [])
    screening = _load_json(run_dir / "screening_results.json") or None
    hyps = _load_json(run_dir / "hypothesis_state_graphs.json").get("hypotheses", [])
    official = _load_json(run_dir / "official_hypothesis.json")
    selected_ids = official.get("source_hypothesis_ids", []) if official else []
    hyp_table = sections.hypotheses_table(hyps, selected_ids)
    summary_md = ""
    if (run_dir / "validation_summary.md").exists():
        summary_md = (run_dir / "validation_summary.md").read_text(encoding="utf-8")

    paper_dir = run_dir / "manuscript"
    paper_dir.mkdir(parents=True, exist_ok=True)

    # Figure agents (real numbers/geometries only).
    fig_blocks, fig_paths = _render_figures(run_dir, paper_dir, rich, fidelity,
                                            screening=screening)

    # Section-writer swarm (LangGraph fan-out; deterministic fallback offline).
    cited = _select_citations(citations, rich.get("hypothesis", {}))
    payload = _build_payload(run_id, rich, fidelity, plan, summary_md, cited,
                             screening=screening)
    drafts = swarm.run_swarm(payload, offline=offline)

    body = _assemble_body(drafts, fig_blocks, rich, fidelity, run_id,
                          screening=screening, hyp_table=hyp_table)
    # Strip \cite keys that are not in the bibliography so IEEEtran doesn't print "[?]"
    # (the LLM occasionally cites a key we never provided). Also sanitize the abstract.
    valid_keys = {sections._safe_key(c.get("id", "")) for c in cited}
    body = _sanitize_cites(body, valid_keys)
    drafts["abstract"] = _sanitize_cites(drafts.get("abstract", ""), valid_keys)
    # Strip \ref to any float we did not render so IEEEtran doesn't print an
    # undefined-reference "??" (safety net for LLM drift).
    body = _sanitize_refs(body)
    # Remove internal ADR-NNN design-record tags from the published text.
    body = _strip_adr(body)
    drafts["abstract"] = _strip_adr(drafts.get("abstract", ""))

    hyp = rich.get("hypothesis", {})
    gs = sections.latex_escape(hyp.get("growth_surface", ""))
    ngs = sections.latex_escape(hyp.get("non_growth_surface", ""))
    precursor = sections.latex_escape(hyp.get("precursor", ""))
    film = sections.latex_escape(hyp.get("target_film", ""))
    if screening and screening.get("winner"):
        # Screening campaign: frame the paper around the two Challenge-4 deliverables
        # (the amorphous surface builder + the agentic inhibitor screen), not one molecule.
        n_scored = len([r for r in screening.get("rows", []) if r.get("S_mean") is not None])
        n_pool = (screening.get("config") or {}).get("pool_size") or n_scored
        title = (
            "The Fidenz AI Co-scientist for Area-Selective ALD: A Fidelity-Gated "
            f"Amorphous {gs}/{ngs} Surface Builder and an Agentic Screen of "
            f"{n_pool} Inhibitor Candidates for {precursor}-Based {film} Growth"
        )
    else:
        title = (
            "An Agentic In-Silico Co-Scientist for Area-Selective ALD: "
            f"Fidelity-Gated Amorphous {gs}/{ngs} Surfaces and Site-Matched "
            f"{sections.latex_escape(hyp.get('inhibitor', ''))} Passivation for "
            f"{precursor}-Based {film} Growth"
        )

    filled = _TEMPLATE.read_text(encoding="utf-8")
    for key, val in {
        "__TITLE__": title,
        "__DATE__": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "__RUN_ID__": sections.latex_escape(run_id),
        "__ABSTRACT__": drafts.get("abstract", ""),
        "__KEYWORDS__": drafts.get("keywords", ""),
        "__BODY__": body,
        "__BIBLIOGRAPHY__": sections.bibliography(cited),
    }.items():
        filled = filled.replace(key, val)

    tex_path = paper_dir / "manuscript.tex"
    tex_path.write_text(filled, encoding="utf-8")
    logger.info("wrote manuscript source to %s (%d chars, %d figures)",
                tex_path, len(filled), len(fig_paths))

    pdf_path = compile_pdf(tex_path)

    return PaperResult(
        run_id=run_id,
        tex_path=tex_path,
        pdf_path=pdf_path,
        figure_path=fig_paths[0] if fig_paths else None,
        verdict=rich.get("verdict", "inconclusive"),
        figures=fig_paths,
    )
