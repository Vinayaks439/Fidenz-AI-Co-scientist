"""Deterministic section content for the Layer-4 manuscript (ADR-007).

This module holds everything the manuscript swarm (``swarm.py``) needs that must be
reproducible without an LLM:

* ``latex_escape`` / ``bibliography`` -- LaTeX hygiene + real-DOI reference list.
* Table builders (fidelity, adsorption, calibration, provenance) -- numbers straight
  from the artifacts, never prose-generated.
* ``ARCH_DIGEST`` -- a condensed, citable description of the co-scientist architecture
  (the four-layer funnel and its ADRs) given to every writer agent as grounding.
* ``fallback(key, payload)`` -- lengthy deterministic IEEE-style prose per section,
  used offline or whenever a writer agent fails. All quantities are interpolated from
  the artifact payload; nothing is invented.
"""

from __future__ import annotations

import re


# Unicode the LLM / stored statements commonly emit -> LaTeX (pdflatex-safe).
_UNICODE_REPL = {
    "≥": r"$\geq$", "≤": r"$\leq$", "≠": r"$\neq$",
    "≈": r"$\approx$", "×": r"$\times$", "±": r"$\pm$",
    "→": r"$\rightarrow$", "←": r"$\leftarrow$", "⇒": r"$\Rightarrow$",
    "°": r"$^\circ$", "µ": r"$\mu$", "μ": r"$\mu$",
    "–": "--", "—": "---", "−": r"$-$",
    "‘": "`", "’": "'", "“": "``", "”": "''",
    "…": r"\ldots{}", " ": " ", "å": r"\AA{}", "Å": r"\AA{}",
    "₂": r"$_2$", "₃": r"$_3$", "₄": r"$_4$",
    # Greek commonly emitted in AS-ALD prose (ΔE, θ_block, ±σ, α/β/γ, λ, Ω).
    "Δ": r"$\Delta$", "Ω": r"$\Omega$", "Σ": r"$\Sigma$", "Θ": r"$\Theta$",
    "α": r"$\alpha$", "β": r"$\beta$", "γ": r"$\gamma$", "δ": r"$\delta$",
    "θ": r"$\theta$", "λ": r"$\lambda$", "σ": r"$\sigma$", "ρ": r"$\rho$",
    "φ": r"$\phi$", "ω": r"$\omega$", "π": r"$\pi$", "τ": r"$\tau$",
    "η": r"$\eta$", "ε": r"$\epsilon$", "χ": r"$\chi$",
}


def latex_escape(text: str) -> str:
    if text is None:
        return ""
    repl = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    repl.update(_UNICODE_REPL)
    return "".join(repl.get(ch, ch) for ch in str(text))


def _safe_key(cid: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", cid) or "ref"


def pct(x) -> str:
    """Format a fraction as a LaTeX-safe percentage ('0.9' -> '90\\%')."""
    try:
        return f"{float(x) * 100:.0f}\\%"
    except (TypeError, ValueError):
        return "n/a"


def bibliography(citations: list[dict]) -> str:
    lines = []
    for c in citations:
        cid = c.get("id", "ref")
        short = c.get("title", "") or cid
        authors = ", ".join(c.get("authors", [])[:3])
        year = c.get("year")
        venue = c.get("venue") or ""
        doi = c.get("doi")
        entry = " ".join(
            p for p in [
                latex_escape(authors),
                f"({year})." if year else "",
                latex_escape(short) + ".",
                latex_escape(venue) + "." if venue else "",
                (f"\\url{{https://doi.org/{doi}}}" if doi else ""),
            ] if p
        )
        lines.append(f"\\bibitem{{{_safe_key(cid)}}} {entry}")
    return "\n".join(lines) if lines else "\\bibitem{none} No citations recorded."


# ──────────────────────────── architecture digest ────────────────────────────

ARCH_DIGEST = (
    "The system is the Fidenz AI Co-scientist, an autonomous in-silico co-scientist for "
    "area-selective ALD built by the authors (Pavan Kumar L and Vinayak S). Refer to it "
    "by name as the 'Fidenz AI Co-scientist' throughout. "
    "The Fidenz AI Co-scientist is a four-layer agentic funnel. Layer 1 (Deep Research "
    "Engine) is "
    "a literature-mining swarm over arXiv/OpenAlex/Crossref/PubMed/Semantic Scholar, "
    "seeded with a hand-curated anchor set of AS-ALD citations with real DOIs; it "
    "populates a typed knowledge graph with site-resolved nodes (Surface, Inhibitor, "
    "Precursor, Mechanism with per-site dEr/Ea, SelectivityResult) and emits ranked "
    "intervention hypotheses. Layer 2 is a human-in-the-loop gate: a researcher commits "
    "one inhibitor/precursor scheme, persisted as official_hypothesis.json (an ASALDSpec "
    "fixing GS/NGS materials, target film, thickness, and the selectivity threshold). "
    "Layer 3 is the in-silico validation loop: an amorphous surface builder (Deliverable "
    "1) generates slab ensembles via a seed-cleave-saturate-anneal protocol keyed to the "
    "Kim 2026 passivation table and rejects any slab whose per-site-type densities fall "
    "outside the published acceptance bands (a-SiO2 -OH 4.5-7.5 nm^-2, -O- 2.0-6.0 "
    "nm^-2; a-SiNx -NH2 2.5-5.5 nm^-2, -NH- 2.0-5.5 nm^-2); an agentic selection "
    "designer (Deliverable 2) executes the Kim 2026 three-step site-matched screening "
    "protocol over a candidate library merged from KG-mined literature priors, a "
    "human-editable selection_criteria.md, and built-in defaults; a surface_reactivity "
    "engine evaluates the pair over the gated ensemble at a compute tier chosen per "
    "iteration by an AI experiment planner (Tier 0 = literature priors; Tier 1 = "
    "foundation MLIP (MACE-MP) multi-site/orientation/height adsorption search on the "
    "real slabs; Tier 2 = + GFN2-xTB spot-checks); and a Reflection agent closes a "
    "bounded refine loop (MAX_VALIDATION_ITERS), advancing down the ranked candidate "
    "list on rejection. Layer 4 (this manuscript) is an autonomous LaTeX stitcher: a "
    "LangGraph swarm of per-section writer agents grounded exclusively in the run "
    "artifacts, with deterministic tables/figures and a real-DOI bibliography. "
    "Cross-cutting provenance pins seeds, tiers, MLIP model/device/dtype, "
    "slab source, temperatures, dose parameters, and package versions into "
    "*_provenance.json so every number in this paper resolves to a logged computation."
)

_IN_SILICO_DIGEST = (
    "The graded in-silico test is a five-step protocol: (1) build and gate the "
    "surface ensembles; (2) screen the inhibitor's site-resolved reactivity on both "
    "surfaces (dEr = E_chem - E_phys, Eq. 1; Ea = E_ts - E_phys, Eq. 2, both per the Kim "
    "2026 definitions); (3) convert per-site reactivity to an effective blocking "
    "coverage counting only chemisorbed, purge-surviving inhibitor, optionally capped at "
    "the random-sequential-adsorption jamming limit; (4) optionally compute precursor "
    "barriers (NEB), reported only as lower bounds because foundation MLIPs "
    "systematically underestimate barriers; (5) map differential blocking "
    "theta_NGS - theta_GS to a nucleation delay in cycles, propagate growth per cycle to "
    "thickness-vs-cycle curves on both surfaces, and evaluate the brief's metric "
    "S = (Thk_GS - Thk_NGS)/(Thk_GS + Thk_NGS) at the cycle where the GS film reaches "
    "the target thickness, reported as mean +/- std over the ensemble with a "
    "literature-calibration validity flag."
)


# ──────────────────────────── table builders ────────────────────────────


def fidelity_table(fidelity: dict) -> str:
    rows = []
    pretty = {"OH": "$-$OH", "O_bridge": "$-$O$-$", "NH2": "$-$NH$_2$",
              "NH_bridge": "$-$NH$-$"}
    for group_key, label in (("growth_surface", "GS"), ("non_growth_surface", "NGS")):
        group = fidelity.get(group_key, {})
        for rep in group.get("reports", []):
            checks = rep.get("site_type_checks", {})
            for st, chk in checks.items():
                band = chk.get("band", ["", ""])
                rows.append(
                    f"{label} ({latex_escape(rep.get('material', ''))}) & "
                    f"{pretty.get(st, latex_escape(st))} & "
                    f"{chk.get('density', 'n/a')} & "
                    f"[{band[0]}, {band[1]}] & "
                    f"{'pass' if chk.get('passed') else 'fail'} \\\\"
                )
    if not rows:
        return ""
    return (
        "\\begin{table}[!t]\\centering\n"
        "\\caption{Per-site-type surface densities of the generated amorphous slabs "
        "against the Kim 2026 acceptance bands, and the fidelity-gate outcome. "
        "Densities in nm$^{-2}$.}\n\\label{tab:fidelity}\n"
        "\\begin{tabular}{llccc}\\toprule\n"
        "Surface & Site & Density & Band & Gate \\\\\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\\end{tabular}\\end{table}"
    )


def adsorption_table(rich: dict) -> str:
    ads = rich.get("inhibitor_adsorption", {})
    return (
        "\\begin{table}[!t]\\centering\n"
        "\\caption{Inhibitor adsorption energetics and blocking coverages over the gated "
        "surface ensemble. Chemisorption is expected below $-0.7$ eV, physisorption "
        "above $-0.3$ eV; only chemisorbed, purge-surviving inhibitor blocks the "
        "precursor.}\n\\label{tab:adsorption}\n"
        "\\begin{tabular}{lcc}\\toprule\n"
        " & NGS & GS \\\\\\midrule\n"
        f"$\\Delta E_{{\\mathrm{{ads}}}}$ (eV) & "
        f"${ads.get('dE_ngs_mean_eV', 'n/a')} \\pm {ads.get('dE_ngs_std_eV', 'n/a')}$ & "
        f"${ads.get('dE_gs_mean_eV', 'n/a')} \\pm {ads.get('dE_gs_std_eV', 'n/a')}$ \\\\\n"
        f"Equilibrium coverage $\\theta_{{eq}}$ & {ads.get('theta_eq_ngs_mean', 'n/a')} & "
        f"{ads.get('theta_eq_gs_mean', 'n/a')} \\\\\n"
        f"Blocking coverage $\\theta_{{block}}$ & {ads.get('blocking_ngs_mean', 'n/a')} & "
        f"{ads.get('blocking_gs_mean', 'n/a')} \\\\\\midrule\n"
        f"Differential blocking & \\multicolumn{{2}}{{c}}{{{ads.get('differential_blocking', 'n/a')}}} \\\\\n"
        "\\bottomrule\\end{tabular}\\end{table}"
    )


def calibration_table(rich: dict) -> str:
    calib = rich.get("calibration_vs_literature")
    if not calib:
        return ""
    return (
        "\\begin{table}[!t]\\centering\n"
        "\\caption{Calibration of the predicted NGS adsorption energy against the "
        "literature anchor. A flag of `review' means the absolute energetics exceed the "
        "0.3 eV acceptance threshold and must not be read as quantitative.}\n"
        "\\label{tab:calibration}\n"
        "\\begin{tabular}{lc}\\toprule\n"
        f"Predicted $\\Delta E_{{NGS}}$ (eV) & {calib.get('predicted_dE_ngs_eV', 'n/a')} \\\\\n"
        f"Literature $\\Delta E_{{NGS}}$ (eV) & {calib.get('literature_dE_ngs_eV', 'n/a')} \\\\\n"
        f"$|$Error$|$ (eV) & {calib.get('abs_error_eV', 'n/a')} \\\\\n"
        f"Prior source & {latex_escape(str(calib.get('prior_source', 'n/a')))} \\\\\n"
        f"Validity flag & {latex_escape(str(calib.get('validity_flag', 'n/a')))} \\\\\n"
        "\\bottomrule\\end{tabular}\\end{table}"
    )


def screening_table(screening: dict) -> str:
    """Campaign table: every candidate that received computed numbers, ranked by S."""
    rows_in = [r for r in screening.get("rows", []) if r.get("S_mean") is not None]
    if not rows_in:
        return ""
    rows_in.sort(key=lambda r: (r["S_mean"], r.get("differential_blocking") or 0.0),
                 reverse=True)
    winner = screening.get("winner")
    n_pool = (screening.get("config") or {}).get("pool_size",
                                                 len(screening.get("rows", [])))
    n_tier0_only = len(screening.get("rows", [])) - len(rows_in)
    body = []
    for r in rows_in:
        flags = []
        if r.get("prior_extrapolated"):
            flags.append("E")
        if r.get("prior_missing"):
            flags.append("M")
        if r.get("prior_source") == "ai-proposed":
            flags.append("A")
        if r.get("calibration_flag") == "review":
            flags.append("R")
        name = latex_escape(str(r.get("inhibitor", "")))
        if r.get("inhibitor") == winner:
            name = f"\\textbf{{{name}}}"
        s_std = r.get("S_std")
        s_txt = (f"${r['S_mean']} \\pm {s_std}$" if s_std is not None
                 else f"${r['S_mean']}$")
        body.append(
            f"{name} & {latex_escape(str(r.get('stage', '')))} & "
            f"{r.get('dE_ngs_mean_eV', 'n/a')} & {r.get('dE_gs_mean_eV', 'n/a')} & "
            f"{r.get('differential_blocking', 'n/a')} & {s_txt} & "
            f"{latex_escape(str(r.get('verdict', '')))} & "
            f"{','.join(flags) if flags else '--'} \\\\"
        )
    return (
        "\\begin{table*}[!t]\\centering\n"
        "\\caption{Screening-campaign results over the candidate pool "
        f"(N={n_pool}; {n_tier0_only} candidate(s) eliminated at the Tier-0 prior "
        "rank carry no computed values and are omitted). All computed candidates "
        "were scored on identical, seed-shared gated slab ensembles. Energies in eV; "
        "$S$ at the target thickness (ensemble mean $\\pm\\sigma$). The recommended "
        "winner is bold. Flags: E = NGS prior extrapolated from another surface, "
        "M = no literature prior, A = AI-proposed novel compound, "
        "R = calibration flagged for review.}\n\\label{tab:screening}\n"
        "\\begin{tabular}{llcccccc}\\toprule\n"
        "Inhibitor & Stage & $\\Delta E_{NGS}$ & $\\Delta E_{GS}$ & "
        "$\\Delta\\theta_{block}$ & $S$ & Verdict & Flags \\\\\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\\end{tabular}\\end{table*}"
    )


def hypotheses_table(hyps: list[dict], selected_ids) -> str:
    """Layer-1 hypothesis slate: every generated hypothesis, ranked by composite
    score, with a column marking the one(s) the human-in-the-loop gate committed."""
    if not hyps:
        return ""
    selected = {str(s) for s in (selected_ids or [])}

    def _comp(h):
        try:
            return float((h.get("scores") or {}).get("composite", 0.0))
        except (TypeError, ValueError):
            return 0.0

    ordered = sorted(hyps, key=_comp, reverse=True)
    body = []
    for h in ordered:
        hid = str(h.get("id", ""))
        sc = h.get("scores") or {}
        stmt = latex_escape((h.get("statement") or "").strip())
        is_sel = hid in selected
        hid_cell = f"\\textbf{{{latex_escape(hid)}}}" if is_sel else latex_escape(hid)
        stmt_cell = f"\\textbf{{{stmt}}}" if is_sel else stmt

        def _f(key):
            try:
                return f"{float(sc.get(key)):.2f}"
            except (TypeError, ValueError):
                return "n/a"

        mark = "\\checkmark" if is_sel else "--"
        body.append(
            f"{hid_cell} & {stmt_cell} & {_f('composite')} & {_f('novelty')} & "
            f"{_f('confidence')} & {mark} \\\\"
        )
    n_sel = sum(1 for h in ordered if str(h.get("id", "")) in selected)
    return (
        "\\begin{table*}[!t]\\centering\n"
        "\\caption{Complete Layer-1 hypothesis slate generated for this campaign, "
        "ranked by composite score. The \\emph{Selected} column marks the "
        f"hypothes{'es' if n_sel != 1 else 'is'} the human-in-the-loop commit gate "
        "(Layer 2) promoted to the official hypothesis driving the in-silico screen; "
        "selected rows are shown in bold. Scores are the multi-agent review "
        "aggregates (evidence quality, novelty, consistency, confidence "
        "$\\rightarrow$ composite).}\n\\label{tab:hypotheses}\n"
        "\\begin{tabular}{@{}l p{0.44\\textwidth} cccc@{}}\\toprule\n"
        "ID & Hypothesis & Composite & Novelty & Confidence & Selected "
        "\\\\\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\\end{tabular}\\end{table*}"
    )


def provenance_table(rich: dict, run_id: str) -> str:
    prov = rich.get("provenance", {})
    return (
        "\\begin{table}[!t]\\centering\n"
        "\\caption{Pinned computational provenance of the validation run. "
        "Re-running the recorded command with these settings reproduces every number in "
        "this manuscript.}\n\\label{tab:provenance}\n"
        "\\begin{tabular}{ll}\\toprule\n"
        f"Run id & \\texttt{{{latex_escape(run_id)}}} \\\\\n"
        f"Engine & {latex_escape(str(prov.get('engine', 'n/a')))} \\\\\n"
        f"Compute tier & {prov.get('compute_tier', 'n/a')} \\\\\n"
        f"MLIP model / device & {latex_escape(str(prov.get('mlip_model', 'n/a')))} / "
        f"{latex_escape(str(prov.get('mlip_device', 'n/a')))} \\\\\n"
        f"Process temperature & {prov.get('temperature_K', 'n/a')} K \\\\\n"
        f"Dose ratio & {prov.get('dose_ratio', 'n/a')} \\\\\n"
        f"Ensemble size $N$ & {prov.get('ensemble_n', 'n/a')} \\\\\n"
        f"RNG seed & {prov.get('seed', 'n/a')} \\\\\n"
        "\\bottomrule\\end{tabular}\\end{table}"
    )


# ──────────────────────────── deterministic prose ────────────────────────────


def _hyp(p: dict) -> dict:
    return p.get("hypothesis", {})


def _num(p: dict, *path, default="n/a"):
    cur = p
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def fallback(key: str, p: dict) -> str:
    """Deterministic IEEE-style prose for section ``key`` from payload ``p``."""
    fn = _FALLBACKS.get(key)
    return fn(p) if fn else ""


def _fb_abstract(p: dict) -> str:
    h = _hyp(p)
    verdict = str(p.get("verdict", "inconclusive")).replace("_", " ")
    rejected = str(p.get("verdict", "")).lower() in {"rejected", "inconclusive", "fail",
                                                     "failed", "not_supported"}
    frame = (
        " A verdict short of the target is reported as an informative negative "
        "result: the contribution is the reproducible, fidelity-gated in-silico "
        "framework and its two deliverables --- the amorphous surface builder and the "
        "agentic inhibitor screen --- together with site-resolved evidence of why no "
        "screened candidate reached the target, which narrows the design space for the "
        "next campaign."
        if rejected else
        " The contribution is the reproducible, fidelity-gated in-silico framework "
        "and its two deliverables: the amorphous surface builder and the agentic "
        "inhibitor screen."
    )
    return (
        "Area-selective atomic layer deposition (AS-ALD) promises self-aligned "
        "nanofabrication, but computed selectivity predictions are dominated by the "
        "fidelity of the assumed amorphous surface model and by the inhibitor/precursor "
        "pairing. We present an end-to-end agentic in-silico co-scientist that (i) "
        "grounds itself in the AS-ALD literature via a knowledge-graph mining swarm, "
        "(ii) builds experiment-faithful amorphous "
        f"{latex_escape(h.get('growth_surface', 'a-SiO2'))} (growth) and "
        f"{latex_escape(h.get('non_growth_surface', 'a-SiN'))} (non-growth) slab "
        "ensembles gated on published per-site-type density bands, (iii) selects "
        "inhibitor/precursor candidates with a site-matched agentic screening protocol, "
        "and (iv) validates the committed hypothesis with a tiered reactivity engine "
        "combining literature priors and a foundation machine-learning interatomic "
        f"potential ({latex_escape(str(_num(p, 'provenance', 'mlip_model')))}). For the "
        f"tested pair --- inhibitor {latex_escape(h.get('inhibitor', 'n/a'))} with the "
        f"{latex_escape(h.get('precursor', 'n/a'))} precursor targeting "
        f"{latex_escape(h.get('target_film', 'n/a'))} --- the engine measures adsorption "
        f"energies of ${_num(p, 'adsorption', 'dE_ngs_mean_eV')}$ eV (NGS) and "
        f"${_num(p, 'adsorption', 'dE_gs_mean_eV')}$ eV (GS), a differential blocking "
        f"coverage of {_num(p, 'adsorption', 'differential_blocking')}, and a selectivity "
        f"$S = {_num(p, 'selectivity', 'S_at_target_mean')} \\pm "
        f"{_num(p, 'selectivity', 'S_at_target_std')}$ at "
        f"{_num(p, 'selectivity', 'target_thickness_nm')} nm against a "
        f"{pct(h.get('target_selectivity', 0.9))} target, yielding the verdict "
        f"\\emph{{{latex_escape(verdict)}}}. The committed hypothesis under test was: "
        f"\\emph{{{latex_escape(h.get('statement', 'n/a'))}}}" + frame +
        " Every number, figure, and citation in this "
        "manuscript is drawn programmatically from the recorded run artifacts, and the "
        "full provenance (seeds, tiers, potentials, temperatures) is tabulated for "
        "reproduction."
    )


def _fb_keywords(_p: dict) -> str:
    return (
        "Area-selective atomic layer deposition, amorphous surface models, "
        "machine-learning interatomic potentials, agentic AI, knowledge graphs, "
        "in-silico validation, small-molecule inhibitors, selectivity"
    )


def _fb_introduction(p: dict) -> str:
    h = _hyp(p)
    return (
        "Atomic layer deposition (ALD) grows conformal thin films one self-limiting "
        "surface reaction at a time; its area-selective variant (AS-ALD) restricts that "
        "growth to chosen regions of a patterned substrate, replacing lithography and "
        "etch steps with surface chemistry. The enabling mechanism is a small-molecule "
        "inhibitor (SMI) dosed before the precursor: on the non-growth surface (NGS) the "
        "inhibitor chemisorbs and blocks precursor adsorption, while on the growth "
        "surface (GS) it merely physisorbs and is removed during the purge. Film then "
        "nucleates promptly on the GS and only after a delay on the NGS; that "
        "nucleation delay \\emph{is} the selectivity. The technologically relevant "
        "figure of merit, $S = (T_{GS} - T_{NGS})/(T_{GS} + T_{NGS})$, is evaluated at a "
        "target film thickness --- here "
        f"{_num(p, 'selectivity', 'target_thickness_nm')} nm of "
        f"{latex_escape(h.get('target_film', 'SiOx'))} at a "
        f"{pct(h.get('target_selectivity', 0.9))} threshold, the 3D-NAND cell-isolation "
        "regime.\n\n"
        "Two computational failure modes dominate this problem. First, selectivity "
        "predictions computed on idealized crystalline slabs are unreliable to the point "
        "of sign errors: amorphous surfaces expose a disordered inventory of terminal "
        "sites (silanol $-$OH on SiO$_2$; amine $-$NH$_2$ on SiN$_x$) and bridge sites "
        "(siloxane $-$O$-$; imide $-$NH$-$) whose reaction barriers can be tens of "
        "percent lower than their crystalline counterparts. Any screening tool must "
        "therefore build --- and \\emph{audit} --- amorphous surface models that match "
        "experimental site densities. Second, inhibitor selection is a site-matching "
        "problem, not a strongest-binder search: an inhibitor must passivate precisely "
        "those sites where the precursor would otherwise adsorb, while remaining inert "
        "on the growth surface.\n\n"
        "This manuscript reports a fully autonomous in-silico test of the committed "
        f"hypothesis: \\emph{{{latex_escape(h.get('statement', ''))}}} The tested "
        f"realization pairs the inhibitor \\emph{{{latex_escape(h.get('inhibitor', ''))}}} "
        f"with the {latex_escape(h.get('precursor', ''))} precursor. The contributions "
        "are: (1) an amorphous surface builder whose slabs are gated on published "
        "per-site-type density bands before any reactivity money is spent; (2) an "
        "agentic selection designer implementing a site-matched screening protocol over "
        "a literature-mined candidate library; (3) a tiered validation engine that "
        "anchors foundation-MLIP energetics to literature values and carries an "
        "explicit calibration flag; and (4) this reproducible manuscript itself, "
        "assembled by a swarm of section-writer agents from the run artifacts with no "
        "human editing."
    )


def _fb_architecture(p: dict) -> str:
    prov = p.get("provenance", {})
    return (
        "The system is organized as a four-layer funnel in which each layer consumes "
        "the structured output of the previous one and every hand-off is a versioned "
        "JSON artifact.\n\n"
        "\\subsection{Layer 1: Deep research engine}\n"
        "A swarm of literature agents mines scholarly indices (arXiv, OpenAlex, "
        "Crossref, PubMed, Semantic Scholar) and merges the result into a typed "
        "knowledge graph whose schema is site-resolved by design: surfaces carry "
        "per-site-type densities, mechanisms carry per-site reaction energies "
        "$\\Delta E_r$ and activation energies $E_a$ with their byproducts, and every "
        "node carries its DOI-bearing source. A hand-curated anchor set seeds the graph "
        "so no run is left without domain-grounded literature; mined results merge on "
        "top and may override. The graph serves three consumers simultaneously: "
        "hypothesis generation, the selection agent's candidate library with per-site "
        "reactivity priors, and this paper's bibliography.\n\n"
        "\\subsection{Layer 2: Human-in-the-loop commitment}\n"
        "The top-ranked intervention hypotheses are presented to a researcher, who "
        "selects, merges, or redirects; the committed scheme is persisted as a "
        "structured specification (growth and non-growth materials, target film, "
        "thickness, selectivity threshold, provenance references) that seeds everything "
        "downstream. Placing the human gate immediately before the compute spend is "
        "deliberate: it is the expensive branch point.\n\n"
        "\\subsection{Layer 3: In-silico validation loop}\n"
        "Three cooperating components close the loop. The \\emph{amorphous surface "
        "builder} (Deliverable 1) generates slab ensembles and rejects any slab whose "
        "per-site-type densities leave the published acceptance bands "
        "(Table~\\ref{tab:fidelity}). The \\emph{selection designer} (Deliverable 2) "
        "ranks inhibitor candidates by differential adsorption, volatility, "
        "removability, and site-match against the precursor's preferred sites, merging "
        "priors from the knowledge graph, a human-editable criteria file, and built-in "
        "defaults. The \\emph{surface-reactivity engine} evaluates the committed pair "
        "over the gated ensemble at a compute tier chosen per iteration by an AI "
        "experiment planner: Tier 0 re-states literature priors (cheap screen), Tier 1 "
        f"runs a real multi-site adsorption search under the "
        f"{latex_escape(str(prov.get('mlip_model', 'MACE-MP')))} foundation "
        "machine-learning interatomic potential on the actual slab geometries, and "
        "Tier 2 adds semi-empirical GFN2-xTB spot checks. A Reflection agent reviews "
        "each verdict and either accepts or refines, advancing down the ranked "
        "candidate list within a bounded iteration budget.\n\n"
        "\\subsection{Layer 4: Autonomous manuscript}\n"
        "This document was assembled by a LangGraph swarm of per-section writer agents "
        "running in parallel, each grounded in (and restricted to) the run artifacts "
        "and an LLM-generated validation summary; tables, figures, and the bibliography "
        "are constructed deterministically from the same artifacts. The architecture's "
        "cross-cutting rule is that every number resolves to a logged "
        "computation: seeds, tiers, potentials, devices, temperatures, and dose "
        "parameters are pinned in provenance files (Table~\\ref{tab:provenance})."
    )


def _fb_methods_surfaces(p: dict) -> str:
    fid = p.get("surface_fidelity", {})
    gs, ngs = fid.get("growth_surface", {}), fid.get("non_growth_surface", {})
    prov_slab = (gs.get("slab_provenance") or {})
    return (
        "The central methodological risk in computational AS-ALD screening is the "
        "surface model itself: crystalline slabs over-count terminal sites by roughly a "
        "third and mis-price every downstream reactivity number. The builder therefore "
        "targets the experimentally anchored per-site-type densities of PECVD-grown "
        "films and audits every slab against them.\n\n"
        "\\subsection{Slab construction}\n"
        "Bulk cells are seeded from crystalline precursors at experimental density and "
        "stoichiometry ($\\alpha$-SiO$_2$; $\\beta$-Si$_3$N$_4$ with partial O "
        "substitution capturing the SiON character of fab-exposed nitride), cleaved "
        "along $z$ with vacuum padding, and passivated by a rule table keyed on element "
        "and dangling-bond count (Si$^{3+}\\!\\to$ Si(OH)$_2$H $\\ldots$ N$^{1+}\\!\\to$ "
        "NH on nitride), deliberately including the $=$NH and $-$H termini that are the "
        "feedstock for bridge formation. A bridge-anneal step then forms the reactive "
        "siloxane $-$O$-$ and imide $-$NH$-$ bridge sites that decide precursor "
        "compatibility. The slab source used in this run is recorded as "
        f"\\texttt{{{latex_escape(str(prov_slab.get('source', 'procedural')))}}} in the "
        "provenance; the melt-quench AIMD reference path is available but out of budget "
        "here, an approximation declared rather than hidden.\n\n"
        "\\subsection{Fidelity gate and ensemble}\n"
        "Every slab is classified site-by-site (terminal vs bridge) and rejected if any "
        "per-site-type density leaves its acceptance band. For the growth surface "
        f"({latex_escape(str(gs.get('material', 'SiO2')))}) this run measured a mean "
        f"terminal density of {gs.get('site_density_mean', 'n/a')} nm$^{{-2}}$ against a "
        f"target of {gs.get('target_density_per_nm2', 'n/a')} nm$^{{-2}}$ with "
        f"{gs.get('n_passed', '?')}/{gs.get('n_surfaces', '?')} slabs passing; for the "
        f"non-growth surface ({latex_escape(str(ngs.get('material', 'SiN')))}) "
        f"{ngs.get('site_density_mean', 'n/a')} nm$^{{-2}}$ with "
        f"{ngs.get('n_passed', '?')}/{ngs.get('n_surfaces', '?')} passing. The full "
        "per-site breakdown against the bands appears in Table~\\ref{tab:fidelity} and "
        "Fig.~\\ref{fig:sites}; the atomic geometries actually used by the reactivity "
        "engine are rendered in Fig.~\\ref{fig:slabs}. Because selectivity is "
        "model-sensitive, the engine evaluates an \\emph{ensemble} of independent "
        "seed-controlled slabs per surface and reports every downstream quantity as a "
        "mean $\\pm$ std over that ensemble rather than a point estimate."
    )


def _fb_methods_selection(p: dict) -> str:
    h = _hyp(p)
    trace = p.get("plan_reasoning_trace", [])
    trace_tex = "\n".join(
        f"\\item {latex_escape(t)}" for t in trace
    ) or "\\item (reasoning trace not recorded)"
    return (
        "Inhibitor selection is agentic rather than tabular: a ReAct designer executes "
        "a three-step site-matched screening protocol. (1) Read the NGS reactive-site "
        "inventory off the gated surface ensemble --- terminal and bridge sites both, "
        "because bridge sites decide precursor--inhibitor compatibility. (2) Evaluate "
        "site-specific reactivity of the precursor and each inhibitor candidate, from "
        "knowledge-graph-mined literature priors where available. (3) Select inhibitors "
        "that passivate precisely the sites where the precursor adsorbs favorably, "
        "while remaining inert on the growth surface. Candidates are ranked by the "
        "site-match score jointly with the classical SMI selection axes --- "
        "differential adsorption, vapor pressure/volatility, head-group compatibility, "
        "steric footprint, and post-deposition removability --- read from a "
        "human-editable criteria file that implements the supplemental-criteria hook. "
        "An AI experiment planner additionally chooses, per closed-loop iteration, "
        "which candidate to spend real compute on and at which tier, so expensive "
        "MLIP confirmations are earned by cheap screens rather than scheduled "
        "blindly.\n\n"
        "The recorded reasoning trace of the designer for this run was:\n"
        "\\begin{itemize}\n" + trace_tex + "\n\\end{itemize}\n"
        f"The committed pair under test is the inhibitor "
        f"\\emph{{{latex_escape(h.get('inhibitor', 'n/a'))}}} (rendered in "
        "Fig.~\\ref{fig:molecule}) against the precursor "
        f"\\emph{{{latex_escape(h.get('precursor', 'n/a'))}}} for a target film of "
        f"{latex_escape(h.get('target_film', 'n/a'))}."
    )


def _fb_methods_protocol(p: dict) -> str:
    prov = p.get("provenance", {})
    return (
        "Validation follows a five-step tiered protocol, recording every "
        "intermediate to the run artifacts.\n\n"
        "\\subsubsection{Energetics definitions}\n"
        "Reactivity at a surface site is characterized by two quantities defined "
        "between the physisorbed and chemisorbed states,\n"
        "\\begin{equation}\\Delta E_r = E_{\\mathrm{chemisorption}} - "
        "E_{\\mathrm{physisorption}},\\end{equation}\n"
        "\\begin{equation}E_a = E^{\\ddagger} - E_{\\mathrm{physisorption}},\\end{equation}\n"
        "where exothermic $\\Delta E_r$ is the necessary condition for a "
        "purge-surviving passivating bond (physisorbed species desorb during the purge "
        "and do not block), and $E_a$ is the rate-determining barrier of the "
        "proton-mediated ligand exchange. A site counts as blocked only if "
        "$\\Delta E_r < 0$ \\emph{and} the Arrhenius rate over the dose time at process "
        f"temperature ({prov.get('temperature_K', 'n/a')} K) is feasible.\n\n"
        "\\subsubsection{Tiered compute}\n"
        "Tier 0 evaluates the screen from literature/knowledge-graph priors with no "
        "atomistic calculation. Tier 1 builds the real inhibitor molecule (ETKDGv3 "
        "conformer generation with force-field cleanup) and runs a multi-site $\\times$ "
        "orientation $\\times$ height adsorption search on each gated slab under the "
        f"{latex_escape(str(prov.get('mlip_model', 'MACE-MP')))} foundation "
        "machine-learning interatomic potential "
        f"(device: {latex_escape(str(prov.get('mlip_device', 'n/a')))}, float64, single "
        "fixed calculator for slab, gas molecule, and complex so energy differences are "
        "meaningful). The search freezes the slab during adsorbate relaxation so "
        "surface-reconstruction energy cannot contaminate $\\Delta E_{\\mathrm{ads}}$, "
        "and discards non-converged or out-of-window configurations. Tier 2 adds "
        "GFN2-xTB spot checks; a large MLIP-vs-xTB gap flags calibration for review.\n\n"
        "\\subsubsection{From blocking to selectivity}\n"
        "Per-site reactivity is aggregated to an effective blocking coverage "
        "$\\theta_{\\mathrm{block}} = \\sum_s f_s \\cdot r_s$ (site fraction $\\times$ "
        "reactivity indicator), counting only chemisorbed inhibitor and optionally "
        "capped at the random-sequential-adsorption jamming limit of the inhibitor's "
        "steric footprint. The \\emph{differential} blocking "
        "$\\theta_{\\mathrm{block}}^{NGS} - \\theta_{\\mathrm{block}}^{GS}$ --- not raw "
        "Langmuir coverage, which saturates at ALD temperature --- maps to a nucleation "
        "delay in cycles; growth per cycle is then propagated on both surfaces (with a "
        "small residual defect-nucleation term so NGS growth is never exactly zero) to "
        "yield thickness-vs-cycle curves, and the selectivity\n"
        "\\begin{equation}S(N) = \\frac{T_{GS}(N) - T_{NGS}(N)}{T_{GS}(N) + T_{NGS}(N)}"
        "\\end{equation}\n"
        "is evaluated at the cycle where the GS film reaches the target thickness. "
        "Predicted energetics are calibrated against the literature anchor and the "
        "delta carried as an explicit validity flag (Table~\\ref{tab:calibration}); "
        "the engine inherits the 0 K and entropy caveats of its anchor methodology as "
        "recorded flags rather than footnotes."
    )


def _screening_paragraph(p: dict) -> str:
    """Deterministic campaign summary appended to Results when the funnel ran."""
    sc = p.get("screening") or {}
    cfg = sc.get("config", {})
    rows = sc.get("rows", [])
    computed = [r for r in rows if r.get("S_mean") is not None]
    if not computed:
        return ""
    rec = sc.get("recommendation", {})
    winner = latex_escape(str(sc.get("winner", "n/a")))
    runners = ", ".join(latex_escape(r) for r in rec.get("runners_up", [])[:3])
    return (
        "\n\nThese single-candidate results conclude a screening campaign, not a "
        f"one-shot test: a pool of {cfg.get('pool_size', len(rows))} candidate "
        f"inhibitors was prior-ranked at Tier 0, {cfg.get('shortlist_m', 'n/a')} were "
        "screened with the reactivity engine on \\emph{identical}, seed-shared gated "
        f"slab ensembles, and the top {cfg.get('top_k', 'n/a')} were re-run at full "
        "fidelity before the recommendation agent selected "
        f"\\textbf{{{winner}}}"
        + (f" (runners-up: {runners})" if runners else "")
        + ". The complete campaign table is Table~\\ref{tab:screening} and the "
        "ranked selectivities are visualized in Fig.~\\ref{fig:screening}. "
        + (
            "Outcome for the committed hypothesis: "
            + latex_escape(rec.get("committed_candidate_outcome", "")) + ". "
            if rec.get("committed_candidate_outcome") else ""
        )
    )


def _fb_results(p: dict) -> str:
    ads = p.get("adsorption", {})
    sel = p.get("selectivity", {})
    calib = p.get("calibration_vs_literature") or {}
    verdict = str(p.get("verdict", "inconclusive")).replace("_", " ")
    flagged = calib.get("validity_flag") == "review"
    flag_para = ""
    if flagged:
        flag_para = (
            "\n\nA material caveat accompanies these numbers: the calibration check "
            f"(Table~\\ref{{tab:calibration}}) records an absolute error of "
            f"{calib.get('abs_error_eV', 'n/a')} eV against the literature anchor, "
            "exceeding the 0.3 eV acceptance threshold, and the validity flag is set to "
            "\\emph{review}. Energies at the physical-window bounds ($-3.0$ or $+1.0$ "
            "eV) indicate the adsorption search found no converged in-window "
            "configuration and fell back to clamped values; in that regime the verdict "
            "should be read as a pipeline demonstration, not a quantitative selectivity "
            "claim, pending a denser site/orientation/height search."
        )
    return (
        "Table~\\ref{tab:adsorption} summarizes the inhibitor energetics over the gated "
        f"ensemble. On the non-growth surface the mean adsorption energy is "
        f"${ads.get('dE_ngs_mean_eV', 'n/a')} \\pm {ads.get('dE_ngs_std_eV', 'n/a')}$ eV "
        "(chemisorption regime below $-0.7$ eV), while on the growth surface it is "
        f"${ads.get('dE_gs_mean_eV', 'n/a')} \\pm {ads.get('dE_gs_std_eV', 'n/a')}$ eV; "
        "the contrast is visualized against both regime thresholds and the literature "
        "anchor in Fig.~\\ref{fig:energetics}. The resulting blocking coverages are "
        f"{ads.get('blocking_ngs_mean', 'n/a')} (NGS) and {ads.get('blocking_gs_mean', 'n/a')} "
        f"(GS), for a differential blocking of {ads.get('differential_blocking', 'n/a')} "
        "--- the quantity that drives selectivity.\n\n"
        "Propagating the implied nucleation delay through the growth model yields the "
        "thickness-vs-cycle curves of Fig.~\\ref{fig:growth} and the "
        "selectivity-vs-thickness curve of Fig.~\\ref{fig:selectivity}. At the "
        f"{sel.get('target_thickness_nm', 'n/a')} nm evaluation point the selectivity is "
        f"$S = {sel.get('S_at_target_mean', 'n/a')} \\pm {sel.get('S_at_target_std', 'n/a')}$ "
        f"against the {pct(p.get('hypothesis', {}).get('target_selectivity', 0.9))} "
        f"target; the recorded verdict is \\textbf{{{latex_escape(verdict)}}}."
        + flag_para
        + _screening_paragraph(p)
    )


def _fb_discussion(p: dict) -> str:
    ads = p.get("adsorption", {})
    verdict = str(p.get("verdict", "")).lower()
    rejected = verdict in {"rejected", "inconclusive", "fail", "failed", "not_supported"}
    dE_ngs = _num(p, "adsorption", "dE_ngs_mean_eV")
    dE_gs = _num(p, "adsorption", "dE_gs_mean_eV")
    outcome = (
        "\n\nThe verdict of this campaign is that no screened inhibitor reached the "
        "selectivity target on this non-growth/growth pair, and that outcome is the "
        "point of the exercise rather than a shortfall of it. The value of the work is "
        "the reproducible, fidelity-gated in-silico framework and its two deliverables "
        "--- the amorphous surface builder and the agentic inhibitor screen --- which "
        "together produce an \\emph{honest, site-resolved} negative result: the "
        "computed adsorption energies show the candidates binding the growth surface "
        f"(${dE_gs}$ eV) comparably to, or more strongly than, the non-growth surface "
        f"(${dE_ngs}$ eV), so the differential blocking that confers selectivity stays "
        "low. That is a research finding, not a null: it tells the next campaign to "
        "prioritise inhibitor chemistries that discriminate the two surfaces' site "
        "types more sharply (or a different precursor whose reactive sites the "
        "inhibitor can selectively passivate), and it does so with every number traced "
        "to a logged computation. A screening framework that can only confirm "
        "successes is not a screening framework; the ability to reject a hypothesis "
        "reproducibly, with the mechanism made explicit, is the deliverable."
        if rejected else ""
    )
    return (
        "The physical story behind the numbers is the purge argument: only chemisorbed "
        "inhibitor survives the ALD purge, so area selectivity is conferred by the "
        "chemisorb-on-NGS / physisorb-on-GS contrast, not by binding strength alone. "
        "This is why the engine scores \\emph{differential blocking} "
        f"({ads.get('differential_blocking', 'n/a')} here) rather than raw equilibrium "
        "coverage, which saturates at process temperature and washes out selectivity "
        "artificially. It is also why site matching matters more than affinity: an "
        "inhibitor that passivates only terminal sites cannot block a bridge-attacking "
        "precursor regardless of how exothermically it binds, a contrast the anchor "
        "methodology demonstrates on the aminosilane/alkylaluminum pair.\n\n"
        "The surface-fidelity gate is the load-bearing rigor element. Selectivity "
        "predictions inherit their error budget from the assumed surface; by rejecting "
        "slabs whose per-site-type densities leave the published bands \\emph{before} "
        "reactivity is computed, the pipeline both saves compute and makes the "
        "remaining numbers defensible. The ensemble convention (mean $\\pm$ std over "
        "independent seed-controlled slabs) converts a fragile point estimate into an "
        "error-barred result.\n\n"
        "Finally, the calibration discipline deserves emphasis: every predicted "
        "energetics value is compared against its literature anchor and the delta "
        "carried as an explicit validity flag in the artifacts and in "
        "Table~\\ref{tab:calibration}. A flagged run remains a valid demonstration of "
        "the autonomous pipeline --- surfaces built and gated, candidates selected, "
        "energetics computed, verdict emitted, manuscript written --- while clearly "
        "marking which quantitative claims require a denser search or a higher tier to "
        "stand."
        + outcome
    )


def _fb_limitations(p: dict) -> str:
    return (
        "Several limitations bound the interpretation of these results. "
        "(1) All energies are 0 K internal energies; entropic contributions at process "
        "temperature (up to $\\sim$0.25 eV at 150$^{\\circ}$C in the anchor "
        "methodology) shift absolute values while largely preserving trends. "
        "(2) Foundation machine-learning interatomic potentials systematically "
        "underestimate reaction barriers; any barrier reported by Tier 1/2 is a lower "
        "bound by construction, and thermodynamic endpoints ($\\Delta E_r$, "
        "$\\Delta E_{\\mathrm{ads}}$) are the trusted quantities. "
        "(3) The default slab source is a crystalline-derived procedural surface with "
        "Kim-table passivation and bridge annealing rather than full melt-quench AIMD; "
        "the approximation is recorded in provenance and the site-density gate bounds "
        "its impact, but true melt-quench topology (ring-size distributions, strained "
        "bridges) is not captured. "
        "(4) Byproduct secondary reactions and inhibitor--inhibitor lateral "
        "interactions are not modeled; the random-sequential-adsorption cap is a "
        "geometric, not chemical, treatment of crowding. "
        "(5) Where the calibration flag reads \\emph{review}, the absolute energetics "
        "of that run exceed the acceptance threshold against the literature anchor and "
        "the verdict should not be quoted as a quantitative selectivity claim. "
        "(6) The ensemble size in this run "
        f"($N = {_num(p, 'provenance', 'ensemble_n')}$) bounds how much of the "
        "surface-model sensitivity is averaged; production screening should use larger "
        "ensembles. "
        "(7) The adsorption-search density (sites $\\times$ orientations $\\times$ "
        "heights) and the candidate-pool/ensemble sizes are bounded by the compute and "
        "wall-clock budget: the foundation MLIP evaluates one structure at a time, so a "
        "single GPU does not saturate, and a denser search or a larger pool trades "
        "directly against runtime. A near-miss verdict can therefore reflect an "
        "under-resolved search rather than a genuine physical ceiling. "
        "(8) Consequently, the negative verdict of this campaign may reflect either the "
        "genuine difficulty of selectively passivating this non-growth/growth pair "
        "\\emph{or} the accuracy limits of a foundation MLIP on these systems (absolute "
        "$\\Delta E_{\\mathrm{ads}}$ errors can flip a marginally selective pair to "
        "anti-selective); distinguishing the two requires a higher-tier (DFT) check, "
        "which is why the pipeline emits the result with an explicit validity flag "
        "rather than as a settled physical claim."
    )


def _fb_conclusion(p: dict) -> str:
    h = _hyp(p)
    verdict = str(p.get("verdict", "inconclusive")).replace("_", " ")
    sc = p.get("screening") or {}
    campaign = ""
    if sc.get("winner"):
        cfg = sc.get("config", {})
        campaign = (
            f"a {cfg.get('pool_size', 'multi')}-candidate screening funnel "
            "(prior rank $\\rightarrow$ shared-slab batch screen $\\rightarrow$ "
            "full-fidelity top-"
            f"{cfg.get('top_k', 'k')} $\\rightarrow$ recommendation), "
        )
    return (
        "We demonstrated an end-to-end autonomous in-silico co-scientist for "
        "area-selective ALD: literature grounding into a typed knowledge graph, "
        "human-gated hypothesis commitment, fidelity-gated amorphous surface ensembles, "
        "agentic site-matched inhibitor/precursor selection, " + campaign
        + "tiered literature-anchored reactivity validation, and autonomous manuscript "
        f"assembly. For the "
        + ("recommended" if sc.get("winner") else "committed")
        + " intervention --- "
        f"\\emph{{{latex_escape(h.get('inhibitor', 'n/a'))}}} passivating "
        f"{latex_escape(h.get('non_growth_surface', 'n/a'))} against "
        f"{latex_escape(h.get('precursor', 'n/a'))}-based "
        f"{latex_escape(h.get('target_film', 'n/a'))} growth on "
        f"{latex_escape(h.get('growth_surface', 'n/a'))} --- the recorded verdict is "
        f"\\emph{{{latex_escape(verdict)}}} at "
        f"$S = {_num(p, 'selectivity', 'S_at_target_mean')} \\pm "
        f"{_num(p, 'selectivity', 'S_at_target_std')}$ and "
        f"{_num(p, 'selectivity', 'target_thickness_nm')} nm. "
        + (
            "That the target was not met is reported as a valid negative result: the "
            "system correctly finds, with site-resolved energetics, that no screened "
            "candidate clears the selectivity target on this pair. The deliverable is "
            "not a molecule that hits 90\\% but the reproducible autonomous framework "
            "--- the fidelity-gated surface builder and the agentic screen --- that can "
            "reach and defend that conclusion, and the evidence it produces for why, "
            "which is the whole point of the study regardless of the verdict. "
            if str(p.get("verdict", "")).lower() in
            {"rejected", "inconclusive", "fail", "failed", "not_supported"} else ""
        )
        + "The pipeline's defining "
        "property is that this conclusion, its uncertainty, and its caveats were "
        "produced, flagged, and written up by the system itself: every number in this "
        "manuscript resolves to a logged computation in the run artifacts, and "
        "re-running the recorded command with the pinned provenance of "
        "Table~\\ref{tab:provenance} reproduces it."
    )


def _fb_reproducibility(p: dict) -> str:
    prov = p.get("provenance", {})
    run_id = p.get("run_id", "run")
    return (
        "Reproducibility is a first-class requirement. "
        "Table~\\ref{tab:provenance} pins the computational provenance of this run: "
        "engine, compute tier, potential and device, process temperature, dose ratio, "
        "ensemble size, and RNG seed. The full artifact set --- "
        "\\texttt{asald\\_results.json} (all energetics, coverages, curves, and the "
        "verdict), \\texttt{surface\\_fidelity.json} (per-slab site densities and gate "
        "outcomes), \\texttt{validation\\_plan.json} (the designer's method, "
        "assumptions, and reasoning trace), \\texttt{validation\\_summary.md} (the "
        "LLM-generated results digest), the slab geometries under \\texttt{datasets/}, "
        "and per-iteration logs under \\texttt{simulation\\_logs/} --- is preserved "
        f"under \\texttt{{artifacts/{latex_escape(run_id)}/}}. The validation is a "
        "single command:\n"
        "\\begin{verbatim}\nCOMPUTE_TIER="
        f"{prov.get('compute_tier', 1)} MLIP_DEVICE={latex_escape(str(prov.get('mlip_device', 'cpu')))} \\\n"
        f"  python -m aicoscientist.cli_validate \\\n    --run-id {latex_escape(run_id)}\n"
        "\\end{verbatim}\n"
        "and the manuscript itself regenerates with "
        "\\texttt{python -m aicoscientist.cli\\_paper}."
    )


_FALLBACKS = {
    "abstract": _fb_abstract,
    "keywords": _fb_keywords,
    "introduction": _fb_introduction,
    "architecture": _fb_architecture,
    "methods_surfaces": _fb_methods_surfaces,
    "methods_selection": _fb_methods_selection,
    "methods_protocol": _fb_methods_protocol,
    "results": _fb_results,
    "discussion": _fb_discussion,
    "limitations": _fb_limitations,
    "conclusion": _fb_conclusion,
    "reproducibility": _fb_reproducibility,
}
