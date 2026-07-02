"""Gradio front-end for the AS-ALD Co-Scientist pipeline, for Hugging Face Spaces.

Wraps the three CLI stages (aicoscientist -> aicoscientist-validate ->
aicoscientist-paper) as one guided run: enter an idea, watch the logs stream,
download the resulting artifacts (knowledge graph, validation results,
manuscript).
"""

from __future__ import annotations

import html as htmllib
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

import gradio as gr

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))

DEFAULT_IDEA = "passivate a-SiN, grow SiOx-on-a-SiO2 to 90% selectivity at 10 nm"

STAGES = [
    ("Layer 1-2: literature research + hypothesis selection", "aicoscientist"),
    ("Layer 3: in-silico selectivity validation", "aicoscientist-validate"),
    ("Layer 4: manuscript stitching", "aicoscientist-paper"),
]

# Tier-1 MLIP sampling presets. Bigger slabs (SLAB_SUPERCELL) put more atoms through each
# MACE force eval, and more sites/rotations/heights/ensemble members keep the GPU busy
# longer -- the two levers that actually load a big GPU (MACE runs one structure at a time
# via ASE, so a small slab never saturates an A100). Runtime scales with all of these.
SAMPLING_PRESETS = {
    "Standard": {
        "SLAB_SUPERCELL": "2,2", "SURFACE_ENSEMBLE_N": "5",
        "N_ADSORPTION_SITES": "4", "ADSORPTION_ROTATIONS": "4",
        "ADSORPTION_HEIGHTS": "1.8,2.4",
    },
    "High (GPU)": {
        "SLAB_SUPERCELL": "3,3", "SURFACE_ENSEMBLE_N": "6",
        "N_ADSORPTION_SITES": "8", "ADSORPTION_ROTATIONS": "6",
        "ADSORPTION_HEIGHTS": "1.8,2.2,2.6",
    },
    "Max (GPU)": {
        "SLAB_SUPERCELL": "4,4", "SURFACE_ENSEMBLE_N": "8",
        "N_ADSORPTION_SITES": "12", "ADSORPTION_ROTATIONS": "8",
        "ADSORPTION_HEIGHTS": "1.6,2.0,2.4,2.8,3.2",
    },
}


# One-click runtime profiles, calibrated for the melt-quench surface model on an A100.
# Each bundles the cost levers (sampling, ensemble, melt/quench steps, autotune, reflection
# iterations). "Custom" defers to the individual fields. Wall-times are estimates (per-eval
# time is only known after the first real run); with crystalline slabs runs finish faster.
RUNTIME_PROFILES = {
    "Custom (use fields below)": None,
    "Fast (< 30 min)": {
        "sampling": "Standard", "ensemble": 2, "melt_steps": 800,
        "quench_steps": 1500, "autotune": False, "max_iters": 1,
    },
    "Balanced (< 1 hr)": {
        "sampling": "Standard", "ensemble": 3, "melt_steps": 1500,
        "quench_steps": 3000, "autotune": False, "max_iters": 2,
    },
    "High-fidelity (> 1 hr)": {
        "sampling": "High (GPU)", "ensemble": 5, "melt_steps": 3000,
        "quench_steps": 8000, "autotune": True, "max_iters": 2,
    },
}

# Tier-1 surface model (SLAB_SOURCE). Crystalline-derived is the fast default and passes
# the fidelity gate; md-amorphous runs a real MACE melt-quench MD (heat->melt->quench->
# relax) for a genuinely amorphized network -- most faithful, slowest (adds ~melt+quench
# MD steps per slab), with graceful fallback if the high-T MD goes unstable.
SLAB_SOURCES = {
    "Crystalline-derived (fast, default)": "procedural",
    "Geometric amorphous (disorder knob)": "amorphous",
    "MLIP melt-quench MD (real, slow)": "md-amorphous",
}


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _gpu_status() -> str:
    """One-line summary of whether a CUDA GPU is visible to torch."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return f"✅ CUDA GPU detected: {name} — Tier-1 will run on GPU."
        return (
            "⚠️ No CUDA GPU on this hardware. Tier-1 still runs, but on CPU "
            "(much slower). Upgrade the Space in Settings → Hardware to a GPU."
        )
    except Exception:  # noqa: BLE001
        return (
            "⚠️ torch not importable — Tier-1 (MLIP) unavailable. "
            "Use Tier-0 (CPU) here."
        )


def _stream_command(cmd: list[str], env: dict):
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    # Split on BOTH newline and carriage return so tqdm progress bars (which redraw with
    # "\r") stream live into the log instead of arriving as one blob when the bar finishes.
    seg_re = re.compile(r"[^\r\n]*[\r\n]")

    def _emit(seg: str):
        # Tee to container stdout so `curl .../logs/run` follows it from a terminal too.
        sys.stdout.write(seg)
        sys.stdout.flush()
        return seg

    buffer = ""
    while True:
        chunk = proc.stdout.read(256)
        if chunk == "":
            break
        buffer += chunk
        pos = 0
        for m in seg_re.finditer(buffer):
            yield _emit(m.group(0))
            pos = m.end()
        buffer = buffer[pos:]
    if buffer:
        yield _emit(buffer)
    proc.wait()
    if proc.returncode != 0:
        yield _emit(f"\n[stage exited with code {proc.returncode}]\n")


def _viz_files(run_dir: Path) -> list[str]:
    """Structure files to show in the 3D viewer: molecules, the adsorption complex,
    and one representative slab per material (the ensemble writes several)."""
    ds = run_dir / "datasets"
    if not ds.exists():
        return []
    mols = sorted(ds.glob("mol_*.xyz"))
    complexes = sorted(ds.glob("complex_*.extxyz"))
    picked, seen = [], set()
    for s in sorted(ds.glob("slab_*.extxyz")):  # slab_<material>_<seed>.extxyz
        parts = s.stem.split("_")
        mat = parts[1] if len(parts) > 1 else s.stem
        if mat not in seen:
            seen.add(mat)
            picked.append(s)
    return [str(p) for p in (mols + complexes + picked)]


def _to_plain_xyz(path: str) -> str | None:
    """Normalize any (ext)xyz file to a clean 4-column XYZ string.

    3Dmol's extended-xyz path can silently parse zero atoms on some headers (e.g. an
    adsorption complex written with pbc="T T F" and a placed molecule outside the cell),
    which shows as a blank panel. Stripping to ``element x y z`` per atom bypasses that
    path so every structure renders the same reliable way molecules do."""
    try:
        lines = Path(path).read_text().splitlines()
    except Exception:  # noqa: BLE001
        return None
    if len(lines) < 3:
        return None
    try:
        n = int(lines[0].split()[0])
    except (ValueError, IndexError):
        return None
    atoms = []
    for ln in lines[2:2 + n]:
        tok = ln.split()
        if len(tok) >= 4:
            atoms.append(f"{tok[0]} {tok[1]} {tok[2]} {tok[3]}")
    if not atoms:
        return None
    return f"{len(atoms)}\n\n" + "\n".join(atoms)


def _structure_viewer_html(files: list[str]) -> str:
    """Render structures as interactive 3D views using 3Dmol.js (client-side, in an
    iframe so its scripts run inside Gradio). Every file is normalized to plain XYZ."""
    if not files:
        return (
            "<p style='color:#888;font-family:sans-serif'>3D structures (molecules, "
            "passivated slabs, adsorption complex) will appear here as the in-silico "
            "layer builds them…</p>"
        )
    divs, scripts = [], []
    for i, f in enumerate(files):
        data = _to_plain_xyz(f)
        if data is None:
            continue
        label = htmllib.escape(Path(f).name)
        divs.append(
            f'<div style="display:inline-block;margin:6px;text-align:center;'
            f'font-family:sans-serif;font-size:12px;color:#333">'
            f'<div id="v{i}" style="width:340px;height:260px;position:relative;'
            f'border:1px solid #ddd;border-radius:6px"></div>{label}</div>'
        )
        scripts.append(
            f"var D{i}={json.dumps(data)};"
            f'var m{i}=$3Dmol.createViewer("v{i}",{{backgroundColor:"white"}});'
            f'm{i}.addModel(D{i},"xyz");'
            f"m{i}.setStyle({{}},{{stick:{{radius:0.12}},sphere:{{scale:0.30}}}});"
            f"m{i}.zoomTo();m{i}.render();"
        )
    inner = (
        '<script src="https://3Dmol.org/build/3Dmol-min.js"></script>'
        '<div style="text-align:center">' + "".join(divs) + "</div>"
        "<script>" + "".join(scripts) + "</script>"
    )
    height = 300 * ((len(files) + 1) // 2) + 40
    return (
        f'<iframe srcdoc="{htmllib.escape(inner, quote=True)}" width="100%" '
        f'height="{height}" style="border:none"></iframe>'
    )


def run_pipeline(idea: str, offline: bool, gemini_key: str, auto_decision: str,
                 tier1_gpu: bool, sampling: str = "High (GPU)",
                 slab_source_label: str = "Crystalline-derived (fast, default)",
                 mq_melt_t: float = 3500.0, mq_melt_steps: float = 3000,
                 mq_quench_steps: float = 8000, mq_timestep: float = 0.5,
                 mq_ensemble: float = 0, mq_autotune: bool = False,
                 runtime_profile: str = "Custom (use fields below)",
                 screen_pool: float = 20, screen_top_k: float = 3):
    empty_viewer = _structure_viewer_html([])
    idea = (idea or "").strip()
    if not idea:
        yield "Please enter a research idea.", None, empty_viewer
        return

    env = os.environ.copy()
    # Screening funnel: pool of N candidates -> Tier-0 rank -> MLIP batch on shared
    # slabs -> top-k full fidelity -> recommendation agent -> paper.
    env["SCREENING_MODE"] = "funnel"
    env["SCREEN_POOL_SIZE"] = str(int(screen_pool))
    env["SCREEN_TOP_K"] = str(int(screen_top_k))
    if not offline:
        gemini_key = (gemini_key or "").strip()
        if not gemini_key:
            yield "Provide a Gemini API key, or switch on offline/mock mode.", None, empty_viewer
            return
        env["LLM_PROVIDER"] = "google_genai"
        # Stable Gemini 3.1 Flash-family model (there is no plain gemini-3.1-flash;
        # the other 3.1 Flash variants are Live/TTS preview-only).
        env["LLM_MODEL"] = "gemini-3.1-flash-lite"
        env["GOOGLE_API_KEY"] = gemini_key

    gpu_note = ""
    if tier1_gpu:
        # A runtime profile (if not "Custom") overrides the individual cost levers so one
        # choice fixes the whole compute budget.
        prof = RUNTIME_PROFILES.get(runtime_profile)
        if prof:
            sampling = prof["sampling"]
            mq_melt_steps = prof["melt_steps"]
            mq_quench_steps = prof["quench_steps"]
            mq_ensemble = prof["ensemble"]
            mq_autotune = prof["autotune"]
            env["MAX_VALIDATION_ITERS"] = str(prof["max_iters"])
            gpu_note = f"profile '{runtime_profile}' | "

        # Tier-1: foundation-MLIP (MACE) adsorption search on real rdkit/pymatgen
        # structures. Force CUDA when a GPU is present so the in-silico layer runs on the
        # GPU (float64 MACE is fine on CUDA); fall back to auto->cpu only if there is none.
        env["COMPUTE_TIER"] = "1"
        env["SLAB_SOURCE"] = SLAB_SOURCES.get(slab_source_label, "procedural")
        if _has_cuda():
            env["MLIP_DEVICE"] = "cuda"
            gpu_note += "MLIP device: cuda (GPU)"
        else:
            env["MLIP_DEVICE"] = "auto"
            gpu_note += "MLIP device: auto (no CUDA GPU found -> CPU, slow)"
        gpu_note += f" | surface: {env['SLAB_SOURCE']}"
        preset = SAMPLING_PRESETS.get(sampling, SAMPLING_PRESETS["High (GPU)"])
        env.update(preset)  # user's own env still wins if set on the Space
        gpu_note += f" | sampling '{sampling}': " + ", ".join(
            f"{k}={v}" for k, v in preset.items()
        )
        # Melt-quench knobs applied AFTER the preset so an explicit ensemble override wins.
        if env["SLAB_SOURCE"] == "md-amorphous":
            env["MQ_MELT_TEMPERATURE_K"] = str(float(mq_melt_t))
            env["MQ_MELT_STEPS"] = str(int(mq_melt_steps))
            env["MQ_QUENCH_STEPS"] = str(int(mq_quench_steps))
            env["MQ_TIMESTEP_FS"] = str(float(mq_timestep))
            if int(mq_ensemble) > 0:  # smaller ensemble to bound melt-quench cost
                env["SURFACE_ENSEMBLE_N"] = str(int(mq_ensemble))
            env["MQ_AUTOTUNE"] = "true" if mq_autotune else "false"
            gpu_note += (
                f" | melt-quench: T={float(mq_melt_t):.0f}K melt={int(mq_melt_steps)} "
                f"quench={int(mq_quench_steps)} dt={float(mq_timestep)}fs"
                + (f" ensemble={int(mq_ensemble)}" if int(mq_ensemble) > 0 else "")
                + (" | LLM autotune ON" if mq_autotune else "")
            )
    else:
        env["COMPUTE_TIER"] = "0"

    run_id = _new_run_id()
    run_dir = ARTIFACTS_DIR / run_id
    offline_flag = ["--offline"] if offline else []
    tier_label = "1 (MLIP)" if tier1_gpu else "0 (CPU priors)"
    log = (
        f"run id: {run_id}\n"
        f"mode: {'offline (mock)' if offline else 'live (Gemini)'}\n"
        f"compute tier: {tier_label}\n"
        f"screening funnel: pool={int(screen_pool)} -> top-{int(screen_top_k)} "
        f"-> recommendation\n"
    )
    if gpu_note:
        log += gpu_note + "\n"
    if tier1_gpu:
        log += _gpu_status() + "\n"
    yield log, None, empty_viewer

    commands = [
        ["aicoscientist", "--idea", idea, "--run-id", run_id, "--verbose",
         "--auto", auto_decision or "select:1", *offline_flag],
        ["aicoscientist-validate", "--run-id", run_id, "--verbose", *offline_flag],
        ["aicoscientist-paper", "--run-id", run_id, "--verbose", *offline_flag],
    ]

    # Refresh the 3D viewer whenever new structure files appear (near-live).
    last_sig, viewer = None, empty_viewer
    print(f"\n[run {run_id}] {log.strip()}", flush=True)
    for (title, _), cmd in zip(STAGES, commands):
        header = f"\n=== {title} ===\n"
        log += header
        print(header, end="", flush=True)
        yield log, None, viewer
        for chunk in _stream_command(cmd, env):
            if chunk.endswith("\r"):
                # tqdm progress redraw: overwrite the current line (like a terminal) so the
                # bar updates in place instead of piling up hundreds of lines.
                body = chunk[:-1]
                head = log.rsplit("\n", 1)[0] if "\n" in log else ""
                log = (head + "\n" + body) if "\n" in log else body
            else:
                log += chunk
            sig = tuple(_viz_files(run_dir))
            if sig != last_sig:
                last_sig = sig
                viewer = _structure_viewer_html(list(sig))
            yield log, None, viewer

    files = sorted(str(p) for p in run_dir.rglob("*") if p.is_file()) if run_dir.exists() else []
    log += f"\n\nDone. {len(files)} artifact file(s) written to {run_dir}\n"
    yield log, (files or None), _structure_viewer_html(_viz_files(run_dir))


with gr.Blocks(title="AS-ALD Co-Scientist") as demo:
    gr.Markdown(
        "# AS-ALD Co-Scientist\n"
        "Autonomous in-silico co-scientist for area-selective atomic layer deposition "
        "(AS-ALD). Enter a research idea and run the full pipeline: literature research "
        "→ hypothesis selection → in-silico selectivity validation → manuscript.\n\n"
        "**Offline/mock mode** is deterministic and needs no API key. Uncheck it and add a "
        "Gemini API key (from [Google AI Studio](https://aistudio.google.com/apikey)) to run "
        "with a real LLM (`gemini-3.1-flash-lite`) for Layer 1 research, hypothesis scoring, "
        "and manuscript prose."
    )
    idea = gr.Textbox(label="Research idea", value=DEFAULT_IDEA, lines=2)
    with gr.Row():
        offline = gr.Checkbox(label="Offline / mock mode (no API key needed)", value=True)
        auto_decision = gr.Textbox(
            label="Auto hypothesis decision", value="select:1",
            info="e.g. select:1, merge:1,2, new",
        )
    gemini_key = gr.Textbox(label="Gemini API key (Google AI Studio)", type="password")

    with gr.Row():
        screen_pool = gr.Slider(
            minimum=10, maximum=50, value=20, step=1,
            label="Screening pool size (N inhibitors)",
            info=(
                "How many candidate inhibitors enter the screening funnel: library + "
                "KG-mined + AI-proposed novel molecules. All are prior-ranked; the "
                "shortlist is screened with the reactivity engine on identical slabs."
            ),
        )
        screen_top_k = gr.Slider(
            minimum=1, maximum=10, value=3, step=1,
            label="Top-k full-fidelity re-runs",
            info=(
                "The best k candidates from the batch screen are re-run at full "
                "ensemble fidelity before the recommendation agent picks the winner."
            ),
        )

    tier1_gpu = gr.Checkbox(
        label="Tier-1: foundation-MLIP (MACE) reactivity — uses GPU when available",
        value=False,
        info=(
            "Off = Tier-0 (fast, CPU, literature/xTB priors). On = real 3D molecules + "
            "crystalline-derived slabs + MACE adsorption search. Needs a GPU Space to be "
            "fast; on CPU hardware it runs but is slow."
        ),
    )
    runtime_profile = gr.Dropdown(
        choices=list(RUNTIME_PROFILES.keys()),
        value="Custom (use fields below)",
        label="Runtime profile (Tier-1, A100)",
        info=(
            "One-click compute budget for the melt-quench surface model. Overrides "
            "sampling, ensemble, melt/quench steps, autotune, and reflection iterations. "
            "'Fast' <30 min · 'Balanced' <1 hr · 'High-fidelity' >1 hr (AI autotune on). "
            "'Custom' uses the fields below. Crystalline slabs finish faster than these."
        ),
    )
    with gr.Row():
        sampling = gr.Dropdown(
            choices=list(SAMPLING_PRESETS.keys()),
            value="High (GPU)",
            label="In-silico GPU sampling (Tier-1 only)",
            info=(
                "Bigger slabs + more sites/rotations/heights/ensemble = more GPU work and "
                "better statistics, but longer runs. 'Max' loads an A100 hardest; "
                "'Standard' is the fast default. Ignored at Tier-0."
            ),
        )
        slab_source = gr.Dropdown(
            choices=list(SLAB_SOURCES.keys()),
            value="Crystalline-derived (fast, default)",
            label="Surface model (Tier-1 only)",
            info=(
                "How the SiO₂/SiN slabs are built. 'MLIP melt-quench MD' runs a real MACE "
                "heat→melt→quench→relax to make a genuinely amorphous network (slowest, "
                "adds MD per slab; falls back gracefully if the high-T MD is unstable). "
                "Ignored at Tier-0."
            ),
        )

    with gr.Accordion(
        "Melt-quench MD settings (only used when Surface model = MLIP melt-quench MD)",
        open=False,
    ):
        gr.Markdown(
            "Literature-grounded defaults (NVT quench; a-SiO₂ melt ~4000 K, a-Si₃N₄ "
            "~2500–5000 K → 3500 K compromise; a-Si₃N₄ RDF is insensitive to quench rate "
            "over 10¹³–10¹⁵ K/s). Each slab ≈ melt+quench MACE evals, so this is a "
            "multi-hour job — lower the amorphization ensemble and/or quench steps to trim."
        )
        with gr.Row():
            mq_melt_t = gr.Number(value=3500, label="Melt temperature (K)")
            mq_melt_steps = gr.Number(value=3000, precision=0, label="Melt/equilibrate steps")
            mq_quench_steps = gr.Number(value=8000, precision=0, label="Quench steps")
        with gr.Row():
            mq_timestep = gr.Number(value=0.5, label="Timestep (fs)")
            mq_ensemble = gr.Number(
                value=0, precision=0,
                label="Amorphization ensemble N (0 = use sampling preset)",
            )
        mq_autotune = gr.Checkbox(
            value=False,
            label="AI auto-tune melt T / quench (LLM agent)",
            info=(
                "Iterate melt T & quench steps on a small probe slab, scored by the "
                "fidelity gate + Si coordination, until the LLM agent converges — then "
                "reuse the tuned params for the ensemble. Adds a few probe MD runs up "
                "front; needs the Gemini key (falls back to a heuristic offline)."
            ),
        )
    gpu_status = gr.Markdown(_gpu_status())

    run_btn = gr.Button("Run pipeline", variant="primary")
    log_box = gr.Textbox(label="Log", lines=25, autoscroll=True)

    gr.Markdown(
        "### Live 3D structures\n"
        "Interactive (drag to rotate, scroll to zoom). Populated during the in-silico "
        "layer: inhibitor/precursor molecules, the passivated SiO₂/SiN slabs, and the "
        "placed adsorption complex. Tier-1 writes the real slabs/complex; molecules show "
        "at any tier."
    )
    viewer_html = gr.HTML(value=_structure_viewer_html([]))
    files_out = gr.File(label="Artifacts", file_count="multiple")

    run_btn.click(
        run_pipeline,
        inputs=[idea, offline, gemini_key, auto_decision, tier1_gpu, sampling, slab_source,
                mq_melt_t, mq_melt_steps, mq_quench_steps, mq_timestep, mq_ensemble,
                mq_autotune, runtime_profile, screen_pool, screen_top_k],
        outputs=[log_box, files_out, viewer_html],
    )
    demo.load(_gpu_status, inputs=None, outputs=gpu_status)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
