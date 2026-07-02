"""Gradio front-end for the AS-ALD Co-Scientist pipeline, for Hugging Face Spaces.

Wraps the three CLI stages (aicoscientist -> aicoscientist-validate ->
aicoscientist-paper) as one guided run: enter an idea, watch the logs stream,
download the resulting artifacts (knowledge graph, validation results,
manuscript).

Connection-loss resilience: the pipeline is executed in a DETACHED background
thread that appends to ``artifacts/<run_id>/ui_run.log``; the Gradio callback only
*tails* that file. A browser refresh, dropped WebSocket/SSE, or proxy hiccup
(the "427"/connection-errored class of failures) cancels the tail, never the run.
Use the "Reattach to run" box to resume streaming any run id after a refresh.
The tail also emits a heartbeat every poll so the HF proxy never sees an idle
stream, and a run may execute/stream for up to 24 h (MAX_RUN_SECONDS).
"""

from __future__ import annotations

import html as htmllib
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import gradio as gr


def _resolve_artifacts_dir() -> tuple[Path, bool]:
    """Prefer HF persistent storage when present.

    Hugging Face mounts persistent storage (Settings -> Storage) at /data; anything
    elsewhere is wiped on every Space restart/sleep. Priority:
      1. explicit ARTIFACTS_DIR env (operator override),
      2. /data/artifacts when /data exists and is writable (persistent),
      3. ./artifacts (ephemeral fallback).
    Returns (path, is_persistent).
    """
    env_dir = os.environ.get("ARTIFACTS_DIR")
    if env_dir:
        p = Path(env_dir)
        return p, str(p).startswith("/data")
    data = Path("/data")
    if data.is_dir() and os.access(data, os.W_OK):
        return data / "artifacts", True
    return Path("artifacts"), False


ARTIFACTS_DIR, ARTIFACTS_PERSISTENT = _resolve_artifacts_dir()
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
# Make the CLI stages (which read ARTIFACTS_DIR themselves) agree with the UI.
os.environ["ARTIFACTS_DIR"] = str(ARTIFACTS_DIR)

# When persistent storage exists, also keep the model caches there: MACE weights,
# torch hub, and HF downloads then survive restarts instead of re-downloading on
# every container boot. Must happen before any calculator is created (the pipeline
# runs in subprocesses that inherit this environment).
if ARTIFACTS_PERSISTENT and str(ARTIFACTS_DIR).startswith("/data"):
    _cache_root = Path("/data/.cache")
    try:
        _cache_root.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(_cache_root)          # MACE (~/.cache/mace)
        os.environ["TORCH_HOME"] = str(_cache_root / "torch")     # torch hub
        os.environ["HF_HOME"] = str(_cache_root / "huggingface")  # HF downloads
    except Exception:  # noqa: BLE001 -- storage quirks must never block startup
        pass

DEFAULT_IDEA = "passivate a-SiN, grow SiOx-on-a-SiO2 to 90% selectivity at 10 nm"

# Hard ceiling on one pipeline run (execution AND ui streaming): 24 hours.
MAX_RUN_SECONDS = 24 * 60 * 60
# Tail poll / heartbeat interval (an SSE frame is sent at least this often, which
# keeps the HF reverse proxy from idling out the stream during quiet MD stretches).
POLL_SECONDS = 2.0

STAGES = [
    ("Layer 1-2: literature research + hypothesis selection", "aicoscientist"),
    ("Layer 3: in-silico selectivity validation", "aicoscientist-validate"),
    ("Layer 4: manuscript stitching", "aicoscientist-paper"),
]

# Tier-1 MLIP sampling presets. Bigger slabs (SLAB_SUPERCELL) put more atoms through each
# MACE force eval, and more sites/rotations/heights keep the GPU busy longer -- the levers
# that actually load a big GPU (MACE runs one structure at a time via ASE, so a small slab
# never saturates an A100). Runtime scales with all of these. The surface-ensemble size is
# a separate, explicit UI field (default 1) so a long run is opt-in, not a preset surprise.
SAMPLING_PRESETS = {
    "Standard": {
        "SLAB_SUPERCELL": "2,2",
        "N_ADSORPTION_SITES": "4", "ADSORPTION_ROTATIONS": "4",
        "ADSORPTION_HEIGHTS": "1.8,2.4",
    },
    "High (GPU)": {
        "SLAB_SUPERCELL": "3,3",
        "N_ADSORPTION_SITES": "8", "ADSORPTION_ROTATIONS": "6",
        "ADSORPTION_HEIGHTS": "1.8,2.2,2.6",
    },
    "Max (GPU)": {
        "SLAB_SUPERCELL": "4,4",
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
        "sampling": "Standard", "ensemble": 1, "melt_steps": 800,
        "quench_steps": 1500, "autotune": False, "max_iters": 1,
    },
    "Balanced (< 1 hr)": {
        "sampling": "Standard", "ensemble": 2, "melt_steps": 1500,
        "quench_steps": 3000, "autotune": False, "max_iters": 1,
    },
    "High-fidelity (> 1 hr)": {
        "sampling": "High (GPU)", "ensemble": 3, "melt_steps": 3000,
        "quench_steps": 8000, "autotune": True, "max_iters": 1,
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

# In-process registry of pipeline threads (survives browser disconnects; does NOT
# survive a container restart -- the on-disk log/status files cover reattach for
# anything the container still has).
RUNS: dict[str, dict] = {}
_LATEST_RUN_ID: str | None = None


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


# ──────────────────────── detached run execution ────────────────────────


def _log_path(run_id: str) -> Path:
    return ARTIFACTS_DIR / run_id / "ui_run.log"


def _status_path(run_id: str) -> Path:
    return ARTIFACTS_DIR / run_id / "ui_run.status"


def _append_log(run_id: str, text: str) -> None:
    """Append to the run's log file and tee to container stdout.

    newline='' disables newline translation so tqdm's "\\r" redraws survive the
    write/read round-trip and _render_log can collapse them for display."""
    with open(_log_path(run_id), "a", encoding="utf-8", errors="replace",
              newline="") as fh:
        fh.write(text)
    sys.stdout.write(text)
    sys.stdout.flush()


def _read_log(run_id: str) -> str:
    try:
        with open(_log_path(run_id), "r", encoding="utf-8", errors="replace",
                  newline="") as fh:
            return fh.read()
    except Exception:  # noqa: BLE001
        return ""


def _set_status(run_id: str, status: str) -> None:
    _status_path(run_id).write_text(status, encoding="utf-8")


def _get_status(run_id: str) -> str:
    try:
        return _status_path(run_id).read_text(encoding="utf-8").strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _execute_pipeline(run_id: str, commands: list[list[str]], env: dict) -> None:
    """Run all stages sequentially in a background thread, appending to the log file.

    This function owns the run: it keeps executing even if every browser tab is
    closed. Only a container restart (Space sleep/reboot) or the 24 h ceiling
    stops it.
    """
    start = time.time()
    status = "done"
    try:
        for (title, _), cmd in zip(STAGES, commands):
            _append_log(run_id, f"\n=== {title} ===\n")
            # Binary read + manual decode: text=True would translate the lone "\r" of
            # tqdm progress redraws into "\n" (universal newlines), defeating the
            # carriage-return collapsing that _render_log does for the UI.
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read(256)
                if chunk == b"":
                    break
                _append_log(run_id, chunk.decode("utf-8", errors="replace"))
                if time.time() - start > MAX_RUN_SECONDS:
                    proc.kill()
                    _append_log(run_id, "\n[run exceeded the 24 h ceiling; stage killed]\n")
                    status = "timeout (24h)"
                    break
            proc.wait()
            if status != "done":
                break
            if proc.returncode != 0:
                _append_log(run_id, f"\n[stage exited with code {proc.returncode}]\n")
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        _append_log(run_id, f"\n[pipeline thread error: {exc}]\n")
    elapsed = time.time() - start
    _append_log(run_id, f"\n[pipeline {status} after {elapsed / 60:.1f} min]\n")
    _set_status(run_id, status)


def _render_log(raw: str) -> str:
    """Emulate terminal carriage-return behavior for tqdm-style progress redraws:
    within each line, only the content after the last "\\r" is shown."""
    out_lines = []
    for line in raw.split("\n"):
        out_lines.append(line.rsplit("\r", 1)[-1] if "\r" in line else line)
    return "\n".join(out_lines)


def _latest_running_run() -> str:
    """Most recent run id that is still marked running (for reattach after refresh)."""
    if _LATEST_RUN_ID and _get_status(_LATEST_RUN_ID) in ("running", "unknown"):
        return _LATEST_RUN_ID
    try:
        candidates = sorted(
            (p.parent for p in ARTIFACTS_DIR.glob("*/ui_run.status")
             if p.read_text(encoding="utf-8").strip() == "running"),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        return candidates[0].name if candidates else ""
    except Exception:  # noqa: BLE001
        return ""


# ──────────────────────── 3D structure viewer ────────────────────────


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


# ──────────────────────── UI callbacks ────────────────────────


def _tail_run(run_id: str):
    """Stream a run's log/status/structures to the UI. Cancelling this (refresh,
    dropped connection) does NOT affect the run -- reattach any time."""
    empty_viewer = _structure_viewer_html([])
    run_dir = ARTIFACTS_DIR / run_id
    last_sig: tuple | None = None
    viewer = empty_viewer
    start = time.time()

    while True:
        text = _render_log(_read_log(run_id))

        sig = tuple(_viz_files(run_dir))
        if sig != last_sig:
            last_sig = sig
            viewer = _structure_viewer_html(list(sig))

        status = _get_status(run_id)
        if status not in ("running", "unknown"):
            files = (sorted(str(p) for p in run_dir.rglob("*") if p.is_file())
                     if run_dir.exists() else [])
            text += (f"\n\nRun {run_id} finished with status: {status}. "
                     f"{len(files)} artifact file(s) in {run_dir}\n")
            yield text, (files or None), viewer
            return

        # Heartbeat: yield every poll even when nothing changed, so the SSE stream
        # carries traffic and the HF proxy never idle-kills the connection.
        yield text, None, viewer
        if time.time() - start > MAX_RUN_SECONDS:
            yield (text + "\n[ui stream detached after 24 h; the run may still be "
                   "executing — reattach with the run id]\n"), None, viewer
            return
        time.sleep(POLL_SECONDS)


def attach_run(run_id: str):
    """Reattach the UI to an existing (possibly still running) run after a refresh."""
    empty_viewer = _structure_viewer_html([])
    run_id = (run_id or "").strip()
    if not run_id:
        run_id = _latest_running_run()
    if not run_id:
        yield ("No run id given and no running run found on this container.",
               None, empty_viewer)
        return
    if not _log_path(run_id).exists():
        yield (f"No log found for run id '{run_id}' on this container "
               "(the Space may have restarted — artifacts are ephemeral without "
               "persistent storage).", None, empty_viewer)
        return
    yield from _tail_run(run_id)


def run_pipeline(idea: str, offline: bool, gemini_key: str, auto_decision: str,
                 tier1_gpu: bool, sampling: str = "High (GPU)",
                 slab_source_label: str = "Crystalline-derived (fast, default)",
                 mq_melt_t: float = 3500.0, mq_melt_steps: float = 3000,
                 mq_quench_steps: float = 8000, mq_timestep: float = 0.5,
                 mq_ensemble: float = 0, mq_autotune: bool = False,
                 runtime_profile: str = "Custom (use fields below)",
                 screen_pool: float = 40, screen_shortlist: float = 10,
                 screen_top_k: float = 3, screen_ensemble: float = 2,
                 max_iters: float = 1, ensemble_n: float = 1):
    global _LATEST_RUN_ID
    empty_viewer = _structure_viewer_html([])
    idea = (idea or "").strip()
    if not idea:
        yield "Please enter a research idea.", None, empty_viewer
        return

    env = os.environ.copy()
    # Cost levers the user just asked for, explicit and UI-controlled (both default 1):
    # one reflection iteration, one slab per surface condition. Profiles may override.
    env["MAX_VALIDATION_ITERS"] = str(max(1, int(max_iters)))
    env["SURFACE_ENSEMBLE_N"] = str(max(1, int(ensemble_n)))
    # Screening funnel: pool of N candidates -> Tier-0 rank -> MLIP batch on shared
    # slabs -> top-k full fidelity -> recommendation agent -> paper.
    env["SCREENING_MODE"] = "funnel"
    env["SCREEN_POOL_SIZE"] = str(int(screen_pool))
    env["SCREEN_SHORTLIST_M"] = str(int(screen_shortlist))
    env["SCREEN_TOP_K"] = str(int(screen_top_k))
    env["SCREEN_ENSEMBLE_N"] = str(max(1, int(screen_ensemble)))
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
            mq_autotune = prof["autotune"]
            env["SURFACE_ENSEMBLE_N"] = str(prof["ensemble"])
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
        env.update(preset)
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
    run_dir.mkdir(parents=True, exist_ok=True)
    offline_flag = ["--offline"] if offline else []
    tier_label = "1 (MLIP)" if tier1_gpu else "0 (CPU priors)"
    header = (
        f"run id: {run_id}\n"
        f"mode: {'offline (mock)' if offline else 'live (Gemini)'}\n"
        f"artifacts: {run_dir} "
        f"({'persistent /data storage' if ARTIFACTS_PERSISTENT else 'EPHEMERAL — enable persistent storage in Settings → Storage to keep results across restarts'})\n"
        f"compute tier: {tier_label}\n"
        f"reflection iterations: {env['MAX_VALIDATION_ITERS']} | "
        f"surface ensemble N: {env['SURFACE_ENSEMBLE_N']}\n"
        f"screening funnel: pool={int(screen_pool)} -> MLIP top-{int(screen_shortlist)} "
        f"-> full top-{int(screen_top_k)} -> recommendation | "
        f"fidelity slabs/surface: screen={env['SCREEN_ENSEMBLE_N']}, "
        f"full re-run n={env['SURFACE_ENSEMBLE_N']}\n"
    )
    if gpu_note:
        header += gpu_note + "\n"
    if tier1_gpu:
        header += _gpu_status() + "\n"
    header += (
        "\nThis run executes in the background on the Space: a browser refresh or "
        "lost connection does NOT stop it. To resume watching, paste the run id "
        f"'{run_id}' into 'Reattach to run' below.\n"
    )

    commands = [
        ["aicoscientist", "--idea", idea, "--run-id", run_id, "--verbose",
         "--auto", auto_decision or "select:1", *offline_flag],
        ["aicoscientist-validate", "--run-id", run_id, "--verbose", *offline_flag],
        ["aicoscientist-paper", "--run-id", run_id, "--verbose", *offline_flag],
    ]

    _set_status(run_id, "running")
    _append_log(run_id, header)
    thread = threading.Thread(
        target=_execute_pipeline, args=(run_id, commands, env),
        name=f"pipeline-{run_id}", daemon=True,
    )
    RUNS[run_id] = {"thread": thread, "started": time.time()}
    _LATEST_RUN_ID = run_id
    thread.start()

    yield from _tail_run(run_id)


with gr.Blocks(title="AS-ALD Co-Scientist") as demo:
    gr.Markdown(
        "# AS-ALD Co-Scientist\n"
        "Autonomous in-silico co-scientist for area-selective atomic layer deposition "
        "(AS-ALD). Enter a research idea and run the full pipeline: literature research "
        "→ hypothesis selection → in-silico selectivity validation → manuscript.\n\n"
        "**Offline/mock mode** is deterministic and needs no API key. Uncheck it and add a "
        "Gemini API key (from [Google AI Studio](https://aistudio.google.com/apikey)) to run "
        "with a real LLM (`gemini-3.1-flash-lite`) for Layer 1 research, hypothesis scoring, "
        "and manuscript prose.\n\n"
        "🛡️ **Long runs are refresh-proof**: the pipeline executes in the background on "
        "the Space and keeps going if your browser refreshes or the connection drops — "
        "reattach with the run id below. ⚠️ For multi-hour GPU runs, set the Space's "
        "**Settings → Sleep time** to *Never* (or long enough), otherwise Hugging Face "
        "puts the whole container to sleep after the idle window and the run dies with it. "
        "Enable **persistent storage** to keep artifacts across restarts."
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
        max_iters = gr.Slider(
            minimum=1, maximum=5, value=1, step=1,
            label="Reflection iterations",
            info=(
                "Closed-loop refine budget (MAX_VALIDATION_ITERS). 1 = single pass, "
                "no refinement re-runs — the fast default."
            ),
        )
        ensemble_n = gr.Slider(
            minimum=1, maximum=8, value=1, step=1,
            label="Full-fidelity re-run ensemble (the 'n=' in top-k re-run)",
            info=(
                "Slabs per surface for the top-k full-fidelity re-run — the "
                "'top-K full-fidelity re-run (n=…)' line in the log. Each finalist "
                "repeats the whole adsorption search on every slab, so runtime scales "
                "linearly. 1 = fastest; raise for error-barred statistics. Values up to "
                "the screening ensemble reuse its already-gated slabs (no new fidelity "
                "tests)."
            ),
        )

    with gr.Row():
        screen_pool = gr.Slider(
            minimum=8, maximum=50, value=40, step=1,
            label="Pool size (N inhibitors)",
            info=(
                "How many candidate inhibitors enter the funnel: library + KG-mined + "
                "AI-proposed novel molecules. All N are Tier-0 prior-ranked."
            ),
        )
        screen_shortlist = gr.Slider(
            minimum=1, maximum=50, value=10, step=1,
            label="MLIP shortlist (top-M)",
            info=(
                "The top-M prior-ranked candidates are MLIP-screened on identical "
                "shared slabs (apples-to-apples). Raise toward the pool size to "
                "MLIP-screen more (more GPU cost)."
            ),
        )
        screen_top_k = gr.Slider(
            minimum=1, maximum=10, value=3, step=1,
            label="Full-fidelity top-k",
            info=(
                "The best k from the MLIP screen are re-run at full ensemble fidelity "
                "before the recommendation agent picks the winner."
            ),
        )
        screen_ensemble = gr.Slider(
            minimum=1, maximum=8, value=2, step=1,
            label="Fidelity re-runs per screen (slabs/surface)",
            info=(
                "How many fidelity-gated slabs are built per surface for the candidate "
                "screen (SCREEN_ENSEMBLE_N). Every shortlisted inhibitor is adsorbed on "
                "these same shared slabs; the top-k re-run uses max(this, Surface "
                "ensemble N). 2 = the fidelity test runs twice per surface."
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
            "'Custom' uses the fields above/below. Crystalline slabs finish faster."
        ),
    )
    with gr.Row():
        sampling = gr.Dropdown(
            choices=list(SAMPLING_PRESETS.keys()),
            value="High (GPU)",
            label="In-silico GPU sampling (Tier-1 only)",
            info=(
                "Bigger slabs + more sites/rotations/heights = more GPU work and "
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
                label="Amorphization ensemble N (0 = use 'Surface ensemble N' above)",
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

    with gr.Row():
        attach_id = gr.Textbox(
            label="Reattach to run",
            placeholder="run id (blank = latest running run on this container)",
            info=(
                "Refreshed the page or lost connection? The run kept going. Paste its "
                "run id here (or leave blank) and click Reattach to resume streaming."
            ),
        )
        attach_btn = gr.Button("Reattach")

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
                mq_autotune, runtime_profile, screen_pool, screen_shortlist, screen_top_k,
                screen_ensemble, max_iters, ensemble_n],
        outputs=[log_box, files_out, viewer_html],
    )
    attach_btn.click(
        attach_run,
        inputs=[attach_id],
        outputs=[log_box, files_out, viewer_html],
    )
    demo.load(_gpu_status, inputs=None, outputs=gpu_status)
    demo.load(_latest_running_run, inputs=None, outputs=attach_id)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
