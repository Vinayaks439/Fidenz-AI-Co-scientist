"""LaTeX compiler agent (ADR-007).

Tries a sequence of TeX toolchains (tectonic, latexmk, pdflatex). If none is installed
the manuscript is left as a ``.tex`` source -- the pipeline never hard-fails on a missing
TeX distribution (the same graceful-degradation policy used elsewhere in the repo).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def compile_pdf(tex_path: Path) -> Path | None:
    """Compile ``tex_path`` to a PDF; return the PDF path or None if no toolchain.

    ``latexmk`` runs as many pdflatex passes as it takes to resolve cross-references
    on its own. The bare-``pdflatex`` fallback does not, so we run it three times --
    otherwise the \\ref cross-references (hypotheses table, figures, tables) render as
    an unresolved ``??`` on a single pass.
    """
    workdir = tex_path.parent
    pdf_path = tex_path.with_suffix(".pdf")

    # (command, n_passes). latexmk self-iterates; pdflatex needs manual repeats.
    toolchains = [
        (["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
          tex_path.name], 1),
        (["pdflatex", "-interaction=nonstopmode", tex_path.name], 3),
        (["tectonic", tex_path.name], 1),
    ]
    available = [t[0][0] for t in toolchains if shutil.which(t[0][0])]
    logger.info("LaTeX toolchains on PATH: %s", available or "NONE")
    for cmd, passes in toolchains:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            for i in range(passes):
                subprocess.run(
                    cmd, cwd=workdir, check=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600,
                )
            if pdf_path.exists():
                logger.info("compiled manuscript with %s (%d pass%s) -> %s",
                            cmd[0], passes, "es" if passes != 1 else "", pdf_path)
                return pdf_path
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s failed: %s", cmd[0], exc)
    logger.warning(
        "no LaTeX toolchain succeeded; leaving .tex source at %s. On HF Spaces this "
        "means the container was built before TeX Live was added -- rebuild the Space.",
        tex_path)
    return None
