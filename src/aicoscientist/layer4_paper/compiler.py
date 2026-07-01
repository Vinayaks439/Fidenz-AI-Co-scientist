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
    """Compile ``tex_path`` to a PDF; return the PDF path or None if no toolchain."""
    workdir = tex_path.parent
    pdf_path = tex_path.with_suffix(".pdf")

    toolchains = [
        ["tectonic", tex_path.name],
        ["latexmk", "-pdf", "-interaction=nonstopmode", tex_path.name],
        ["pdflatex", "-interaction=nonstopmode", tex_path.name],
    ]
    for cmd in toolchains:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            subprocess.run(
                cmd, cwd=workdir, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
            )
            if pdf_path.exists():
                logger.info("compiled manuscript with %s -> %s", cmd[0], pdf_path)
                return pdf_path
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s failed: %s", cmd[0], exc)
    logger.warning("no LaTeX toolchain succeeded; leaving .tex source at %s", tex_path)
    return None
