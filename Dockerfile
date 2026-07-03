# AS-ALD Co-Scientist — Hugging Face Space (Gradio UI over the CLI pipeline).
# Ships both compute tiers: Tier-0 (CPU, literature/xTB priors) and Tier-1
# (foundation-MLIP / MACE reactivity). The default PyPI torch wheel is CUDA-enabled,
# so Tier-1 uses a GPU automatically when the Space runs on GPU hardware, and falls
# back to CPU otherwise. Upgrade in Settings → Hardware to get the GPU.
FROM python:3.12-slim

# Runtime libs: libgomp1 (torch/OpenMP), libGL/libglib (matplotlib/pymatgen),
# libxrender/libxext (rdkit).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates \
        libgomp1 libglib2.0-0 libgl1 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# tectonic for Layer-4 PDF compilation (optional; .tex is emitted regardless). The
# drop installer ignores --dest and unpacks into the *current* directory, so cd into a
# PATH dir first, then verify it's runnable. Pre-warm the package cache by compiling a
# doc with the manuscript's exact preamble (IEEEtran + friends), so the first real
# compile doesn't have to download the TeX bundle within the 300 s compile timeout.
RUN cd /usr/local/bin \
    && curl -fsSL https://drop-sh.fullyjustified.net | sh \
    && tectonic --version \
    && printf '%s\n' \
        '\documentclass[journal]{IEEEtran}' \
        '\usepackage{graphicx}\usepackage{booktabs}\usepackage{amsmath}' \
        '\usepackage{amssymb}\usepackage[hidelinks]{hyperref}\usepackage{url}' \
        '\begin{document}\title{warm}\author{a}\maketitle warmup $x^2$' \
        '\begin{equation}E=mc^2\end{equation}\end{document}' > /tmp/warm.tex \
    && (cd /tmp && tectonic warm.tex && rm -f warm.tex warm.pdf) \
    || echo "tectonic install/warmup skipped (manuscript will emit .tex only)"

WORKDIR /app

# Install deps first from just the build metadata + src, so editing app.py (or other
# runtime files) doesn't bust the heavy MLIP install layer on rebuilds.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[mlip,structures]" "gradio>=4.0"

# Now bring in the rest (app.py, selection_criteria.md, etc.).
COPY . /app

# The standalone tectonic above is non-functional here (Debian has no tectonic package
# and the binary is missing shared libs like libgraphite2). Drop it and compile the
# manuscript with texlive + latexmk instead — compiler.py tries tectonic -> latexmk ->
# pdflatex, so removing the broken tectonic sends it straight to latexmk. The template
# uses a manual thebibliography block (no biber/bibtex), and IEEEtran lives in
# texlive-publishers. This toolchain needs no runtime network download.
RUN rm -f /usr/local/bin/tectonic \
    && apt-get update && apt-get install -y --no-install-recommends \
        latexmk \
        texlive-latex-base texlive-latex-recommended texlive-latex-extra \
        texlive-publishers texlive-fonts-recommended \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs the container as uid 1000 with no matching /etc/passwd entry, which
# breaks libraries that call getpwuid() to locate $HOME (MACE/torch model caches) --
# that is what made Tier-1 MLIP silently fall back to Tier-0. Create the user, point all
# caches at a writable HOME, and disable torch.compile/dynamo (its mega-cache
# double-registers on the reflection loop's 2nd MLIP call). Together these let Tier-1
# actually run the MACE reactivity engine instead of falling back.
RUN useradd -m -u 1000 user 2>/dev/null || true
ENV HOME=/home/user \
    XDG_CACHE_HOME=/home/user/.cache \
    TORCH_HOME=/home/user/.cache/torch \
    HF_HOME=/home/user/.cache/huggingface \
    MPLCONFIGDIR=/home/user/.cache/matplotlib \
    TORCHDYNAMO_DISABLE=1 \
    TORCH_COMPILE_DISABLE=1 \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor
RUN mkdir -p /home/user/.cache /tmp/torchinductor \
    && chmod -R 777 /home/user /tmp/torchinductor

ENV PYTHONUNBUFFERED=1
# ARTIFACTS_DIR is intentionally NOT set here: app.py auto-detects HF persistent
# storage and uses /data/artifacts when the Space has storage enabled (Settings ->
# Storage), falling back to ./artifacts (ephemeral) otherwise. Setting it here would
# pin runs to the ephemeral container filesystem. /app/artifacts stays as the fallback.
RUN mkdir -p /app/artifacts && chmod 777 /app/artifacts

EXPOSE 7860
CMD ["python", "app.py"]
