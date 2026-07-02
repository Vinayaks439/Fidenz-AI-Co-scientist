"""Environment-driven settings for the AI Co-Scientist."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.3, alias="LLM_TEMPERATURE")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Google Gemini via AI Studio API keys (provider: google_genai)
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    # Google Gemini via Vertex AI (provider: google_vertexai; uses ADC / service account)
    google_cloud_project: str | None = Field(default=None, alias="GOOGLE_CLOUD_PROJECT")
    google_cloud_location: str = Field(
        default="us-central1", alias="GOOGLE_CLOUD_LOCATION"
    )
    google_application_credentials: str | None = Field(
        default=None, alias="GOOGLE_APPLICATION_CREDENTIALS"
    )

    # Sources
    contact_email: str = Field(default="anonymous@example.com", alias="CONTACT_EMAIL")
    semantic_scholar_api_key: str | None = Field(
        default=None, alias="SEMANTIC_SCHOLAR_API_KEY"
    )
    max_results_per_source: int = Field(default=8, alias="MAX_RESULTS_PER_SOURCE")

    # Run tuning
    max_domains: int = Field(default=4, alias="MAX_DOMAINS")
    num_hypotheses: int = Field(default=8, alias="NUM_HYPOTHESES")
    artifacts_dir: str = Field(default="artifacts", alias="ARTIFACTS_DIR")

    # Layer 3 — bounded reflection / closed-loop refinement budget
    max_validation_iters: int = Field(default=2, alias="MAX_VALIDATION_ITERS")

    # Layer 3 — screening-funnel campaign (batch candidate screening).
    # funnel: assemble a pool of N candidates, Tier-0 prior-rank all, MLIP-screen the
    # shortlist on SHARED gated slabs (apples-to-apples), full-fidelity re-run of the
    # top-k, then a recommendation agent picks the winner. single: legacy behavior
    # (one candidate per reflect iteration).
    screening_mode: str = Field(
        default="funnel", alias="SCREENING_MODE", description="funnel | single"
    )
    screen_pool_size: int = Field(
        default=20, alias="SCREEN_POOL_SIZE",
        description="candidate pool size for the screening funnel (10-50)",
    )
    screen_shortlist_m: int = Field(
        default=8, alias="SCREEN_SHORTLIST_M",
        description="candidates advancing from the Tier-0 prior rank to the MLIP batch",
    )
    screen_top_k: int = Field(
        default=3, alias="SCREEN_TOP_K",
        description="candidates re-run at full fidelity before the final recommendation",
    )
    screen_ensemble_n: int = Field(
        default=2, alias="SCREEN_ENSEMBLE_N",
        description="slabs per surface during the batch screen (full ensemble for top-k)",
    )

    # Layer 3 — AS-ALD surface-reactivity engine (ADR-004/009)
    compute_tier: int = Field(
        default=0,
        alias="COMPUTE_TIER",
        description="0 = pure-python (literature/xTB dE), 1 = foundation MLIP, 2 = +spot-checks",
    )
    mlip_model: str = Field(default="mace-mp", alias="MLIP_MODEL")
    mlip_device: str = Field(
        default="auto", alias="MLIP_DEVICE", description="auto|cuda|mps|cpu"
    )

    # Tier-1 slab construction (Phase 1)
    slab_source: str = Field(
        default="procedural",
        alias="SLAB_SOURCE",
        description=(
            "procedural = crystalline-derived hydroxylated slab (default; passes the "
            "fidelity gate); amorphous = geometric melt-quench disorder (opt-in, flagged "
            "by the descriptor gate as needing MLIP relaxation); toy = legacy grid"
        ),
    )
    slab_miller: str = Field(default="1,0,0", alias="SLAB_MILLER")
    slab_supercell_str: str = Field(default="2,2", alias="SLAB_SUPERCELL")
    hydroxylation_target: float | None = Field(
        default=None,
        alias="HYDROXYLATION_TARGET",
        description="override target site density (sites/nm^2); default = band-based",
    )
    # Real MLIP-driven melt-quench amorphization (SLAB_SOURCE=md-amorphous, Tier>=1).
    # Heat the slab's mobile region with the MLIP (MACE), hold to melt/disorder, quench on
    # a temperature ramp, then relax to 0 K -> a genuinely amorphized network (vs the
    # geometric 'amorphous' knob). Defaults follow the melt-quench-MD literature, adapted
    # to a foundation MLIP on a GPU (see notes):
    #   * NVT (fixed cell): uMLIPs give unphysically expanded densities under NPT because
    #     their pressure/E-V response is poor -> NVT-quench is the recommended fix
    #     (arXiv:2606.16385). For a vacuum slab NPT is inapplicable anyway.
    #   * Melt T ~3500 K: a-SiO2 melt-quench uses ~4000 K (JPCC 2016); a-Si3N4 ~2500-5000 K
    #     (arXiv:2408.05782) -> 3500 K is a single-value compromise (lower it toward 3000 K
    #     for SiN-heavy stability, raise toward 4000 K for SiO2 fidelity).
    #   * dt 0.5 fs: safer than 1 fs at melt T (Si3N4 MLIP studies use 0.25-1 fs).
    #   * Quench fast is OK: a-Si3N4 RDF is insensitive to rate over 1e13-1e15 K/s
    #     (arXiv:2408.05782); 8000 steps x 0.5 fs over ~3200 K ~ 8e14 K/s sits in that band.
    # A high-T MD blow-up falls back to the geometric amorphizer, then the toy slab.
    mq_melt_temperature_k: float = Field(default=3500.0, alias="MQ_MELT_TEMPERATURE_K")
    mq_final_temperature_k: float = Field(default=300.0, alias="MQ_FINAL_TEMPERATURE_K")
    mq_melt_steps: int = Field(default=3000, alias="MQ_MELT_STEPS")
    mq_quench_steps: int = Field(default=8000, alias="MQ_QUENCH_STEPS")
    mq_timestep_fs: float = Field(default=0.5, alias="MQ_TIMESTEP_FS")
    mq_friction: float = Field(default=0.02, alias="MQ_FRICTION")
    mq_fix_bottom_frac: float = Field(
        default=0.4, alias="MQ_FIX_BOTTOM_FRAC",
        description="freeze this bottom fraction as bulk anchor during melt-quench",
    )
    # LLM param-tuning agent for melt-quench: iterate melt T / quench steps on a small,
    # cheap probe slab, scoring each by the fidelity gate + Si coordination quality, until
    # the agent stops or the trial budget is spent. Tuned params are then reused for the
    # full ensemble. Falls back to a deterministic heuristic if no LLM is available.
    mq_autotune: bool = Field(default=False, alias="MQ_AUTOTUNE")
    mq_autotune_trials: int = Field(default=4, alias="MQ_AUTOTUNE_TRIALS")
    mq_autotune_probe_supercell: str = Field(
        default="2,2", alias="MQ_AUTOTUNE_PROBE_SUPERCELL",
        description="small supercell used only for the cheap tuning probes",
    )
    mq_autotune_probe_quench: int = Field(
        default=1500, alias="MQ_AUTOTUNE_PROBE_QUENCH",
        description="cap quench steps during probes to keep tuning fast",
    )
    # Multi-site adsorption search (Phase 1)
    n_adsorption_sites: int = Field(default=4, alias="N_ADSORPTION_SITES")
    adsorption_rotations: int = Field(default=4, alias="ADSORPTION_ROTATIONS")
    adsorption_heights_str: str = Field(default="1.8,2.4", alias="ADSORPTION_HEIGHTS")
    # Tier-2 GFN2-xTB spot-check (Phase 2)
    xtb_spotcheck_sites: int = Field(default=1, alias="XTB_SPOTCHECK_SITES")
    # RSA steric-coverage cap (Phase 2); Tier-1+ only
    use_rsa_coverage: bool = Field(default=True, alias="USE_RSA_COVERAGE")
    # Novel-compound proposer (Phase 2): let the agent invent new inhibitor candidates
    use_inhibitor_proposer: bool = Field(
        default=False, alias="USE_INHIBITOR_PROPOSER"
    )
    n_proposed_inhibitors: int = Field(default=3, alias="N_PROPOSED_INHIBITORS")
    # AI experiment planner (Layer 3): let the LLM decide, per closed-loop iteration,
    # WHICH inhibitor to test and at WHAT compute tier (cheap Tier-0 screen vs. expensive
    # real-MLIP Tier-1), grounded in the deep research + prior reflection. When disabled,
    # the designer falls back to the deterministic rank-index + global COMPUTE_TIER.
    use_ai_planner: bool = Field(default=False, alias="USE_AI_PLANNER")
    ai_planner_max_tier: int = Field(
        default=1,
        alias="AI_PLANNER_MAX_TIER",
        description="ceiling tier the AI planner may request per iteration (0/1/2)",
    )
    # Kim et al. 2026 site-resolved reactivity (deltaEr + Ea per site type)
    use_site_resolved_reactivity: bool = Field(
        default=True, alias="USE_SITE_RESOLVED_REACTIVITY"
    )
    compute_activation_energy: bool = Field(
        default=False, alias="COMPUTE_ACTIVATION_ENERGY",
        description="run NEB for Ea (expensive); default uses literature priors",
    )
    dose_time_s: float = Field(
        default=60.0, alias="DOSE_TIME_S",
        description="inhibitor dose time for Arrhenius kinetic gate (seconds)",
    )
    ald_temperature_k: float = Field(default=423.0, alias="ALD_TEMPERATURE_K")
    surface_ensemble_n: int = Field(
        default=5, alias="SURFACE_ENSEMBLE_N", description="slabs per surface condition"
    )
    selection_criteria_path: str = Field(
        default="selection_criteria.md", alias="SELECTION_CRITERIA_PATH"
    )
    priors_source: str = Field(
        default="auto",
        alias="PRIORS_SOURCE",
        description=(
            "auto = KG-mined literature priors win; manual = selection_criteria.md wins. "
            "Both are always merged over built-in defaults."
        ),
    )

    @property
    def artifacts_path(self) -> Path:
        path = Path(self.artifacts_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def resolved_mlip_device(self) -> str:
        """Resolve 'auto' to the best available torch device.

        MACE energy differences require float64, which the MPS backend does not
        support, so Apple-silicon runs fall back to CPU for the MLIP tier.
        """
        if self.mlip_device != "auto":
            return self.mlip_device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    @property
    def slab_supercell(self) -> tuple[int, int]:
        try:
            a, b = (int(x) for x in self.slab_supercell_str.split(",")[:2])
            return (a, b)
        except Exception:  # noqa: BLE001
            return (2, 2)

    def slab_miller_for(self, material_key: str) -> tuple[int, int, int]:
        try:
            parts = [int(x) for x in self.slab_miller.split(",")[:3]]
            return tuple(parts)  # type: ignore[return-value]
        except Exception:  # noqa: BLE001
            return (1, 0, 0)

    @property
    def adsorption_heights(self) -> list[float]:
        try:
            return [float(x) for x in self.adsorption_heights_str.split(",")]
        except Exception:  # noqa: BLE001
            return [1.8, 2.4]

    @property
    def user_agent(self) -> str:
        return f"AI-Co-Scientist/0.1 (mailto:{self.contact_email})"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
