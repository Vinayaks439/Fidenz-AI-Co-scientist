# AS-ALD Co-Scientist -- common workflows.
# Override any variable on the command line, e.g.:
#   make run IDEA="passivate a-SiN, grow SiOx to 95% at 12 nm" RUN_ID=exp1
#   make validate RUN_ID=exp1 TIER=1

PYTHON ?= python
RUN_ID ?= demo
IDEA    ?= passivate a-SiN, grow SiOx-on-a-SiO2 to 90% selectivity at 10 nm
AUTO    ?= select:1
TIER    ?= 0

# Use module invocation so targets work whether or not console scripts are installed.
RESEARCH = $(PYTHON) -m aicoscientist.cli
VALIDATE = $(PYTHON) -m aicoscientist.cli_validate
PAPER    = $(PYTHON) -m aicoscientist.cli_paper

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Editable install with the OpenAI extra (Tier-0 + Layer 4 figures)
	$(PYTHON) -m pip install -e ".[openai]"

.PHONY: install-mlip
install-mlip: ## Add the Tier-1 foundation-MLIP stack (ASE + MACE)
	$(PYTHON) -m pip install -e ".[mlip]"

.PHONY: research
research: ## Layer 1-2: literature research + human hypothesis commitment (interactive)
	$(RESEARCH) --idea "$(IDEA)" --run-id $(RUN_ID)

.PHONY: research-offline
research-offline: ## Layer 1-2 offline + auto-select (no network/LLM key)
	$(RESEARCH) --idea "$(IDEA)" --offline --run-id $(RUN_ID) --auto $(AUTO)

.PHONY: validate
validate: ## Layer 3: in-silico surface-reactivity validation (COMPUTE_TIER=$(TIER))
	COMPUTE_TIER=$(TIER) $(VALIDATE) --run-id $(RUN_ID)

.PHONY: validate-offline
validate-offline: ## Layer 3 offline (deterministic designer/reflection)
	COMPUTE_TIER=$(TIER) $(VALIDATE) --run-id $(RUN_ID) --offline

.PHONY: paper
paper: ## Layer 4: stitch the reproducible manuscript
	$(PAPER) --run-id $(RUN_ID)

.PHONY: run
run: research-offline validate-offline paper ## Full offline funnel (L1->L2->L3->L4)
	@echo "Done. See artifacts/$(RUN_ID)/ and artifacts/$(RUN_ID)/manuscript/"

.PHONY: docker
docker: ## Build + run the reproducible Tier-0 container
	docker build -t asald-coscientist .
	docker run --rm asald-coscientist

.PHONY: untrack-egg
untrack-egg: ## Stop tracking the auto-generated *.egg-info (keeps it on disk)
	git rm -r --cached --ignore-unmatch "src/*.egg-info" || true

.PHONY: clean
clean: ## Remove run artifacts, checkpoints, caches, and build metadata
	rm -rf artifacts/ *.sqlite *.sqlite-* build/ dist/
	find . -path ./.venv -prune -o -name '__pycache__' -type d -exec rm -rf {} +
	find src -name '*.egg-info' -type d -exec rm -rf {} + 2>/dev/null || true
