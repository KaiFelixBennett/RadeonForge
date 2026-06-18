# RadeonForge — common tasks. `make` or `make help` lists everything.
# Cross-platform note: `demo` only needs Python; `setup`/`smoke`/`train` need WSL2/Linux + ROCm.
PY   ?= python3
PORT ?= 8765
.DEFAULT_GOAL := help

help: ## show this help
	@echo "RadeonForge — make targets:"
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

demo: ## live dashboard demo, no GPU needed -> http://127.0.0.1:8765/gl
	$(PY) scripts/demo_dashboard.py --port $(PORT) --open

setup: ## bootstrap the WSL2/Linux ROCm training env (idempotent)
	bash scripts/setup.sh

doctor: ## environment health check (driver/ROCm/torch/bnb/HF)
	bash scripts/doctor.sh

smoke: ## 50-step QLoRA smoke test — proves the loss actually falls
	$(PY) scripts/smoke_test.py

train: ## run the worked Gemma-4 QLoRA example
	$(PY) examples/gemma4-12b-qlora/train_qlora.py --config examples/gemma4-12b-qlora/config.yaml

charts: ## regenerate result charts from the A/B JSON
	$(PY) examples/gemma4-12b-qlora/make_charts.py

assets: ## regenerate dashboard screenshots + GIF + branding (needs Node + ffmpeg)
	bash tools/capture/run.sh

clean: ## remove generated demo data
	rm -rf examples/dashboard-demo

.PHONY: help demo setup doctor smoke train charts assets clean
