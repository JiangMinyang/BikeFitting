# ─────────────────────────────────────────────────────────────────────────────
# Bike Fit Analyzer — Makefile
#
# Usage:
#   make build        Build the Docker image
#   make run          Start the web server (http://localhost:8080)
#   make stop         Stop the running container
#   make restart      Rebuild and restart
#   make logs         Tail container logs
#   make clean        Remove container and image
#   make test         Run unit tests (local, no Docker)
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_NAME  := bike-fit-app
CONTAINER   := bike-fit-app
PORT        := 8080
MODELS_DIR  := $(HOME)/.bike-fit-models
RESULTS_DIR := $(HOME)/Documents/BikeFitResults

.PHONY: build run stop restart logs clean test shell download-model

# Download the RTMPose-l ONNX model locally (~150 MB, one-time).
# Must be run before `make build` to bake the model into the image.
# Safe to skip — the container will auto-download on first inference instead.
download-model:
	@echo ""
	@echo "  Downloading RTMPose-l ONNX model (~150 MB)..."
	@mkdir -p $(MODELS_DIR)
	@mkdir -p $(MODELS_DIR)
	python3 -c "import sys; sys.path.insert(0,'$(CURDIR)'); from core.rtmpose_backend import download_model; download_model(model_dir='$(MODELS_DIR)')"
	@echo "  Model saved to $(MODELS_DIR)"
	@echo ""

# Build the Docker image.
# Run `make download-model` first to bake RTMPose-l into the image layer.
build:
	docker build -t $(IMAGE_NAME) .

# Run the web server; visit http://localhost:$(PORT)
# Models dir is mounted so the auto-downloaded model persists between rebuilds.
run:
	@echo ""
	@echo "  Starting Bike Fit Analyzer..."
	@echo "  Open http://localhost:$(PORT) in your browser."
	@echo ""
	docker run --rm -d \
		--name $(CONTAINER) \
		-p $(PORT):8080 \
		-v $(RESULTS_DIR):/app/output \
		-v "$(MODELS_DIR):/app/models" \
		$(IMAGE_NAME)
	@echo "  Container '$(CONTAINER)' is running."

# Dev mode: mount live source files so code changes apply without rebuilding.
# Only requires `make build` once; after that just `make dev` + re-run analysis.
dev: stop
	@echo ""
	@echo "  Starting Bike Fit Analyzer (dev — live source mount)..."
	@echo "  Open http://localhost:$(PORT) in your browser."
	@echo ""
	docker run --rm -d \
		--name $(CONTAINER) \
		-p $(PORT):8080 \
		-v $(RESULTS_DIR):/app/output \
		-v "$(MODELS_DIR):/app/models" \
		-v "$(CURDIR)/core:/app/core" \
		-v "$(CURDIR)/reports:/app/reports" \
		-v "$(CURDIR)/web:/app/web" \
		-v "$(CURDIR)/server.py:/app/server.py" \
		-v "$(CURDIR)/db.py:/app/db.py" \
		-v "$(CURDIR)/main.py:/app/main.py" \
		$(IMAGE_NAME)
	@echo "  Container '$(CONTAINER)' is running (live source mounted)."

# Stop the container
stop:
	docker stop $(CONTAINER) 2>/dev/null || true

# Rebuild and restart
restart: stop build run

# Follow container logs
logs:
	docker logs -f $(CONTAINER)

# Open a shell inside the running container
shell:
	docker exec -it $(CONTAINER) /bin/bash

# Remove container + image
clean: stop
	docker rm $(CONTAINER) 2>/dev/null || true
	docker rmi $(IMAGE_NAME) 2>/dev/null || true

# Run unit tests locally (requires Python + deps installed)
test:
	python -m pytest tests/ -v
