IMAGE_REPOSITORY ?= quay.io/pvc-evictor/pvc-evictor
IMAGE_TAG ?= latest
# Use a separate tag when building the Go image so it can share the ODH
# repository with the Python image.
GO_IMAGE_REPOSITORY ?= quay.io/opendatahub/llm-d-kv-cache

PYTHON_EXE := $(shell command -v python3.12 || command -v python3)
VENV_DIR := .venv
VENV_BIN := $(VENV_DIR)/bin

.PHONY: venv test test_go docker-build docker-build_go

venv: ## Create the dev venv and install the project with dev dependencies
	@if [ ! -f "$(VENV_BIN)/python" ]; then \
		echo "Creating virtual environment in $(VENV_DIR)..."; \
		$(PYTHON_EXE) -m venv $(VENV_DIR); \
		$(VENV_BIN)/pip install --upgrade pip > /dev/null; \
	fi
	@$(VENV_BIN)/pip install -q -e ".[dev]"

test: venv
	$(VENV_BIN)/python -m pytest tests/ -v

test_go:
	cd go && go test ./...

docker-build:
	cd .. && podman build -f pvc_evictor/Dockerfile -t $(IMAGE_REPOSITORY):$(IMAGE_TAG) .

docker-build_go:
	cd go && podman build -t $(GO_IMAGE_REPOSITORY):$(IMAGE_TAG) .
