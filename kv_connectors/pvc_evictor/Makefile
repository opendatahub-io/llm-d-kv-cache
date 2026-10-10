IMAGE_REPOSITORY ?= quay.io/pvc-evictor/pvc-evictor
IMAGE_TAG ?= latest
# The Go image uses its own repository so it never overwrites the Python
# image that the Helm chart deploys.
GO_IMAGE_REPOSITORY ?= quay.io/pvc-evictor/pvc-evictor-go

.PHONY: test test_go docker-build docker-build_go

test:
	python3 -m pytest tests/ -v

test_go:
	cd go && go test ./...

docker-build:
	cd .. && podman build -f pvc_evictor/Dockerfile -t $(IMAGE_REPOSITORY):$(IMAGE_TAG) .

docker-build_go:
	cd .. && podman build -f pvc_evictor/Dockerfile.go -t $(GO_IMAGE_REPOSITORY):$(IMAGE_TAG) .
