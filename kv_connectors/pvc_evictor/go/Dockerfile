ARG BASE_IMAGE=gcr.io/distroless/static:nonroot

FROM --platform=${BUILDPLATFORM} golang:1.26.9 AS go-builder

ARG TARGETOS
ARG TARGETARCH

WORKDIR /workspace

COPY go.mod go.sum ./
RUN go mod download

COPY cmd/ cmd/
COPY internal/ internal/

RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build \
    -trimpath \
    -ldflags="-s -w" \
    -o bin/pvc-evictor \
    ./cmd/pvc-evictor

FROM ${BASE_IMAGE}

WORKDIR /

COPY --from=go-builder --chown=65532:65532 /workspace/bin/pvc-evictor /app/pvc-evictor

USER 65532:65532

ENTRYPOINT ["/app/pvc-evictor"]
