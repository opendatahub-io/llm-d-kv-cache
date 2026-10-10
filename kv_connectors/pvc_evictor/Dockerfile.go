ARG BASE_IMAGE=gcr.io/distroless/static:nonroot

FROM --platform=${BUILDPLATFORM} golang:1.26.9 AS go-builder

ARG TARGETOS
ARG TARGETARCH

WORKDIR /workspace

COPY pvc_evictor/go/go.mod pvc_evictor/go/go.sum ./
RUN go mod download

COPY pvc_evictor/go/cmd/ cmd/
COPY pvc_evictor/go/internal/ internal/

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
