# Go PVC Evictor manifests

These manifests deploy the Go PVC Evictor as a single Kubernetes `Deployment`.
The deployment uses an existing PVC. It does not create one.

All commands below run from this `deploy/` directory.

## Before you apply

1. Set the namespace. Replace `namespace: my-namespace` in
   `kustomization.yaml` with the namespace where the vLLM workload and the
   shared PVC are deployed.
2. Set the PVC. Replace `claimName: my-vllm-cache-pvc` in `deployment.yaml`
   with the name of the existing PVC that holds the vLLM KV cache.
3. Review the settings in `configmap.env`, such as `CLEANUP_THRESHOLD`,
   `TARGET_THRESHOLD` and `CACHE_DIRECTORY`. Kustomize adds a content hash to
   the generated ConfigMap name, so applying new values restarts the Pod.
4. Optional: use another image tag (see below).

Then apply the manifests:

```bash
kubectl apply -k .
```

## Image

The default image is:

```text
quay.io/opendatahub/odh-llm-d-kv-cache-pvc-evictor:odh-stable-go
```

The tag is set in one place, the `images` entry in `kustomization.yaml`. To use
another tag, change `newTag` there. PR images use the tag
`odh-pr-go-<full-commit-sha>`, where `<full-commit-sha>` is the full 40-character
commit SHA, not the short form. You can also set the tag with the `kustomize`
binary, which rewrites `kustomization.yaml` in place:

```bash
kustomize edit set image \
  quay.io/opendatahub/odh-llm-d-kv-cache-pvc-evictor=quay.io/opendatahub/odh-llm-d-kv-cache-pvc-evictor:odh-pr-go-<full-commit-sha>
```

The Deployment pulls the image on every Pod start (`imagePullPolicy: Always`),
because the `odh-stable-go` tag is replaced on every build.

## Storage

The Go process needs a writable POSIX-compatible filesystem mounted at
`/kv-cache`. It does not need a PVC object. To use NFS or a CSI volume, replace
the `persistentVolumeClaim` entry under `volumes` in `deployment.yaml` with that
volume source.

## Permissions

The Pod must be able to read and delete the files that the vLLM workload
writes. The Deployment does not set `runAsUser`, `runAsGroup` or `fsGroup`:

- The Go image runs as UID 65532.
- OpenShift assigns IDs from the namespace range.

If the vLLM workload writes the cache files as a different user, the evictor
cannot delete them. It logs a warning for each file and the volume stays full.
In that case, set matching IDs with a patch.

`patches/security-and-affinity-example.yaml` shows how to set the IDs, the
OpenShift SELinux level, and a Pod affinity rule. The affinity rule keeps the
Pod on the same node as the vLLM Pod, which a `ReadWriteOnce` volume needs.
To use it, replace the example values, then add it to `kustomization.yaml`:

```yaml
patches:
- path: patches/security-and-affinity-example.yaml
```

## Network and API access

By default the Go process opens no network port, so the manifests have no
`Service` and no HTTP probes. If you set `STORAGE_EVENTS_ENDPOINT` in
`configmap.env`, the process listens on that address and publishes
`BlockRemoved` events over ZeroMQ. In that case, add a `containerPort` to the
Deployment and a `Service` for subscribers.

The Go process does not use the Kubernetes API, so no `Role` or `RoleBinding`
is needed.
