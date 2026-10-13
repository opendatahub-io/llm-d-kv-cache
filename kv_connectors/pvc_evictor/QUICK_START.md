# Quick Start Guide

Deploy the Python PVC Evictor with Helm to automatically manage disk space on your vLLM KV-cache PVC.
For the Go implementation, use the manifests in [deploy/](deploy/).

## Prerequisites

- Kubernetes cluster with `kubectl` access
- Helm 3.0+ installed
- Existing PVC with vLLM cache files
- Security context values for your namespace (see below)

## Installation

### Python deployment with Helm

#### 1. Get Security Context Values

**Important:** The evictor must use the same security context as your vLLM pods to access the shared PVC. Find your vLLM pod's security context:

```bash
# Replace <vllm-pod> with your actual vLLM pod name
kubectl get pod <vllm-pod> -n <namespace> -o jsonpath='{.spec.securityContext}'
kubectl get pod <vllm-pod> -n <namespace> -o jsonpath='{.spec.containers[0].securityContext}'
```

You need three values:
- **fsGroup**: Group ownership for mounted volumes (must match vLLM pods)
- **seLinuxOptions.level**: SELinux label for OpenShift multi-tenancy (must match vLLM pods)
- **runAsUser**: User ID for container processes (must match vLLM pods)

#### 2. Install with Helm

**Quick install:**
```bash
helm install pvc-evictor ./helm \
  --set pvc.name=my-vllm-cache \
  --set securityContext.pod.fsGroup=1000960000 \
  --set securityContext.pod.seLinuxOptions.level="s0:c31,c15" \
  --set securityContext.container.runAsUser=1000960000
```

**Using values file (recommended):**

Create `my-values.yaml`:
```yaml
pvc:
  name: my-vllm-cache

securityContext:
  pod:
    fsGroup: 1000960000
    seLinuxOptions:
      level: "s0:c31,c15"
  container:
    runAsUser: 1000960000

config:
  cleanupThreshold: 85.0
  targetThreshold: 70.0
  numCrawlerProcesses: 8
```

Install with the values file:
```bash
helm install pvc-evictor ./helm -f my-values.yaml
```

#### 3. Verify Python Deployment

```bash
# Check status
helm status pvc-evictor
kubectl get pods -l app.kubernetes.io/name=pvc-evictor

# View logs
kubectl logs -f deployment/pvc-evictor-pvc-evictor
```

### Go deployment with manifests

The Go implementation uses the manifests in `deploy/`. Helm is not used. Run
the commands below from the `deploy/` directory. See
[deploy/README.md](deploy/README.md) for details.

#### 1. Edit the manifests

- In `kustomization.yaml`, replace `namespace: my-namespace` with the namespace where the vLLM workload and the shared PVC are deployed.
- In `deployment.yaml`, replace `claimName: my-vllm-cache-pvc` with the name of the existing PVC. To use NFS or a CSI volume, replace the
  `persistentVolumeClaim` entry with that volume source.
- In `configmap.env`, change Go settings such as `CLEANUP_THRESHOLD`, `TARGET_THRESHOLD` or `CACHE_DIRECTORY`. Applying new values restarts the Pod.
- The default image is `quay.io/opendatahub/llm-d-kv-cache:odh-stable-go`. To use another tag, change `newTag` in `kustomization.yaml`. PR images use tag `odh-pr-go-<full-commit-sha>`, with the full 40-character commit SHA.

The Pod must be able to read and delete the files that vLLM writes.
The manifests do not set a user ID or `fsGroup`. 
If vLLM runs as a different user or on OpenShift with an SELinux level, add `patches/security-and-affinity-example.yaml` to the `patches` list in `kustomization.yaml` and replace its example values.
The patch also has a Pod affinity rule for `ReadWriteOnce` volumes.

#### 2. Apply

```bash
kubectl apply -k .
```

The Go process opens no network port by default, so the manifests have no `Service`. 
If you set `STORAGE_EVENTS_ENDPOINT`, the process listens on that address. In that case, add a `containerPort` and a `Service` yourself.

#### 3. Verify Go Deployment

```bash
# Use the same namespace as the vLLM workload.
NAMESPACE=my-namespace
kubectl get deployment llm-d-kv-cache-pvc-evictor -n "$NAMESPACE"
kubectl get pods -l app.kubernetes.io/name=llm-d-kv-cache-pvc-evictor -n "$NAMESPACE"
kubectl logs -f deployment/llm-d-kv-cache-pvc-evictor -n "$NAMESPACE"
```

Key Go log messages:
- `Deletion activated` - Deletion started (usage >= cleanup threshold)
- `Deletion deactivated` - Deletion stopped (usage <= target threshold)
- `Crawler status` and `Deleter status` - Statistics (every 30 seconds)

## Configuration

Key settings (all optional, defaults shown):

| Setting | Default | Description |
|---------|---------|-------------|
| `cleanupThreshold` | 85.0 | Disk usage % to trigger deletion |
| `targetThreshold` | 70.0 | Disk usage % to stop deletion |
| `numCrawlerProcesses` | 8 | Parallel file discovery (1, 2, 4, 8, or 16) |
| `cacheDirectory` | `kv/model-cache/models` | Cache path relative to PVC mount |
| `fileAccessTimeThresholdMinutes` | 60 | Protect files accessed within N minutes |


For complete configuration reference, see [CONFIGURATION.md](CONFIGURATION.md).

## Python monitoring

Watch logs in real-time:
```bash
kubectl logs -f deployment/pvc-evictor-pvc-evictor
```

Key log patterns:
- `DELETION_START` - Deletion triggered (usage >= cleanup threshold)
- `DELETION_END` - Deletion stopped (usage <= target threshold)
- `System Status` - Aggregated statistics (every 30 seconds)

## Python troubleshooting

**Pod not starting?**
- Verify PVC exists: `kubectl get pvc <pvc-name>`
- Check security context matches namespace SCC
- Review pod events: `kubectl describe pod <pod-name>`

**No files being deleted?**
- Check disk usage hasn't reached cleanup threshold
- Verify cache directory path is correct
- Check logs for crawler discovery: `kubectl logs <pod> | grep discovered`

**Need more help?**
- See [README.md](README.md) for architecture details
- See [CONFIGURATION.md](CONFIGURATION.md) for all settings
- See [deploy/README.md](deploy/README.md) for Go deployment details
- See [helm/README.md](helm/README.md) for Helm chart details
