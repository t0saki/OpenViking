# OpenViking Helm Chart

Deploy OpenViking on Kubernetes using Helm.

## Prerequisites

- Kubernetes 1.24+
- Helm 3.x
- A storage class that supports `ReadWriteOnce` persistent volumes (for RocksDB data)

## Installation

### Quick Start

```bash
helm install openviking ./deploy/helm/openviking \
  --set-string config.server.root_api_key="YOUR_ROOT_API_KEY" \
  --set-string config.embedding.dense.api_key="YOUR_VOLCENGINE_API_KEY" \
  --set-string config.vlm.api_key="YOUR_VOLCENGINE_API_KEY"
```

The chart deploys `ghcr.io/volcengine/openviking:latest` by default. To choose
a different image tag:

```bash
# newest image from the main branch
helm upgrade --install openviking ./deploy/helm/openviking --set image.tag=main

# pinned release image
helm upgrade --install openviking ./deploy/helm/openviking --set image.tag=v0.3.17

# use the default latest image
helm upgrade --install openviking ./deploy/helm/openviking --set image.tag=
```

We recommend the `ghcr.io` image. If `ghcr.io` is hard to reach, add
`--set image.repository=openviking-cn-beijing.cr.volces.com/volcengine/openviking`.

### Install with Custom Values

Create a `my-values.yaml` file:

```yaml
replicaCount: 1

resources:
  limits:
    cpu: "4"
    memory: 8Gi
  requests:
    cpu: "1"
    memory: 2Gi

persistence:
  size: 50Gi
  storageClass: "gp3"

config:
  storage:
    workspace: /app/.openviking/openviking_workspace
  log:
    level: INFO
    output: stdout
  server:
    host: "0.0.0.0"
    port: 1933
    workers: 1
    root_api_key: "your-secret-key"
  embedding:
    dense:
      api_base: "https://ark.cn-beijing.volces.com/api/v3"
      api_key: "your-volcengine-api-key"
      provider: "volcengine"
      dimension: 1024
      model: "doubao-embedding-vision-251215"
      input: "multimodal"
    max_concurrent: 10
  vlm:
    api_base: "https://ark.cn-beijing.volces.com/api/v3"
    api_key: "your-volcengine-api-key"
    provider: "volcengine"
    model: "doubao-seed-2-0-lite-260428"
    temperature: 0.0
    max_retries: 2
    thinking: false
    max_concurrent: 100
```

Then install:

```bash
helm install openviking ./deploy/helm/openviking -f my-values.yaml
```

### Using Secrets for API Keys

For production, avoid putting API keys directly in values. Use `extraEnv` with
Kubernetes secrets instead:

```bash
# Create a secret
kubectl create secret generic openviking-api-keys \
  --from-literal=root-api-key="YOUR_ROOT_API_KEY" \
  --from-literal=embedding-api-key="YOUR_KEY" \
  --from-literal=vlm-api-key="YOUR_KEY"
```

Then reference it in your values:

```yaml
config:
  server:
    root_api_key: "${OPENVIKING_ROOT_API_KEY}"
  embedding:
    dense:
      api_key: "${OPENVIKING_EMBEDDING_API_KEY}"
  vlm:
    api_key: "${OPENVIKING_VLM_API_KEY}"

extraEnv:
  - name: OPENVIKING_ROOT_API_KEY
    valueFrom:
      secretKeyRef:
        name: openviking-api-keys
        key: root-api-key
  - name: OPENVIKING_EMBEDDING_API_KEY
    valueFrom:
      secretKeyRef:
        name: openviking-api-keys
        key: embedding-api-key
  - name: OPENVIKING_VLM_API_KEY
    valueFrom:
      secretKeyRef:
        name: openviking-api-keys
        key: vlm-api-key
```

OpenViking expands environment variables inside `ov.conf` at startup, so the
ConfigMap can contain placeholders while the actual secrets stay in Kubernetes
Secrets.

### OpenViking Gateway

Set `gateway.enabled: true` to run [OpenViking Gateway](https://docs.openviking.ai/en/guides/15-gateway)
inside the server container. The chart adds `gateway.enabled` to `ov.conf` and stores
gateway data in `<persistence.mountPath>/gateway` on the persistent volume. There is no
sidecar and no extra Service port: clients use the server port, and the existing Ingress
forwards every path to OpenViking.

Model requests stream, so the Ingress must turn off response buffering and allow request
bodies of at least 32 MiB and timeouts of 600 seconds. The chart does not set these
annotations for you. With NGINX:

```yaml
gateway:
  enabled: true
  existingSecret: ""       # optional: Secret with an encryption-key entry
  managementKeySecret: ""  # optional: Secret with a management-key entry
config:
  server:
    auth_mode: api_key
  gateway:
    public_url: https://ov.example.com
ingress:
  enabled: true
  className: nginx
  annotations:
    nginx.ingress.kubernetes.io/proxy-buffering: "off"
    nginx.ingress.kubernetes.io/proxy-request-buffering: "off"
    nginx.ingress.kubernetes.io/proxy-body-size: 32m
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
  hosts:
    - host: ov.example.com
      paths:
        - path: /
          pathType: Prefix
```

Set `config.gateway.public_url` to the address clients use; Studio's setup guides and file
upload links are built from it.

#### Secrets

- `gateway.existingSecret` names a Secret with an `encryption-key` entry (a Fernet key),
  passed to the gateway as `OPENVIKING_GATEWAY_ENCRYPTION_KEY`. Without it, the gateway
  generates `encryption.key` in its storage directory on the data volume; back it up with
  the volume. If the databases exist but the key is missing, the whole OpenViking server
  refuses to start, and a different key cannot read the stored data.
- `gateway.managementKeySecret` names a Secret with a `management-key` entry (at least 32
  characters). The chart sets `config.gateway.management_key` to
  `$OPENVIKING_GATEWAY_MANAGEMENT_KEY` and fills that variable from the Secret. With it,
  only requests that carry this key can change upstreams and context profiles; see
  [Gateway management key](https://docs.openviking.ai/en/guides/22-gateway-operations#gateway-management-key).

```bash
kubectl create secret generic openviking-gateway \
  --from-literal=encryption-key="$(python3 -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
kubectl create secret generic openviking-gateway-management \
  --from-literal=management-key="$(openssl rand -base64 32)"
```

Keep `replicaCount: 1`: the gateway's SQLite databases serve one pod only. To use more
CPU, raise the server's `workers` setting.

#### Upgrading from the sidecar chart

- Keep `gateway.existingSecret` and the persistent volume. The Secret and its
  `encryption-key` entry are the same ones the sidecar used.
- Remove `gateway.port`, `gateway.workers` and `gateway.resources`.
- Remove `host`, `port`, `workers`, `url`, `openviking_url` and `admin_token_env` from
  `config.gateway`. Any of them stops the server at startup.
- Point `config.gateway.public_url` at the server's public address, not port 1935, and
  move clients to the server port.
- The shared admin token is gone and does not act as a management key. Use
  `gateway.managementKeySecret` to restrict changes to upstreams and context profiles.

For deployment, Studio management, backups and troubleshooting, see
[OpenViking Gateway deployment and operations](https://docs.openviking.ai/en/guides/22-gateway-operations).

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `1` |
| `image.repository` | Container image repository | `ghcr.io/volcengine/openviking` |
| `image.tag` | Container image tag (`latest`, `main`, pinned release, or empty for `latest`) | `latest` |
| `image.pullPolicy` | Image pull policy | `Always` |
| `service.type` | Kubernetes service type | `ClusterIP` |
| `service.port` | Service port | `1933` |
| `persistence.enabled` | Enable persistent storage | `true` |
| `persistence.size` | PVC size | `20Gi` |
| `persistence.storageClass` | Storage class name | `""` (default) |
| `persistence.mountPath` | Container path for OpenViking persistent state | `/app/.openviking` |
| `bot.enabled` | Start vikingbot alongside the API server | `false` |
| `persistence.existingClaim` | Use an existing PVC | `""` |
| `resources.limits.cpu` | CPU limit | `2` |
| `resources.limits.memory` | Memory limit | `4Gi` |
| `resources.requests.cpu` | CPU request | `500m` |
| `resources.requests.memory` | Memory request | `1Gi` |
| `ingress.enabled` | Enable ingress | `false` |
| `config.server.root_api_key` | API key required when server binds to 0.0.0.0 | `""` |
| `config` | Full ov.conf configuration object | See `values.yaml` |
| `extraEnv` | Additional environment variables | `[]` |
| `gateway.enabled` | Run OpenViking Gateway inside the server container, on the server port | `false` |
| `gateway.existingSecret` | Optional Secret with an `encryption-key` entry; without it, a key file is generated on the data volume | `""` |
| `gateway.managementKeySecret` | Optional Secret with a `management-key` entry; with it, changing upstreams and context profiles requires that key | `""` |

## Upgrading

```bash
helm upgrade openviking ./deploy/helm/openviking -f my-values.yaml
```

The deployment uses a `Recreate` strategy to avoid data corruption from
multiple pods accessing the same RocksDB volume simultaneously.

## Uninstalling

```bash
helm uninstall openviking
```

Note: The PersistentVolumeClaim is retained by Helm and becomes orphaned after
uninstall. Delete it explicitly to remove the stored data:

```bash
kubectl delete pvc openviking-data
```
