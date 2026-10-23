# Gateway chart release checklist

The chart currently has `appVersion: 0.0.0-unreleased`. This is not a published
OpenViking version. Until a compatible release exists, `image.tag` is required
and must select a build containing all of the configured gateway features.
Do not point it at an older release or assume a mutable `main` tag has them.

For the first release, and each later gateway chart release:

1. Publish an OpenViking image that includes the chart's configuration fields,
   shared Redis management/kernel stores and Lua resources, shutdown handling,
   startup retry, management switch, tenancy, SSE heartbeats and request logging.
   The image build must pass its Redis import and installed Lua resource check.
2. Validate this chart against the release's `StandaloneGatewayConfig` with
   `PYTHONPATH=. python scripts/gateway_chart_check.py`. Run the gateway kind
   suite against that exact image and confirm shared state, restart recovery,
   graceful rollout and streaming behavior.
3. Set `Chart.yaml`'s `appVersion` to the published OpenViking version without
   `v`, bump the chart `version`, and re-run validation with no `image.tag`
   override. The default image becomes `ghcr.io/volcengine/openviking:v<appVersion>`.
4. Update release notes with visible behavior changes and storage compatibility.
   If the Redis key-layout version changes, document cache invalidation and
   reconfiguration of issued keys, upstreams and profiles before publishing.
5. Confirm the pinned image is available in every registry referenced by the
   deployment values. Package and publish the chart only after these checks.

The standalone chart evolves separately from the embedded `openviking` chart:
its replicas, RollingUpdate strategy, Service and gateway configuration belong
to a separate Deployment. Keep release names and selector labels stable during
upgrades so external routes and NetworkPolicies continue to select its Pods.
