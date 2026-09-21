# Harness plugins renamed to OpenViking

The plugins are named **OpenViking** to reflect the context database they connect to. Claude Code and Codex use `openviking@openviking`; the shared skill is `openviking`, and the diagnostic skill is `ov-plugin-doctor`. DSH publishes as `@openviking/dsh-plugin`. OpenCode writes its runtime log to `openviking.log`.

These changes ship in Claude Code 0.6.0, Codex 0.10.0, the hook-host plugin 0.4.0, DSH 0.5.0, OpenCode 0.4.0, and Agent Plugins 0.2.0. The pi patch version changes because its shared runtime moves; its identity and tools remain unchanged.

## Upgrade an existing installation

Run the [unified installer](../agent-integrations/01-overview.md) for the hosts you use. It removes the old plugin id before installing the new one, migrates managed hooks and MCP entries, and removes Cursor's old rule and skill. An existing OpenViking statusline is updated automatically. Custom statuslines remain unchanged unless you explicitly select replacement.

| Host | Required action |
|---|---|
| Claude Code installed from the repository marketplace | Update the marketplace. Claude Code 2.1.193+ uses the top-level marketplace `renames` map to update `enabledPlugins` and `pluginConfigs`. The remote `git-subdir` source may still require `/plugin install openviking@openviking` to populate the new cache. |
| Claude Code installed with the installer | Run the installer again so its locally generated marketplace manifest is refreshed. |
| All Claude Code installations | Update permission rules and hook matchers containing `mcp__plugin_openviking-memory_openviking__*` or `plugin:openviking-memory:openviking`. The marketplace rename does not rewrite them. |
| Claude Code older than 2.1.193 or managed settings | Upgrade or reinstall the plugin. Administrators must update managed settings themselves. |
| Codex / TraeCode CLI 2.0 | Run the installer again, trust the six hooks again, and approve MCP tools when first used under the new server name. |
| Cursor / TRAE / ZCode | Run the installer again to replace the old managed hooks, rules, skills, and configuration entries. |
| DSH | Run the installer again. It removes `@openviking/dsh-memory-plugin` from the selected profile and installs `@openviking/dsh-plugin`. |
| Hermes | New schemas expose six `openviking_*` tools: search, read, browse, remember, forget, and add_resource. Their `viking_*` aliases remain callable. |

The native Codex installer backs up `config.toml` before removing old plugin enablement and hook-trust tables. Per-tool approvals under `mcp_servers.openviking-memory.tools.*` are preserved for reference; they cannot grant approval to the new server identity. Compatible Codex-format clients are managed through their host CLI. The old TRAE CLI 1.0 integration is removed after its replacement is successfully enabled.

Old shared-runtime, marketplace, and DSH tarball directories are removed only when installed adapters, known host registrations, or DSH profiles no longer reference them. Hosts and profiles can be upgraded separately. Re-running the installer does not add duplicate managed hooks.

Run `ov-plugin-doctor` after upgrading to find old plugin IDs, MCP entries, permission rules, or duplicate Cursor rules and shared directories. Retained files may still serve another host; follow the diagnostic advice before deleting them.

## Compatibility and behavior

Codex's MCP prefix changes from `mcp__openviking_memory__` to `mcp__openviking__`. The server already recognizes the latter, so these calls now contribute to injection usage statistics and Experience lineage. Claude Code's prefix changes from `mcp__plugin_openviking-memory_openviking__` to `mcp__plugin_openviking_openviking__`.

Capture filters accept both `[openviking-memory]` and `[openviking]`. DSH recognizes both historical `source.plugin` values, and Hermes recognizes both generations of recall tool names, preventing recalled context from being captured again.

`OPENVIKING_MEMORY_ENABLED`, `openviking_memory_*` metrics, `ov-experience-memory`, `ov-memory-troubleshoot`, and the `OpenViking memory digest:` protocol header retain their names. OpenClaw and pi identities and tools, OpenWebUI's `ov_*` tools, historical changelogs, and frozen pi snapshots are outside this migration.

The old GitHub shared-installer path forwards to `examples/plugin-shared/install.sh` and is scheduled for removal after two release cycles. TOS publishes identical artifacts under both old and new keys. The old `plugins/memory-plugins.git` URL remains available because existing Codex installations store it as their marketplace source. Old raw GitHub links to individual moved plugin files do not redirect; update those links to the new directories.

## Release coordination

Before merging, publish `@openviking/dsh-plugin` once and configure npm trusted publishing for the new package. After merging, promptly publish the TOS release or dispatch `release-tos.yml` with `update_latest=true`; the website's agent cards publish on main and need the new installer URLs to exist. Deprecate the old DSH npm package after release without unpublishing it.

OpenClaw's remaining tool and skill names, pi's identities and tools, OpenWebUI's tool prefix, the digest protocol header, and workflow/job display names are deferred. Workflow display names must be coordinated with branch protection. Moving the shared directory also triggers an OpenClaw development release through the existing workflow path filter.
