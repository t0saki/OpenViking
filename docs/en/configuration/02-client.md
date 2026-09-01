# ovcli Configuration

`ovcli.conf` is the client configuration file for the `ov` CLI. It stores the server connection, authentication identity, and command defaults.

Agent plugins for Codex, Claude Code, OpenCode, and other clients also read their own `OPENVIKING_*` environment variables for Recall, Capture, diagnostics, and other behavior. Those variables are not part of `ovcli.conf`; configure them in the corresponding [Agent Integration](../agent-integrations/01-overview.md) documentation.

Use `ov config` to create and maintain configurations. Use `ov config show` to inspect the active configuration with secrets redacted.

Default path:

```text
~/.openviking/ovcli.conf
```

To select another file:

```bash
export OPENVIKING_CLI_CONFIG_FILE=/path/to/ovcli.conf
```

## Complete Example

```json
{
  "url": "https://openviking.example.com",
  "api_key": "<user-or-admin-key>",
  "root_api_key": "<root-key>",
  "account": "acme",
  "user": "alice",
  "actor_peer_id": "agent:research-assistant",
  "timeout": 60,
  "output": "table",
  "echo_command": true,
  "show_progress": false,
  "verbose": false,
  "profile": false,
  "upload": {
    "ignore_dirs": "node_modules,.cache,dist",
    "include": "*.md,*.pdf",
    "exclude": "*.tmp,*.log"
  },
  "extra_headers": {
    "X-Tenant": "acme"
  },
  "gateway_token": "<gateway-token>"
}
```

Omit fields you do not need. A local server in `dev` mode usually needs only `url`.

## Connection and Authentication

```json
{
  "url": "https://openviking.example.com",
  "api_key": "<user-or-admin-key>",
  "root_api_key": "<root-key>",
  "account": "acme",
  "user": "alice",
  "actor_peer_id": "agent:research-assistant",
  "extra_headers": {
    "X-Tenant": "acme"
  },
  "gateway_token": "<gateway-token>"
}
```

| Field | Type / Values | Default | Purpose |
|---|---|---|---|
| `url` | HTTP(S) URL | `http://127.0.0.1:1933` | OpenViking server endpoint |
| `api_key` | string / `null` | `null` | User/admin key for normal data operations |
| `root_api_key` | string / `null` | `null` | Root key for `ov --sudo` administrative operations |
| `account` | string / `null` | `null` | Account identity for trusted or root-key-only configurations |
| `user` | string / `null` | `null` | User identity for trusted or root-key-only configurations |
| `actor_peer_id` | string / `null` | `null` | Default Actor Peer identifier |
| `agent_id` | string / `null` | `null` | Compatibility field; use `actor_peer_id` for new configs and do not set both |
| `extra_headers` | object / `null` | `null` | Additional headers sent with every request; `extra_header` is a compatibility alias |
| `gateway_token` | string / `null` | `null` | `X-Gateway-Token` used when retrying a gateway challenge |

### Choosing API Keys

| Configuration | Normal Commands | `ov --sudo` |
|---|---|---|
| `api_key` only | User/admin key | unavailable |
| `root_api_key` plus `account` and `user` | Root key with explicit identity | Root key |
| Both keys | `api_key` | `root_api_key` |
| No keys | Local server with authentication disabled only | unavailable |

`server.root_api_key` in `ov.conf` is accepted by the server. When the CLI manages that server, `root_api_key` in `ovcli.conf` must match it.

## Command Behavior

```json
{
  "timeout": 120,
  "echo_command": true,
  "show_progress": true,
  "verbose": false,
  "profile": false
}
```

| Field | Type / Values | Default | Purpose |
|---|---|---|---|
| `timeout` | number, seconds, `> 0` | `60` | HTTP request timeout |
| `echo_command` | boolean | `true` | Show effective request parameters for commands such as `find`, `search`, and `ls` |
| `show_progress` | boolean | `false` | Show upload progress by default |
| `verbose` | boolean | `false` | Show upload diagnostics by default |
| `profile` | boolean | `false` | Request performance profiles; also requires `server.profile_enabled` |
| `output` | `"table"` / `"json"` | `"table"` | Compatibility field; use `-o table` or `-o json` to select current command output |

Command-line options such as `--profile`, `--progress`, `--no-progress`, and `--verbose` override the configuration for the current command.

## Upload Filters

```json
{
  "upload": {
    "ignore_dirs": "node_modules,.cache,dist",
    "include": "*.md,*.pdf",
    "exclude": "*.tmp,*.log"
  }
}
```

| Field | Type / Format | Default | Purpose |
|---|---|---|---|
| `upload.ignore_dirs` | comma-separated string / `null` | `null` | Directory names to ignore |
| `upload.include` | comma-separated globs / `null` | `null` | Upload only matching files |
| `upload.exclude` | comma-separated globs / `null` | `null` | Exclude matching files |

Local directory uploads also honor `.gitignore`. Command-line `--include` and `--exclude` rules are merged with the configuration.

## Workspace Configuration

A repository can carry its own plugin settings, so the memory behavior of a project travels with the checkout instead of living in each contributor's home directory. Two files sit under the workspace root, and a third layer is kept per machine:

```text
<repo-root>/.openviking/config.json         # committed, shared by the team
<repo-root>/.openviking/config.local.json   # private, not committed
~/.openviking/workspaces/<slot>.json        # per-machine registry, one file per workspace
```

The workspace root is the nearest ancestor directory holding a `.git`; `$HOME` and the filesystem root are never workspace roots. The registry slot name combines the root's directory name with a hash of its full path, so two clones of one repository on one machine never share an entry. These layers are read by the Claude Code and Codex plugins, not by `ov` commands.

### Precedence

Highest first:

| Layer | Scope |
|---|---|
| `OPENVIKING_*` environment variables | Current process |
| `~/.openviking/workspaces/<slot>.json` | This machine, this workspace |
| `<repo-root>/.openviking/config.local.json` | This checkout, private |
| `<repo-root>/.openviking/config.json` | This repository, committed |
| `ovcli.conf` `plugin.<harness>` | This machine, one harness |
| `ovcli.conf` `plugin` | This machine, every harness |
| `ov.conf` harness section | Compatibility layer for older deployments |
| Built-in defaults | |

A scalar from a higher layer replaces the lower one. Lists are unioned across layers; a leading `"!reset"` element drops everything the lower layers contributed, so `["!reset", "*/scratch/*"]` is the whole list.

### Schema

`version: 1` is required. A file declaring another version is skipped with a warning rather than guessed at.

```json
{
  "version": 1,
  "peer": { "source": "git" },
  "recall": { "peer_scope": "actor", "max_items": 20 },
  "capture": { "commit_token_threshold": 20000 },
  "labels": { "team": "search" }
}
```

| Key | Type / Values | Purpose |
|---|---|---|
| `peer.source` | `"git"` / `"cwd"` / `"none"` / template / list of templates | How the workspace peer is derived |
| `peer.id` | string | Pin the peer explicitly; takes precedence over `peer.source` |
| `recall.enabled` | boolean | Whether the plugin recalls at all |
| `recall.peer_scope` | `"all"` / `"actor"` | Search every peer under this user, or only this workspace's peer |
| `recall.dedup_turns` | integer, `0`–`20` | Recent turns a recalled item is deduplicated against |
| `recall.max_items` | integer, `1`–`100` | Maximum recalled items |
| `recall.score_threshold` | number, `0`–`1` | Minimum score for a recalled item |
| `capture.enabled` | boolean | Whether the session is captured |
| `capture.commit_token_threshold` | integer, `1000`–`1000000` | Tokens accumulated before a capture commits |
| `bypass.session_patterns` | list of globs | A session whose id or working directory matches skips recall and capture |
| `labels` | object | Free-form metadata for humans; not read by the plugins |

An out-of-range number is clamped to the nearest bound and reported; an unrecognized enum value is ignored. Keys outside this list are kept in the file and ignored.

### Workspace Peer

`peer.source` decides which peer a workspace writes its memories under. The same setting is spelled `OPENVIKING_PEER_SOURCE` in the environment and `plugin.peerSource` or `plugin.<harness>.peerSource` in `ovcli.conf`.

| Value | Meaning |
|---|---|
| `"git"` | Default. The normalized `origin` URL, else the repository root path, else the working directory — equivalent to `["{git_remote}", "{git_root}", "{cwd}"]`. No prefix is added. |
| `"cwd"` | The working directory with every non-alphanumeric character replaced by `-`, byte for byte what earlier releases sent |
| `"none"` | Send no peer at all; `OPENVIKING_WORKSPACE_PEER=0` means the same |
| template / list of templates | For example `"git-{git_remote}"` or `["{git_remote}", "team-{dir}"]`; templates are tried in order, and one whose variables are empty falls through to the next |

| Variable | Value |
|---|---|
| `{git_remote}` | Normalized `origin`, as `github.com-org-repo`; empty outside a git repository or without an `origin` |
| `{git_root}` | Repository root path, with every non-alphanumeric character replaced by `-` |
| `{cwd}` | Working directory, with every non-alphanumeric character replaced by `-` |
| `{dir}` | The repository root's directory name |

In `/Users/x/Dev/OpenViking/examples/codex-memory-plugin` with `origin` `git@github.com:volcengine/OpenViking.git`, the peer is `github.com-volcengine-openviking` — the same value from any subdirectory, worktree, machine, or clone. Every clone of one repository therefore shares one peer, while a fork has a different `origin` and stays separate. The derivation reads the repository's files directly instead of running `git`, so it also works where `git` is missing from `PATH`, and the URL is normalized so that the ssh and https spellings of one repository agree and a token embedded in the URL never reaches the peer id.

Switching to the `git` default needs no migration: the pre-`git` peer id is recomputed locally, so recall still reaches memories written under it. With the default `recall.peer_scope: "all"` the server's sweep across the user's peers already covers it; under `"actor"` the plugin asks the old peer separately.

### What a Workspace File May Not Set

A hook runs without a prompt, so these files are trusted; what is refused is structural instead:

- Connection and credential keys — `url`, `api_key`, `root_api_key`, `account`, `user`, `extra_headers`, and the rest — are stripped with a warning wherever they appear. Which server the data goes to stays answerable from `ovcli.conf` and the environment alone.
- `${VAR}` is never expanded in these files.
- `cli_config_profile`, which names an `ovcli.conf` profile, is accepted only in the registry.

What a committed file switches off is announced rather than blocked: the plugin's `ov-memory-doctor` reports every workspace-scoped value, the layer it came from, and what it shadowed.

`.gitignore` must not ignore all of `.openviking/`, or `config.json` can never be committed. Narrow the rule to the parser's scratch directories and the private file:

```text
.openviking/media/
.openviking/downloads/
.openviking/config.local.json
```

`ov-memory-doctor` warns when a blanket rule is in effect.

## Related Environment Variables

The `ov` CLI directly uses only a small set of environment variables:

| Environment Variable | Purpose |
|---|---|
| `OPENVIKING_CLI_CONFIG_FILE` | Select the `ovcli.conf` path |
| `OPENVIKING_UPLOAD_MODE` | Select temporary upload mode: `local` or `shared` |

The `--api-key-env <name>` and `--root-api-key-env <name>` options for `ov config add` and `ov config edit` read keys from a named environment variable and write them to the configuration.

Variables such as `OPENVIKING_AUTO_RECALL`, `OPENVIKING_RECALL_LIMIT`, `OPENVIKING_AUTO_CAPTURE`, and `OPENVIKING_DEBUG` are read by Agent plugin processes and are not `ovcli.conf` fields.

## Multiple Servers

Normal `ov` commands, plus `ov config show` and `ov config validate`, resolve the effective configuration in this order:

1. When `OPENVIKING_CLI_CONFIG_FILE` is set, that path is authoritative; a missing file is an error.
2. When the variable is unset, the default active file:

```text
~/.openviking/ovcli.conf
```

The interactive manager and `ov config list`, `switch`, `add`, `edit`, and `delete` always manage the default store. Named configurations in that store live next to the default active file:

```text
~/.openviking/ovcli.conf.<name>
```

For example, a production configuration can contain:

```json
{
  "url": "https://openviking.example.com",
  "api_key": "<production-api-key>",
  "timeout": 120
}
```

Common commands:

```bash
ov config
ov config list
ov config switch <name>
ov config validate
ov config show
```

`ov config switch <name>` copies the named configuration to the default active file. If `OPENVIKING_CLI_CONFIG_FILE` remains set, normal `ov` commands continue to use the environment-selected file; unset it to use the switched default. New `ov` commands reread the effective file, while already-running Agent clients must restart before reading changes.

See [OpenViking CLI Setup](../getting-started/05-cli-setup.md) for interactive and agent-assisted configuration workflows.
