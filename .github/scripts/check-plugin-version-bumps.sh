#!/usr/bin/env bash
# Fail when a marketplace-distributed plugin changed without its version string.
#
# Claude Code resolves an installed plugin by the `version` in its manifest
# rather than by the commit the marketplace ref points at, so a frozen version
# string means users keep running the build they already have no matter how
# many fixes land on main. Every other host in this list distributes by the same
# kind of manifest. The generated `shared/` copies live inside each plugin
# directory, so a shared-library change reaches this check through them.
#
# Usage: check-plugin-version-bumps.sh <base-ref>
set -euo pipefail

BASE_REF="${1:-}"
if [ -z "$BASE_REF" ]; then
  echo "usage: $0 <base-ref>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# <plugin directory>:<manifest holding the version>
PLUGINS=(
  "examples/claude-code-memory-plugin:examples/claude-code-memory-plugin/.claude-plugin/plugin.json"
  "examples/codex-memory-plugin:examples/codex-memory-plugin/.codex-plugin/plugin.json"
  "examples/zcode-memory-plugin:examples/zcode-memory-plugin/.zcode-plugin/plugin.json"
  "examples/opencode-plugin:examples/opencode-plugin/package.json"
  "examples/dsh-memory-plugin:examples/dsh-memory-plugin/package.json"
)

read_version() { # read_version <ref-or-empty> <path>
  local ref="$1" path="$2" json
  if [ -n "$ref" ]; then
    json="$(git show "$ref:$path" 2>/dev/null)" || return 1
  else
    json="$(cat "$path")"
  fi
  printf '%s' "$json" | node -e '
let raw = "";
process.stdin.on("data", (c) => { raw += c; });
process.stdin.on("end", () => {
  try { process.stdout.write(String(JSON.parse(raw).version || "")); }
  catch { process.exit(1); }
});
'
}

# A plugin that carries both a host manifest and an installer manifest has to
# say the same version in both: the host installs by one and the installer
# decides "nothing changed" by the other, so a mismatch means a plugin that
# reports upgraded and behaves like it did not.
PAIRED=(
  "examples/cursor-memory-plugin/.cursor-plugin/plugin.json:examples/cursor-memory-plugin/openviking.integration.json"
  "examples/zcode-memory-plugin/.zcode-plugin/plugin.json:examples/zcode-memory-plugin/openviking.integration.json"
)

failed=0
for entry in "${PAIRED[@]}"; do
  host="${entry%%:*}"
  integration="${entry#*:}"
  [ -f "$host" ] && [ -f "$integration" ] || continue
  host_version="$(read_version "" "$host")"
  integration_version="$(read_version "" "$integration")"
  if [ "$host_version" != "$integration_version" ]; then
    echo "::error file=$host::$host says $host_version and $integration says $integration_version."
    failed=1
  fi
done

for entry in "${PLUGINS[@]}"; do
  dir="${entry%%:*}"
  manifest="${entry#*:}"

  changed="$(git diff --name-only "$BASE_REF...HEAD" -- "$dir" | grep -v '/node_modules/' || true)"
  [ -n "$changed" ] || continue

  # A plugin added in this branch has no baseline version to compare against.
  before="$(read_version "$BASE_REF" "$manifest")" || continue
  after="$(read_version "" "$manifest")"

  if [ "$before" = "$after" ]; then
    echo "::error file=$manifest::$dir changed but its version is still $after."
    echo "  Hosts install by this string, so users receive nothing until it moves."
    echo "  Files changed:"
    printf '    %s\n' $changed
    failed=1
  else
    echo "ok: $dir  $before -> $after"
  fi
done

exit "$failed"
