#!/usr/bin/env bash
# Compatibility entrypoint. Remove after two plugin release cycles.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
if [ -f "$SCRIPT_DIR/../plugin-shared/install.sh" ]; then
  exec bash "$SCRIPT_DIR/../plugin-shared/install.sh" "$@"
fi

installer=$(mktemp "${TMPDIR:-/tmp}/ov-plugin-install.XXXXXX")
trap 'rm -f "$installer"' EXIT
url="${OPENVIKING_SHARED_INSTALL_URL:-https://raw.githubusercontent.com/volcengine/OpenViking/main/examples/plugin-shared/install.sh}"
tos_base="${OPENVIKING_TOS_BASE:-https://ovrelease.tos-cn-beijing.volces.com}"
curl -fsSL --connect-timeout 10 --max-time 60 -o "$installer" "$url" ||
  curl -fsSL --connect-timeout 10 --max-time 60 -o "$installer" "${tos_base%/}/plugin-shared/install.sh"
bash "$installer" "$@"
