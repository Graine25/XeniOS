#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BROKER_PY="${REPO_ROOT}/tools/ios/xenios_ios_lldb_jit_broker.py"
# Escape backslashes/quotes so the path is safe inside LLDB double quotes.
BROKER_PY_LLDB="${BROKER_PY//\\/\\\\}"
BROKER_PY_LLDB="${BROKER_PY_LLDB//\"/\\\"}"
LLDBINIT_XCODE="${HOME}/.lldbinit-Xcode"

if [[ ! -f "${BROKER_PY}" ]]; then
  echo "Broker script not found: ${BROKER_PY}" >&2
  exit 1
fi

touch "${LLDBINIT_XCODE}"

BLOCK_BEGIN="# >>> xenios-ios-jit-broker >>>"
BLOCK_END="# <<< xenios-ios-jit-broker <<<"

tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT

awk -v b="${BLOCK_BEGIN}" -v e="${BLOCK_END}" '
  BEGIN { skipping=0 }
  $0==b { skipping=1; next }
  $0==e { skipping=0; next }
  !skipping { print }
' "${LLDBINIT_XCODE}" > "${tmp}"

cat >> "${tmp}" <<EOF
${BLOCK_BEGIN}
# Auto-load XeniOS iOS JIT broker module for Xcode LLDB sessions.
command script import "${BROKER_PY_LLDB}"
# Reset any stale hooks from prior LLDB sessions.
xenios-jit-broker-reset-hooks
# Stable mode: install scripted broker hook up front (single hook, one-shot detach).
xenios-jit-broker-install
${BLOCK_END}
EOF

mv "${tmp}" "${LLDBINIT_XCODE}"
echo "Updated ${LLDBINIT_XCODE} with XeniOS iOS JIT broker (stable one-shot mode)."
