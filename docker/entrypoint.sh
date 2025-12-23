#!/usr/bin/env bash
set -e

# Safety: ensure nounset is OFF while we source Pixi’s activation hook
set +u

# Activate the selected environment
# shellcheck disable=SC1091
source /shell-hook

# If a command was provided, run it; otherwise open an interactive shell
if [ "$#" -gt 0 ]; then
  exec "$@"
else
  exec bash -i
fi
