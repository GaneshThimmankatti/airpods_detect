#!/bin/bash
# Entry point for every part of the project. Usage:
#   ./run.sh enroll  [args]   build embeddings from enroll_photos/
#   ./run.sh detect  [args]   run the webcam detector directly
#   ./run.sh watch   [args]   run the Bluetooth watcher in the foreground
#   ./run.sh status           is the configured device connected right now?
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "No virtualenv found. Run ./install.sh first." >&2
  exit 1
fi

cmd="${1:-}"
shift || true

case "$cmd" in
  enroll) exec "$PY" "$ROOT/src/enroll.py" "$@" ;;
  detect) exec "$PY" "$ROOT/src/detector.py" "$@" ;;
  watch)  exec "$PY" "$ROOT/src/bt_watcher.py" "$@" ;;
  status) exec "$PY" "$ROOT/src/bt_watcher.py" --once ;;
  *)
    sed -n '2,6p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
