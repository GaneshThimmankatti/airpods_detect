#!/bin/bash
# Set up the virtualenv and (optionally) the login LaunchAgent.
#   ./install.sh                 core deps only
#   ./install.sh --with-facenet  also install torch + facenet-pytorch
#   ./install.sh --launchagent   also install + load the login watcher
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABEL="com.shreya.birthday-surprise.watcher"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

WITH_FACENET=0
WITH_AGENT=0
for arg in "$@"; do
  case "$arg" in
    --with-facenet) WITH_FACENET=1 ;;
    --launchagent)  WITH_AGENT=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 1 ;;
  esac
done

# --- Python toolchain -------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  if [[ -x "$HOME/.local/bin/uv" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    echo "==> installing uv (no sudo, lands in ~/.local/bin)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

echo "==> creating virtualenv"
uv venv --python 3.12 "$ROOT/.venv"

echo "==> installing core dependencies"
uv pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/requirements.txt"

if [[ "$WITH_FACENET" == 1 ]]; then
  echo "==> installing facenet backend (large: torch + torchvision)"
  uv pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/requirements-facenet.txt"
fi

mkdir -p "$ROOT/enroll_photos" "$ROOT/media" "$ROOT/data" "$ROOT/logs"
chmod +x "$ROOT/run.sh"

# --- LaunchAgent ------------------------------------------------------------
if [[ "$WITH_AGENT" == 1 ]]; then
  echo "==> installing LaunchAgent $LABEL"
  mkdir -p "$HOME/Library/LaunchAgents"
  sed "s|__ROOT__|$ROOT|g" "$ROOT/launchagent/watcher.plist.template" > "$PLIST"
  launchctl bootout "gui/$UID/$LABEL" 2>/dev/null || true
  launchctl bootstrap "gui/$UID" "$PLIST"
  launchctl enable "gui/$UID/$LABEL"
  echo "    loaded. logs: $ROOT/logs/watcher.{out,err}.log"
fi

echo
echo "Done. Next:"
echo "  1. Put 5-15 photos of the birthday girl in $ROOT/enroll_photos/"
echo "  2. Put the surprise video at $ROOT/media/surprise.mp4"
echo "  3. ./run.sh detect --selftest     # grants Camera permission"
echo "  4. ./run.sh enroll"
echo "  5. ./run.sh detect                # try it live"
