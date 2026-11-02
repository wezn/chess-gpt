#!/usr/bin/env bash
# Install (or refresh) the pipeline as systemd user services. They survive logouts (linger is on)
# and restart on failure. Logs stay in data/logs/. Manage with:
#   systemctl --user status 'chess-*'            journalctl --user -u chess-train -f
#   systemctl --user stop chess-train             systemctl --user restart 'chess-watch@*'
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p ~/.config/systemd/user
for f in scripts/systemd/*.service; do
  sed "s#@ROOT@#$PWD#g; s#@UV@#$(command -v uv)#g" "$f" > ~/.config/systemd/user/"$(basename "$f")"
done
systemctl --user daemon-reload
systemctl --user enable --now chess-pull chess-tokenize chess-train chess-watch@pgn-char chess-watch@uci-move chess-watch@uci-square
systemctl --user --no-pager status 'chess-*' | grep -E "●|Active:"
