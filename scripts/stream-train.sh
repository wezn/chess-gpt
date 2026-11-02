#!/usr/bin/env bash
# Start the round-robin streaming trainer, resuming the latest group if one exists.
set -euo pipefail
cd "$(dirname "$0")/.."
stamp=$(ls -d runs/stream-large-pgn-char/*/ 2>/dev/null | xargs -n1 basename 2>/dev/null | sort | tail -1 || true)
if [ -n "$stamp" ] && [ -f "runs/stream-large-pgn-char/$stamp/group.yaml" ]; then
  echo "resuming group $stamp" >&2
  exec "${UV:-uv}" run --no-sync python -m chessgpt.stream_train --resume "$stamp"
fi
exec "${UV:-uv}" run --no-sync python -m chessgpt.stream_train configs/experiments/stream-large.yaml
