#!/usr/bin/env bash
# Evaluate snapshots of the latest streaming run of one tokenizer: stream-watch.sh uci-move
set -euo pipefail
cd "$(dirname "$0")/.."
tok=$1
until d=$(ls -d "runs/stream-large-$tok"/*/ 2>/dev/null | sort | tail -1) && [ -n "$d" ] && [ -f "$d/config.yaml" ]; do sleep 30; done
exec "${UV:-uv}" run --no-sync python -m chessgpt.eval.watch "$d" --workers "${WATCH_WORKERS:-4}"
