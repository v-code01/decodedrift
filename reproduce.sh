#!/usr/bin/env bash
# Regenerate the decode-drift sweep for both models. Servers must run with a large single-slot
# context on the given ports. Usage: ./reproduce.sh [PORT_15B] [PORT_05B]
set -euo pipefail
cd "$(dirname "$0")"
P15="${1:-8001}"; P05="${2:-8002}"
. .venv/bin/activate
caffeinate -dimsu python tools/run_sweep.py --port15 "$P15" --port05 "$P05" --npred 6000 --reps 3
python tools/analyze.py
python tools/verify.py
echo "regenerated results; see bench_results/frontier.md"
