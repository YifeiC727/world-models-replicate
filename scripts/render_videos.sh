#!/usr/bin/env bash
# Render the showcase videos into media/ (needs the viz extra and trained models:
# runs/carracing-deepdish from the server, runs/catcher from scripts/run_catcher.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"

render() { echo "== $* | $(date '+%T')"; "$PY" -m world_models.visualize "$@" --gif 2>&1 | grep -v -i warn || true; }

render carracing-real --mode vm --seed 1000004               # best local V+M track (908.7)
render carracing-compare --seed 1000008                      # largest V only vs V+M gap (277.6 vs 882.5)
render carracing-dream --seed 1000004 --temperature 1.0
render catcher --seed 1000003 --temperature 1.25
ls -la media/
