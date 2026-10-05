#!/usr/bin/env bash
# PixelCatcher (hidden wind + fog) experiment. Rerunning resumes finished stages.
#
#   bash scripts/run_catcher.sh world        # collect -> vae -> encode -> mdn -> heldout -> baselines
#   bash scripts/run_catcher.sh controllers  # real V+M, real V, dream V+M (tau 1.0, 1.25), each tested in the real game
#
# Environment overrides: CONFIG, PY, DEVICE (for V and M).
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/catcher.toml}"
PY="${PY:-.venv/bin/python}"
DEVICE="${DEVICE:-auto}"
WORKDIR="$("$PY" -c "import tomllib, sys; print(tomllib.load(open(sys.argv[1], 'rb'))['experiment']['workdir'])" "$CONFIG")"
mkdir -p "$WORKDIR"

run() {
  local stage="$1" log="$2"; shift 2
  echo "== $stage $* | $(date '+%F %T')" | tee -a "$WORKDIR/$log.log"
  "$PY" -u -m world_models.catcher_pipeline "$stage" --config "$CONFIG" "$@" 2>&1 \
    | { grep --line-buffered -v -i "warn" || true; } | tee -a "$WORKDIR/$log.log"
}

case "${1:-}" in
  world)
    run collect collect
    run vae vae --device "$DEVICE"
    run encode encode --device "$DEVICE"
    run mdn mdn --device "$DEVICE"
    run heldout heldout --device "$DEVICE"
    run baselines baselines
    ;;
  controllers)
    for spec in "real_vm --mode real" "real_v --mode real --no-memory" \
                "dream_vm_t1 --mode dream --temperature 1.0" "dream_vm_t1.25 --mode dream --temperature 1.25"; do
      set -- $spec; name="$1"; shift
      run controller "controller_$name" "$@"
      run evaluate "evaluate_$name" "$@"
    done
    ;;
  *) sed -n '2,7p' "$0"; exit 2 ;;
esac
echo "== done | $(date '+%F %T')"
