#!/usr/bin/env bash
# Run pipeline stages on the shared MLDS server at low priority on one GPU.
# Every stage is resumable, so after an interruption just rerun the same command.
#
#   bash scripts/run_server.sh world          # collect -> vae -> encode -> mdn -> heldout
#   bash scripts/run_server.sh controller vm  # CMA-ES + evaluation with [z, h]
#   bash scripts/run_server.sh controller v   # V-only ablation with z
#
# Environment overrides: CONFIG, PY, DEVICE, GPU (CUDA device index), WORKERS.
set -euo pipefail

CONFIG="${CONFIG:-configs/carracing-deepdish.toml}"
PY="${PY:-$HOME/envs/wm/bin/python}"
DEVICE="${DEVICE:-cuda}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
WORKDIR="$("$PY" -c "import tomllib, sys; print(tomllib.load(open(sys.argv[1], 'rb'))['experiment']['workdir'])" "$CONFIG")"
mkdir -p "$WORKDIR"

run() {
  local stage="$1"; shift; local log="$WORKDIR/$stage${TAG:-}.log"
  echo "== $stage $* | $(date '+%F %T')" | tee -a "$log"
  nice -n 10 "$PY" -u -m world_models.carracing_pipeline "$stage" --config "$CONFIG" "$@" 2>&1 \
    | { grep --line-buffered -v -e "Could not import matplotlib" -e "_warnings.warn" || true; } \
    | tee -a "$log"
}

case "${1:-}" in
  world)
    run collect ${WORKERS:+--workers "$WORKERS"}
    run vae --device "$DEVICE" --resume
    run encode --device "$DEVICE"
    run mdn --device "$DEVICE" --resume
    run heldout --device "$DEVICE"
    ;;
  controller)
    case "${2:-}" in
      vm) memory="" ;;
      v) memory="--no-memory" ;;
      *) echo "usage: $0 controller vm|v" >&2; exit 2 ;;
    esac
    TAG="_$2"
    run controller --device cpu --resume ${memory:+"$memory"} ${WORKERS:+--workers "$WORKERS"}
    run evaluate --device cpu ${memory:+"$memory"} ${WORKERS:+--workers "$WORKERS"}
    ;;
  *) sed -n '2,9p' "$0"; exit 2 ;;
esac
echo "== done | $(date '+%F %T')"
