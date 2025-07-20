#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Використання: $0 <шлях_до_LAS> [розмір_тайла] [перекриття]" >&2
  exit 1
fi

INPUT="$1"
TILE_SIZE="${2:-100}"     # за замовчуванням 100
BUFFER="${3:-10}"         # за замовчуванням 10

# Витягуємо базову назву файлу без розширення
BASE=$(basename "${INPUT%.*}")

OUTDIR="outputs/${BASE}"
mkdir -p "${OUTDIR}"

echo "Розбиття '${INPUT}' → тайли ${TILE_SIZE}×${TILE_SIZE} з перекриттям ${BUFFER}…"

pdal  tile \
  -i "${INPUT}" \
  -o "${OUTDIR}/tile_#.laz" \
  --length "${TILE_SIZE}" \
  --buffer "${BUFFER}"

echo "Готово — тайли збережено в ${OUTDIR}/"
