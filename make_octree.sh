#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Використання: $0 <las/laz файл> [max_threads]"
    exit 1
fi

INPUT="$1"
BASENAME=$(basename "$INPUT" .las)
BASENAME=$(basename "$BASENAME" .laz)
OUTPUT="${BASENAME}_ept"

if command -v nproc >/dev/null 2>&1; then
    CPU_TOTAL=$(nproc)
    THREADS=$((CPU_TOTAL > 2 ? CPU_TOTAL - 2 : 1))
else
    CPU_TOTAL=$(sysctl -n hw.ncpu)
    THREADS=$((CPU_TOTAL > 2 ? CPU_TOTAL - 2 : 1))
fi

if [ ! -z "$2" ]; then
    THREADS="$2"
fi

entwine build \
  -i "$INPUT" \
  -o "$OUTPUT" \
  --threads "$THREADS"

echo "Octree (EPT) успішно згенеровано в директорії: $OUTPUT"
