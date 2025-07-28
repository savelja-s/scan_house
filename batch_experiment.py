#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import joblib
import numpy as np
import time
from pathlib import Path
import pdal
import laspy
import os
import sys
import signal
import psutil
from multiprocessing import Pool, cpu_count, get_context


# ------------------- Shutdown Handler -------------------
def shutdown(signum, frame):
    print("\n⏹ STOP ALL PROCESSES", file=sys.stderr)
    pid = os.getpid()
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            print(f"  - Killing child PID {child.pid}", file=sys.stderr)
            try:
                child.kill()
            except Exception as e:
                print(f"    (Already terminated or cannot kill: {e})", file=sys.stderr)
        parent.kill()
    except Exception as e:
        print(f"Error killing processes: {e}", file=sys.stderr)
    sys.exit(1)


for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, shutdown)


# ------------------- PDAL pipeline -------------------
def run_pdal_to_numpy(tile_path: Path):
    """Читає LAS через PDAL з потрібними фільтрами та повертає масив NumPy."""
    pipeline = [
        {"type": "readers.las", "filename": str(tile_path)},
        {"type": "filters.assign", "assignment": "Classification[:]=0"},
        {
            "type": "filters.smrf",
            "ignore": "Classification[7:7]",
            "scalar": 1.5,
            "slope": 0.3,
            "threshold": 0.3,
            "window": 4.0
        },
        {"type": "filters.hag_nn", "count": 12},
        {"type": "filters.normal", "knn": 24},
        {"type": "filters.radialdensity", "radius": 1.5}
    ]
    t0 = time.time()
    pipe = pdal.Pipeline(json.dumps(pipeline))
    n = pipe.execute()
    arr = pipe.arrays[0]
    return arr, time.time() - t0


# ------------------- Classification -------------------
def classify_and_write(arr, model, feature_cols, out_file: Path):
    """Класифікує точки та записує non-trees у LAS."""
    available = list(arr.dtype.names)
    missing = [f for f in feature_cols if f not in available]
    if missing:
        raise ValueError(f"Відсутні колонки: {missing}\nДоступні: {available}")

    # Формуємо матрицю X
    X = np.column_stack([arr[f] for f in feature_cols])

    # Інференс
    t0 = time.time()
    y_pred = model.predict(X)
    infer_time = time.time() - t0

    # Фільтр non-trees
    mask_non = (y_pred == 0)
    arr_non = arr[mask_non]

    # Записуємо у LAS
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x = arr_non["X"]
    las.y = arr_non["Y"]
    las.z = arr_non["Z"]
    las.intensity = arr_non["Intensity"]

    # Якщо є колір
    if "Red" in arr_non.dtype.names:
        las.red = arr_non["Red"].astype(np.uint16)
    if "Green" in arr_non.dtype.names:
        las.green = arr_non["Green"].astype(np.uint16)
    if "Blue" in arr_non.dtype.names:
        las.blue = arr_non["Blue"].astype(np.uint16)

    las.write(out_file)

    return mask_non, infer_time


# ------------------- Processing -------------------
def process_tile(tile_path: Path, args, feature_cols, model):
    try:
        print(f"[PID {os.getpid()}] Обробка тайла: {tile_path.name}")

        out_dir = Path(args.out_dir)
        out_file = out_dir / f"{tile_path.stem}_non_trees.las"
        stats_file = out_dir / f"{tile_path.stem}_stats.json"

        # 1) PDAL → NumPy
        arr, pdal_time = run_pdal_to_numpy(tile_path)

        # 2) Класифікація
        mask_non, infer_time = classify_and_write(arr, model, feature_cols, out_file)

        # 3) Статистика
        total_points = int(len(arr))
        non_points = int(mask_non.sum())
        stats = {
            "tile": tile_path.name,
            "total_points": total_points,
            "pred_non_trees": non_points,
            "pred_trees": total_points - non_points,
            "pct_non_trees": float(non_points / total_points) if total_points else 0.0,
            "pct_trees": float(1.0 - (non_points / total_points)) if total_points else 0.0,
            "pdal_time_sec": float(pdal_time),
            "infer_time_sec": float(infer_time),
            "out_non_trees": str(out_file)
        }
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        return stats
    except Exception as e:
        print(f"[ERR] Тайл {tile_path.name}: {e}", file=sys.stderr)
        return None


# ------------------- Main -------------------
def parse_args():
    p = argparse.ArgumentParser(description="Batch PDAL → RF inference (fast, no temp LAS)")
    p.add_argument("--tiles-dir", required=True, help="Папка з LAS/LAZ тайлами")
    p.add_argument("--model", required=True, help="Шлях до моделі .joblib")
    p.add_argument("--features", required=True, help="JSON зі списком ознак")
    p.add_argument("--out-dir", required=True, help="Вихідна папка")
    p.add_argument("--workers", type=int, default=6, help="Кількість паралельних процесів (за замовчуванням 6)")
    return p.parse_args()


def main():
    args = parse_args()
    tiles_dir = Path(args.tiles_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    las_files = sorted(list(tiles_dir.glob("*.las")) + list(tiles_dir.glob("*.laz")))
    if not las_files:
        print(f"[ERR] У папці {tiles_dir} немає LAS/LAZ файлів.")
        return

    feature_cols = json.load(open(args.features))
    model = joblib.load(args.model)
    workers = min(args.workers, cpu_count())

    print(f"[INFO] Запускаємо {workers} процесів...")

    with get_context("spawn").Pool(workers) as pool:
        results = pool.starmap(
            process_tile,
            [(tile, args, feature_cols, model) for tile in las_files]
        )

    print("\n=== SUMMARY ===")
    for res in results:
        if res:
            print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
