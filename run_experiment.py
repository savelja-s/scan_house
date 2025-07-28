#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import joblib
import numpy as np
import subprocess
import time
from pathlib import Path
import pdal
import laspy


def parse_args():
    p = argparse.ArgumentParser(description="PDAL -> NumPy -> RF inference (без laspy, без chunk-ів).")
    p.add_argument("--pipeline", required=True)
    p.add_argument("--produced-las", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def run_pdal_cli(pipeline_path: Path) -> float:
    print(f"[INFO] Running PDAL pipeline: {pipeline_path}")
    t0 = time.time()
    subprocess.run(["pdal", "pipeline", str(pipeline_path)], check=True)
    dt = time.time() - t0
    print(f"[INFO] PDAL finished in {dt:.2f}s")
    return dt


def write_non_trees_las(las_in: Path, las_out: Path, mask_non: np.ndarray):
    """Запис LAS без дерев (класифікованих точок) через laspy."""
    las = laspy.read(las_in)
    keep_idx = np.where(mask_non)[0]
    las.points = las.points[keep_idx]
    las.write(las_out)
    print(f"[INFO] Записано LAS без дерев: {las_out} (точок: {len(las.points)})")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Виконуємо PDAL pipeline
    pdal_time = run_pdal_cli(Path(args.pipeline))

    # 2) Читаємо LAS як масив NumPy через PDAL
    pipe = pdal.Pipeline(json.dumps([{"type": "readers.las", "filename": args.produced_las}]))
    n = pipe.execute()
    if n == 0:
        print("[ERR] PDAL повернув 0 точок. Перевір шлях produced-las та writers.las у pipeline.")
        return
    arr = pipe.arrays[0]
    print(f"[INFO] Loaded points: {len(arr)}")

    # 3) Завантажуємо фічі та модель
    feature_cols = json.load(open(args.features))
    available = list(arr.dtype.names)
    missing = [f for f in feature_cols if f not in available]
    if missing:
        raise ValueError(f"У LAS немає полів: {missing}\nДоступні: {available}")

    X = np.column_stack([arr[f] for f in feature_cols])
    model = joblib.load(args.model)

    t0 = time.time()
    y_pred = model.predict(X)
    infer_time = time.time() - t0

    mask_non = (y_pred == 0)

    # 4) Записуємо LAS без дерев
    out_non = out_dir / "non_trees.las"
    write_non_trees_las(Path(args.produced_las), out_non, mask_non)

    # 5) Зберігаємо статистику
    stats = {
        "total_points": int(len(arr)),
        "pred_non_trees": int(mask_non.sum()),
        "pred_trees": int((~mask_non).sum()),
        "pct_non_trees": float(mask_non.mean()),
        "pct_trees": float((~mask_non).mean()),
        "pdal_time_sec": pdal_time,
        "infer_time_sec": infer_time,
        "non_trees_path": str(out_non)
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n===== SUMMARY =====")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("===================")


if __name__ == "__main__":
    main()
