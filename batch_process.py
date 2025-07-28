#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Batch processing of LAS tiles using run_experiment.py")
    p.add_argument("--tiles-dir", required=True, help="Папка з вхідними тайлами LAS/LAZ")
    p.add_argument("--pipeline", required=True, help="PDAL pipeline для підготовки feature-ів")
    p.add_argument("--model", required=True, help="Шлях до моделі .joblib")
    p.add_argument("--features", required=True, help="JSON з переліком ознак")
    p.add_argument("--out-dir", required=True, help="Вихідна папка")
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

    for las_file in las_files:
        print(f"\n=== Обробка файла: {las_file} ===")
        tile_out_dir = out_dir / las_file.stem
        tile_out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", "run_experiment.py",
            "--pipeline", args.pipeline,
            "--produced-las", str(las_file),
            "--model", args.model,
            "--features", args.features,
            "--out-dir", str(tile_out_dir)
        ]

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
