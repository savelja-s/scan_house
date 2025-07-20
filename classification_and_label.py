import os
import glob
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
import pdal


def run_pipeline(pipeline_def: dict) -> int:
    """Запускає PDAL pipeline та повертає кількість оброблених точок."""
    pipeline = pdal.Pipeline(json.dumps(pipeline_def))
    return pipeline.execute()


def process_tile(in_file: str, base_out: str, params) -> None:
    """
    Поетапна обробка одного тайла:
      1) Видалення ґрунту
      2) Видалення вегетації (PMF)
      3) Маркування будівель (Class=6 за HAG)
    Кожен результат зберігається в окремій папці.
    """
    base_name = os.path.splitext(os.path.basename(in_file))[0]

    # 1) Ground removal
    stage1_dir = os.path.join(base_out, "ground_removed")
    os.makedirs(stage1_dir, exist_ok=True)
    ground_file = os.path.join(stage1_dir, f"{base_name}_no_ground.laz")
    pipeline1 = {
        "pipeline": [
            {"type": "readers.las", "filename": in_file},
            {"type": "filters.smrf", "scalar": params.scalar,
             "slope": params.slope, "threshold": params.threshold,
             "window": params.window},
            {"type": "filters.range", "limits": "Classification![2:2]"},
            {"type": "writers.las", "filename": ground_file}
        ]
    }
    try:
        cnt1 = run_pipeline(pipeline1)
        print(f"[{base_name}][ground_removed] pts={cnt1} → {ground_file}")
    except Exception as e:
        print(f"[{base_name}][ground_removed] ERROR: {e}")
        return

    # 2) Vegetation removal via PMF (use initial_distance instead of threshold)
    stage2_dir = os.path.join(base_out, "vegetation_removed")
    os.makedirs(stage2_dir, exist_ok=True)
    veg_file = os.path.join(stage2_dir, f"{base_name}_no_veg.laz")
    pipeline2 = {
        "pipeline": [
            {"type": "readers.las", "filename": ground_file},
            {"type": "filters.pmf", "max_window_size": params.pmf_max,
             "slope": params.pmf_slope, "initial_distance": params.pmf_initial},
            {"type": "writers.las", "filename": veg_file}
        ]
    }
    try:
        cnt2 = run_pipeline(pipeline2)
        print(f"[{base_name}][vegetation_removed] pts={cnt2} → {veg_file}")
    except Exception as e:
        print(f"[{base_name}][vegetation_removed] ERROR: {e}")
        return

    # 3) Label buildings (Class=6) based on HAG
    stage3_dir = os.path.join(base_out, "labeled")
    os.makedirs(stage3_dir, exist_ok=True)
    labeled_file = os.path.join(stage3_dir, f"{base_name}_labeled.laz")
    pipeline3 = {
        "pipeline": [
            {"type": "readers.las", "filename": veg_file},
            {"type": "filters.hag_nn"},
            {"type": "filters.assign",
             "value": f"Classification = 6 WHERE HAG > {params.hag_thresh}"},
            {"type": "writers.las", "filename": labeled_file}
        ]
    }
    try:
        cnt3 = run_pipeline(pipeline3)
        print(f"[{base_name}][labeled] pts={cnt3} → {labeled_file}")
    except Exception as e:
        print(f"[{base_name}][labeled] ERROR: {e}")
        return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Модульна класифікація тайлів з проміжними файлами"
    )
    parser.add_argument("tiles_dir", help="Папка з тайлами outputs/<basename>/")
    parser.add_argument("--workers", type=int, default=4,
                        help="Кількість процесів (default:4)")
    # SMRF parameters
    parser.add_argument("--scalar", type=float, default=1.25, help="SMRF scalar")
    parser.add_argument("--slope", type=float, default=0.15, help="SMRF slope")
    parser.add_argument("--threshold", type=float, default=0.5, help="SMRF threshold")
    parser.add_argument("--window", type=float, default=16, help="SMRF window size")
    # PMF parameters
    parser.add_argument("--pmf-max", type=int, default=33, help="PMF max window size")
    parser.add_argument("--pmf-slope", type=float, default=1.0, help="PMF slope")
    parser.add_argument("--pmf-initial", type=float, default=0.15, help="PMF initial distance")
    # HAG threshold
    parser.add_argument("--hag-thresh", type=float, default=2.0,
                        help="Height above ground threshold for building classification")

    args = parser.parse_args()
    base_out = os.path.join(args.tiles_dir, "classification")
    os.makedirs(base_out, exist_ok=True)

    tile_files = sorted(glob.glob(os.path.join(args.tiles_dir, "tile_*.laz")))
    if not tile_files:
        print(f"Не знайдено тайлів у {args.tiles_dir}")
        return

    params = argparse.Namespace(
        scalar=args.scalar,
        slope=args.slope,
        threshold=args.threshold,
        window=args.window,
        pmf_max=args.pmf_max,
        pmf_slope=args.pmf_slope,
        pmf_initial=args.pmf_initial,
        hag_thresh=args.hag_thresh
    )

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for in_file in tile_files:
            executor.submit(process_tile, in_file, base_out, params)

    print(f"Класифікація завершена. Проміжні та фінальні файли в {base_out}")


if __name__ == "__main__":
    main()
