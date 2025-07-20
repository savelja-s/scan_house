import os
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import pdal
import signal
import sys
import psutil


def cluster_tile(in_file: str, out_dir: str, min_points: int, tolerance: float) -> None:
    """
    Виконує кластеризацію точок Class=6 в одному файлі та пише результат.
    """
    base = os.path.splitext(os.path.basename(in_file))[0]
    out_file = os.path.join(out_dir, f"{base}_cluster.laz")

    pipeline_def = {
        "pipeline": [
            {"type": "readers.las", "filename": in_file},
            # Залишаємо тільки будівлі (Class=6)
            {"type": "filters.range", "limits": "Classification[6:6]"},
            # Кластеризація
            {"type": "filters.cluster", "min_points": min_points, "tolerance": tolerance},
            # Запис результату
            {"type": "writers.las", "filename": out_file}
        ]
    }

    pipeline = pdal.Pipeline(json.dumps(pipeline_def))
    count = pipeline.execute()
    print(f"[{base}] cluster points: {count} → {out_file}")


def main() -> None:
    pid = os.getpid()
    parser = argparse.ArgumentParser(
        description="Кластеризація точок Class=6 у кожному тайлі"
    )
    parser.add_argument(
        "classification_dir",
        help="Папка з класифікованими файлами (outputs/<basename>/classification)")
    parser.add_argument("--workers", type=int, default=4, help="Кількість процесів (default:4)")
    parser.add_argument("--min-points", type=int, default=100, help="Мінімальна кількість точок у кластері")
    parser.add_argument("--tolerance", type=float, default=2.0, help="Допуск DBSCAN (м) для кластеризації")

    args = parser.parse_args()

    def shutdown(signum, frame):
        print("STOP ALL PROCESSES", file=sys.stderr)
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            print("child", child)
            child.kill()

        parent.kill()
        sys.exit(1)

    for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, shutdown)

    labeled_dir = args.classification_dir
    project_dir = os.path.dirname(os.path.dirname(labeled_dir))
    out_dir = os.path.join(project_dir, "clusters")
    os.makedirs(out_dir, exist_ok=True)

    labeled_files = sorted(glob.glob(os.path.join(labeled_dir, "*_labeled.laz")))

    print(f'Start worker for CLUSTER with PID {pid}')

    if not labeled_files:
        print(f"Не знайдено файлів *_labeled.laz у {labeled_dir}")
        sys.exit(1)
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for in_file in labeled_files:
                pool.submit(cluster_tile, in_file, out_dir, args.min_points, args.tolerance)
    except Exception as e:
        print(f'SOME error {e}')

    print(f"Кластеризація завершена. Результати в {out_dir}")


if __name__ == "__main__":
    main()
