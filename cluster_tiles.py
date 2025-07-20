import os
import sys
import glob
import json
import pdal
import signal
import ctypes
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool


# --- 1) Ініціалізатор воркера: ігнор SIGINT + death‑sig на випадок, якщо батько помре
def init_worker():
    # ігноруємо Ctrl+C у воркерах
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # встановлюємо сигнал смерті воркера, коли помирає батько
    libc = ctypes.CDLL("libc.so.6")
    PR_SET_PDEATHSIG = 1
    libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


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


def terminate_workers(executor: ProcessPoolExecutor):
    # пряме завершення через внутрішні об’єкти
    for p in getattr(executor, "_processes", {}).values():
        try:
            p.terminate()
        except Exception:
            pass


def main() -> None:
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

    # ловимо і SIGHUP, і CTRL+C, і SIGTERM
    def shutdown(signum, frame):
        print("Отримано сигнал зупинки — зупиняємо все…", file=sys.stderr)
        if executor_ref[0]:
            terminate_workers(executor_ref[0])
        sys.exit(1)

    for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, shutdown)

    labeled_dir = args.classification_dir
    project_dir = os.path.dirname(os.path.dirname(labeled_dir))
    out_dir = os.path.join(project_dir, "clusters")
    os.makedirs(out_dir, exist_ok=True)

    labeled_files = sorted(glob.glob(os.path.join(labeled_dir, "*_labeled.laz")))

    if not labeled_files:
        print(f"Не знайдено файлів *_labeled.laz у {labeled_dir}")
        return

    executor_ref = [None]  # щоб замикання бачили executor у shutdown

    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
            executor_ref[0] = pool
            futures = [pool.submit(cluster_tile, f, out_dir, args.min_points, args.tolerance) for f in labeled_files]
            for fut in as_completed(futures):
                fut.result()  # якщо воркер гине — отримаємо BrokenProcessPool
    except BrokenProcessPool:
        print("Воркери загинули аварійно — зупиняємо все.", file=sys.stderr)
        if executor_ref[0]:
            terminate_workers(executor_ref[0])
        sys.exit(1)

    print(f"Кластеризація завершена. Результати в {out_dir}")


if __name__ == "__main__":
    main()
