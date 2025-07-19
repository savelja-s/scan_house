import os
import geojson
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count

from utils import get_las_bounds, split_area_into_tiles
from processing import process_tile_worker

# --- КОНФІГУРАЦІЯ ---
FILE = "../lidar_files/1.las"
OUTPUT = "../output/geojson/buildings.geojson"
TILE_SIZE = 50  # м
MIN_BUILDING_AREA = 10  # м^2
WORKERS = max(1, cpu_count() - 2)


def main():
    print(f"Старт: {FILE}")

    minx, miny, maxx, maxy = get_las_bounds(FILE)
    print(f"Межі: X ({minx:.2f}, {maxx:.2f}), Y ({miny:.2f}, {maxy:.2f})")

    tiles = split_area_into_tiles(minx, miny, maxx, maxy, TILE_SIZE)
    print(f"Тайлів: {len(tiles)}")

    worker_func = partial(process_tile_worker, filepath=FILE, min_building_area=MIN_BUILDING_AREA)
    features = []

    print(f"Паралельна обробка на {WORKERS} ядрах...")
    with Pool(WORKERS) as pool:
        results = list(tqdm(pool.imap_unordered(worker_func, tiles), total=len(tiles)))

    for tile_features in results:
        if tile_features:
            features.extend(tile_features)

    if not features:
        print("Нічого не знайдено — перевірте параметри.")
        return

    feature_collection = geojson.FeatureCollection(features)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        geojson.dump(feature_collection, f, indent=2)

    print(f"OK! Знайдено {len(features)} будівель. Результат: {OUTPUT}")


if __name__ == "__main__":
    main()
