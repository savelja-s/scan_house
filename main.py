import os
from pdal_pipeline import build_pipeline

# Налаштування
INPUT_FILE      = "lidar_files/1.las"
OUTPUT_FILE     = "output/geojson/buildings.geojson"
TEMP_FILE       = OUTPUT_FILE + ".ndgeojson"

CAPACITY        = 200_000     # ~макс. точок у чіпі
HEIGHT_THRESH   = 2.0         # мін. висота, м
CLUSTER_TOL     = 1.5         # м
MIN_PTS         = 100         # мін. точок у кластері

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    # 1) Запускаємо PDAL, який пише фічі в тимчасовий ND-GeoJSON
    pipeline = build_pipeline(
        filepath=INPUT_FILE,
        output_geojson=TEMP_FILE,
        capacity=CAPACITY,
        height_thresh=HEIGHT_THRESH,
        cluster_tol=CLUSTER_TOL,
        min_pts=MIN_PTS
    )
    pipeline.execute()
    # 2) Обгортаємо в повний FeatureCollection
    with open(TEMP_FILE, "r") as src, open(OUTPUT_FILE, "w") as dst:
        dst.write('{"type":"FeatureCollection","features":[\n')
        first = True
        for line in src:
            feat = line.strip()
            if not feat:
                continue
            if not first:
                dst.write(",\n")
            dst.write(feat)
            first = False
        dst.write("\n]}")
    os.remove(TEMP_FILE)
    print(f"✅ Готово: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
