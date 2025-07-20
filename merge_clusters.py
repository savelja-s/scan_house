import os
import glob
import argparse
import laspy
import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPoint
from shapely.ops import unary_union


def extract_polygons(clusters_dir: str, buffer_dist: float) -> gpd.GeoDataFrame:
    """
    Для кожного кластерного файлу створює полігон (convex hull + buffer).
    Повертає GeoDataFrame з колонками: geometry, source_file, cluster_id.
    """
    records = []
    pattern = os.path.join(clusters_dir, "*_cluster.laz")
    for file_path in glob.glob(pattern):
        try:
            las = laspy.read(file_path)
        except Exception as e:
            print(f"ERROR reading {file_path}: {e}")
            continue
        # отримаємо масив cluster IDs
        try:
            clids = las.point.get_dimension("ClusterID")
        except Exception:
            print(f"No 'ClusterID' dimension in {file_path}")
            continue
        # координати
        xs = las.x
        ys = las.y
        for cid in np.unique(clids):
            if cid < 0:
                continue
            mask = (clids == cid)
            pts = np.column_stack((xs[mask], ys[mask]))
            if pts.shape[0] < 3:
                # замало точок для hull
                continue
            hull = MultiPoint(pts).convex_hull
            poly = hull.buffer(buffer_dist)
            records.append({
                "geometry": poly,
                "source_file": os.path.basename(file_path),
                "cluster_id": int(cid)
            })
    if not records:
        raise ValueError(f"No polygons extracted from {clusters_dir}")
    gdf = gpd.GeoDataFrame(records)
    return gdf


def merge_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Об'єднує всі полігони, що перетинаються, у незалежні будівлі.
    Додає у вихідну GeoDataFrame колонку building_id.
    """
    # єдина мультиполігональна геометрія
    all_union = unary_union(gdf.geometry.values)
    # розділяємо на окремі полігони
    parts = []
    if all_union.geom_type == 'Polygon':
        parts = [all_union]
    else:
        parts = list(all_union)
    merged = gpd.GeoDataFrame(
        {"geometry": parts}
    )
    merged["building_id"] = merged.index + 1
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Об'єднання кластерів на межах тайлів"
    )
    parser.add_argument(
        "clusters_dir",
        help="Папка з кластерними .laz файлами (outputs/<basename>/clusters)"
    )
    parser.add_argument(
        "--buffer", type=float, default=1.0,
        help="Буфер (м) для розширення hull полігону перед злиттям"
    )
    parser.add_argument(
        "--output", help="Шлях до вихідного GeoJSON (default: buildings_merged.geojson)",
        default=None
    )
    args = parser.parse_args()

    # Витягуємо полігони з кожного кластера
    print(f"Extracting polygons from clusters in {args.clusters_dir}...")
    try:
        gdf = extract_polygons(args.clusters_dir, args.buffer)
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    print(f"Merging {len(gdf)} polygons by spatial intersection...")
    merged = merge_polygons(gdf)

    # Визначаємо шлях для збереження
    if args.output:
        out_file = args.output
    else:
        out_file = os.path.join(os.path.dirname(args.clusters_dir), "buildings_merged.geojson")

    merged.to_file(out_file, driver="GeoJSON")
    print(f"✔ Merged buildings saved to {out_file}")


if __name__ == "__main__":
    main()
