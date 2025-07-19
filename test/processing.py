import pdal
import json
import numpy as np
from shapely.geometry import Polygon, mapping
import pcl


def process_tile_worker(tile_bounds, filepath, min_building_area):
    minx, miny, maxx, maxy = tile_bounds

    # --- PDAL: класифікація, фільтрація ---
    pipeline_json = [
        {"type": "readers.las", "filename": filepath},
        {"type": "filters.crop", "bounds": f"([{minx},{maxx}],[{miny},{maxy}])"},
        {"type": "filters.smrf", "returns": "last,first"},
        {"type": "filters.range", "limits": "Classification![2:2]"}
    ]

    try:
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        if pipeline.execute() < 100:
            return None
        non_ground_points = pipeline.arrays[0]
    except RuntimeError as e:
        print(f"PDAL error {tile_bounds}: {e}")
        return None

    cloud_points = np.vstack([non_ground_points['X'], non_ground_points['Y'], non_ground_points['Z']]).transpose()
    if cloud_points.shape[0] < 50:
        return None

    cloud = pcl.PointCloud(cloud_points.astype(np.float32))

    # --- PCL: сегментація площин ---
    seg = cloud.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.5)
    indices, _ = seg.segment()
    if len(indices) < 50:
        return None

    building_cloud = cloud.extract(indices, negative=False)

    # --- PCL: кластеризація ---
    tree = building_cloud.make_kdtree()
    ec = building_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(2.5)
    ec.set_MinClusterSize(50)
    ec.set_MaxClusterSize(25000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.extract()
    if not cluster_indices:
        return None

    features = []
    for indices in cluster_indices:
        points = [building_cloud[indice][:3] for indice in indices]
        points_2d = np.array(points)[:, :2]
        try:
            hull = Polygon(points_2d).convex_hull
        except Exception:
            continue

        if not hull.is_valid or hull.area < min_building_area:
            continue

        feature = {
            "type": "Feature",
            "properties": {
                "area_m2": round(hull.area, 2),
                "points_count": len(points)
            },
            "geometry": mapping(hull)
        }
        features.append(feature)
    return features if features else None
