import gc
import numpy as np
import pcl
from shapely.geometry import Polygon, mapping

def cluster_and_extract(pts: np.ndarray, min_area: float):
    """
    Євклідова кластеризація через python-pcl + побудова полігонів.
    Повертає список GeoJSON features.
    """
    if pts.shape[0] < 100:
        return []

    cloud = pcl.PointCloud(pts)
    tree = cloud.make_kdtree()
    ec = cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(1.5)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(50000)
    ec.set_SearchMethod(tree)
    clusters = ec.Extract()

    features = []
    for idxs in clusters:
        coords = pts[idxs]
        hull = Polygon(coords[:, :2]).convex_hull
        if not hull.is_valid or hull.area < min_area:
            continue
        features.append({
            "type": "Feature",
            "properties": {
                "area_m2": round(hull.area, 2),
                "points_count": int(len(idxs))
            },
            "geometry": mapping(hull)
        })

    # звільняємо пам’ять
    del cloud, tree, ec, clusters
    gc.collect()
    return features
