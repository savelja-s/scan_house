import json
import pdal
import numpy as np
import gc
import pcl
from shapely.geometry import Polygon, mapping


def process_tile(tile_bounds, filepath, min_building_area, height_threshold=2.0):
    """
    Process a single tile: read with bounds, classify ground, compute HAG via plugin,
    filter by height, and extract building clusters.
    """
    minx, miny, maxx, maxy = tile_bounds

    # PDAL pipeline using bounds directly in reader to limit memory use
    pipeline_steps = [
        {"type": "readers.las", "filename": filepath},
        {"type": "filters.crop", "bounds": f"([{minx},{maxx}],[{miny},{maxy}])"},
        {"type": "filters.smrf"},          # ground classification
        {"type": "filters.hag_nn"},       # height above ground (nearest-neighbor)
        # filters.hag_delaunay         Computes height above ground using delaunay interpolation of ground returns.
        # filters.hag_dem              Computes height above ground using a DEM raster.
        # filters.hag_nn               Computes height above ground using nearest-neighbor ground-classified returns.
        {"type": "filters.range", "limits": f"HAG[{height_threshold}:]"}  # keep high points
    ]

    try:
        pipeline = pdal.Pipeline(json.dumps(pipeline_steps))
        count = pipeline.execute()
        if count == 0:
            return []
        arr = pipeline.arrays[0]
        pts = np.vstack((arr['X'], arr['Y'], arr['Z'])).T.astype(np.float32)
    except Exception as e:
        print(f"[WARNING] PDAL error on tile {tile_bounds}: {e}")
        return []

    if pts.shape[0] < 100:
        return []

    # Euclidean clustering with PCL
    cloud = pcl.PointCloud(pts)
    tree = cloud.make_kdtree()
    ec = cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(1.5)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(50000)
    ec.set_SearchMethod(tree)
    # Correct clustering call for python-pcl
    cluster_indices = ec.Extract()

    features = []
    for pi in cluster_indices:
        idxs = pi.indices
        coords = pts[idxs]
        hull = Polygon(coords[:, :2]).convex_hull
        if not hull.is_valid or hull.area < min_building_area:
            continue
        features.append({
            "type": "Feature",
            "properties": {
                "area_m2": round(hull.area, 2),
                "points_count": int(len(idxs))
            },
            "geometry": mapping(hull)
        })

    del pts, arr, pipeline, cloud, tree, ec, cluster_indices
    gc.collect()
    return features 