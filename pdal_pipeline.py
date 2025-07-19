import json
import pdal

def build_pipeline(
    filepath: str,
    output_geojson: str,
    capacity: int,
    height_thresh: float,
    cluster_tol: float,
    min_pts: int
):
    """
    Єдиний PDAL‑конвеєр, який:
      • читає весь LAS (streaming C++)  
      • розбиває на чіпи розміру tile_size (метри) з ~capacity точок кожен  
      • класифікує ґрунт (SMRF)  
      • обчислює висоту над ґрунтом (HAG NN)  
      • відфільтровує все нижче height_thresh  
      • робить EUCLIDEAN clustering  
      • виводить GeoJSON‑рядки через writers.text  
    Пам’ять обмежується ємністю кожного чіпа.  
    """
    spec = [
        {
            "type": "readers.las",
            "filename": filepath
        },
        {
            "type": "filters.chipper",
            "capacity": capacity
        },
        {"type": "filters.smrf"},
        {"type": "filters.hag_nn"},
        {"type": "filters.range", "limits": f"HAG[{height_thresh}:]"},
        {
            "type": "filters.cluster",
            "tolerance": cluster_tol,
            "min_points": min_pts
        },
        {
            "type": "writers.text",
            "filename": output_geojson,
            "format": "geojson",
            "order": "X,Y,Z,ClusterId"
        }
    ]
    pipeline = pdal.Pipeline(json.dumps(spec))
    return pipeline
