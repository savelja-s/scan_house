from typing import List
import geojson
from lidar_processor.cluster_processor import Building

class GeoJSONExporter:
    def export(self, buildings: List[Building], output_path: str):
        features = []
        for b in buildings:
            if not b.polygon:
                continue
            poly = geojson.Polygon([b.polygon + [b.polygon[0]]])  # Закриваємо контур
            feature = geojson.Feature(geometry=poly, properties={"height": b.height, "cluster": b.cluster_id})
            features.append(feature)
        fc = geojson.FeatureCollection(features)
        with open(output_path, "w") as f:
            geojson.dump(fc, f)
