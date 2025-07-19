import json
import numpy as np
import pdal


class ClusterExtractor:
    def __init__(self, tolerance=2.0, min_points=50):
        self.tolerance = tolerance
        self.min_points = min_points

    def apply(self, input_file: str) -> np.ndarray:
        pipeline_json = json.dumps([
            {"type": "readers.las", "filename": input_file},
            {"type": "filters.smrf"},
            {"type": "filters.range", "limits": "Classification![2:2]"},
            {
                "type": "filters.cluster",
                "tolerance": self.tolerance,
                "min_points": self.min_points,
                "is3d": True
            }
        ])
        pipeline = pdal.Pipeline(pipeline_json)
        pipeline.execute()
        return pipeline.arrays[0]
