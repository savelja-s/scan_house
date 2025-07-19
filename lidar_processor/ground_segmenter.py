import json
import numpy as np
import pdal


class GroundSegmenter:
    def __init__(self, scalar=1.2, slope=0.2, threshold=0.45, window=16.0):
        self.params = {
            "type": "filters.smrf",
            "scalar": scalar,
            "slope": slope,
            "threshold": threshold,
            "window": window
        }

    def apply(self, input_file: str) -> np.ndarray:
        pipeline_json = json.dumps([
            {"type": "readers.las", "filename": input_file},
            self.params
        ])
        pipeline = pdal.Pipeline(pipeline_json)
        pipeline.execute()
        return pipeline.arrays[0]
