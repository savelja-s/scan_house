import numpy as np
import pdal
import json

class LASReader:
    def __init__(self, filename: str):
        self.filename = filename

    def load_points(self) -> np.ndarray:
        pipeline_json = json.dumps([{"type": "readers.las", "filename": self.filename}])
        pipeline = pdal.Pipeline(pipeline_json)
        pipeline.execute()
        return pipeline.arrays[0]
