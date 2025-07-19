import json
import numpy as np
import pdal

FILE = "lidar_files/1.las"
tolerance = 2.0
min_points = 50


def get_list():
    pipeline_json = json.dumps([
        {"type": "readers.las", "filename": FILE},
        {"type": "filters.smrf"},
        {"type": "filters.range", "limits": "Classification![2:2]"},
        {
            "type": "filters.cluster",
            "tolerance": tolerance,
            "min_points": min_points,
            "is3d": True
        }
    ])
    pipeline = pdal.Pipeline(pipeline_json)
    pipeline.execute()
    return pipeline.arrays[0]


if __name__ == "__main__":
    print('Start Process')

    arr = get_list()

    print(f'LEN {len(arr)}')
