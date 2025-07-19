from typing import List, Tuple
import pcl
import numpy as np

class BoundaryBuilder:
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def build(self, points: np.ndarray) -> List[Tuple[float, float]]:
        if len(points) < 3:
            return []
        cloud = pcl.PointCloud(np.array(points[:, :3], dtype=np.float32))
        chull = cloud.make_ConcaveHull()
        chull.set_Alpha(self.alpha)
        result, _ = chull.reconstruct()
        return [(pt[0], pt[1]) for pt in result]
