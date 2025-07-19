import numpy as np


class BuildingFilter:
    def filter_non_ground(self, points: np.ndarray) -> np.ndarray:
        return points[points['Classification'] != 2]
