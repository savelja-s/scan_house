import numpy as np


class HeightCalculator:
    def compute(self, points: np.ndarray) -> float:
        z_max = np.max(points[:, 2])
        z_min = np.min(points[:, 2])
        return float(z_max - z_min)
