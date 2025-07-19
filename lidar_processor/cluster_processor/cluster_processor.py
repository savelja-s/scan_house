from typing import List
import numpy as np
from lidar_processor.cluster_processor import Building
from lidar_processor.cluster_processor.boundary_builder import BoundaryBuilder
from lidar_processor.cluster_processor.height_calculator import HeightCalculator


class ClusterProcessor:
    def __init__(self, boundary_builder: BoundaryBuilder, height_calculator: HeightCalculator):
        self.boundary_builder = boundary_builder
        self.height_calculator = height_calculator

    def process(self, points: np.ndarray) -> List[Building]:
        buildings = []
        cluster_ids = np.unique(points['ClusterID'])
        for cluster_id in cluster_ids:
            cluster_points = points[points['ClusterID'] == cluster_id]
            if len(cluster_points) < 3:
                continue
            xy_points = np.vstack((cluster_points['X'], cluster_points['Y'], cluster_points['Z'])).T
            polygon = self.boundary_builder.build(xy_points)
            height = self.height_calculator.compute(xy_points)
            buildings.append(Building(polygon=polygon, height=height, cluster_id=int(cluster_id)))
        return buildings
