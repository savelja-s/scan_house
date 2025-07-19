from lidar_processor.cluster_extractor import ClusterExtractor
from lidar_processor.cluster_processor.boundary_builder import BoundaryBuilder
from lidar_processor.cluster_processor.processor import ClusterProcessor
from lidar_processor.cluster_processor.height_calculator import HeightCalculator
from lidar_processor.geo_json_exporter import GeoJSONExporter


class LiDARProcessor:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file

    def run(self):
        cluster_extractor = ClusterExtractor()
        print(f'Start read file: {self.input_file} .')
        points = cluster_extractor.apply(self.input_file)
        print(f'GET {points} POINTS.')

        boundary_builder = BoundaryBuilder()
        height_calculator = HeightCalculator()
        processor = ClusterProcessor(boundary_builder, height_calculator)
        print(f'START ClusterProcessor.')
        buildings = processor.process(points)
        print(f'RESULT  ClusterProcessor  {len(buildings)}')

        exporter = GeoJSONExporter()
        exporter.export(buildings, self.output_file)
