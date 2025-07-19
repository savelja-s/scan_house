from lidar_processor.lidar_processor import LiDARProcessor


if __name__ == "__main__":
    print('Start Process')
    processor = LiDARProcessor("lidar_files/1.las", "output/geojson/buildings.geojson")
    processor.run()
