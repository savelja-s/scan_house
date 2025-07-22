import json
import pandas as pd
import pdal

file_las_path = 'outputs/1/tile_0_3.laz'


# file_las_path = 'lidar_files/1.las'


def main():
    # 1. PDAL‑pipeline: читаємо LAS, SMRF, HAG_NN і decimation(step=200)
    pipeline = pdal.Pipeline(json.dumps([
        {"type": "readers.las", "filename": file_las_path},
        {"type": "filters.smrf", "scalar": 1.25, "slope": 0.15, "threshold": 0.5, "window": 16.0},
        {"type": "filters.hag_nn"},
        {"type": "filters.decimation", "step": 200}  # приблизно 1/200 ≈ 0.5%
    ]))
    pipeline.execute()
    arr = pipeline.arrays[0]

    print(arr.dtype.names)
    # 2. DataFrame із базовими ознаками
    df = pd.DataFrame({
        'X': arr['X'],
        'Y': arr['Y'],
        'Z': arr['Z'],
        'HAG': arr['HeightAboveGround'],
        'Intensity': arr['Intensity'],
        'Label': '',
        'Classification': arr['Classification'],
        'PointSourceId': arr['PointSourceId']
    })
    df.to_csv('ml_files/features_1_tail_0_3_c.csv', index=False)
    print(f"Збережено features.csv з {len(df)} точок.")


if __name__ == "__main__":
    main()
