#!/usr/bin/env python3
"""
process_tile.py

PDAL pipeline to extract extended features from a .laz tile
and export them to CSV for ML processing, with rounding.
"""

import sys
import json
import pdal
import pandas as pd
import time

def find_column(arr, options, required=True):
    for key in options:
        if key in arr.dtype.names:
            return key
    if required:
        raise ValueError(f"Не знайдено жодної колонки з варіантів: {options}")
    return None

def process_tile(input_file, output_csv):
    # 1) Визначаємо PDAL‑pipeline
    pipeline_def = [
        {"type": "readers.las",        "filename": input_file},
        {"type": "filters.smrf",       "scalar": 1.25, "slope": 0.15, "threshold": 0.5, "window": 16.0},
        {"type": "filters.hag_nn"},    
        {"type": "filters.normal",     "knn": 8},
        {"type": "filters.eigenvalues","knn": 8},      
        {"type": "filters.decimation", "step": 1}
    ]
    pipeline = pdal.Pipeline(json.dumps(pipeline_def))

    # 2) Виконуємо pipeline
    start_time = time.time()
    count = pipeline.execute()
    print(f"Pipeline executed, processed {count} points.")

    # 3) Виводимо всі поля для дебагу
    arr = pipeline.arrays[0]
    print("Fields in PDAL output:", arr.dtype.names)

    # 4) Динамічно знаходимо потрібні поля
    norm_keys = [
        find_column(arr, ['Normal_0', 'NormalX']),
        find_column(arr, ['Normal_1', 'NormalY']),
        find_column(arr, ['Normal_2', 'NormalZ'])
    ]
    eig_keys = [
        find_column(arr, ['Eigenvalue_0', 'Eigenvalue0']),
        find_column(arr, ['Eigenvalue_1', 'Eigenvalue1']),
        find_column(arr, ['Eigenvalue_2', 'Eigenvalue2'])
    ]
    # Перевірка наявності кольорових каналів (не завжди присутні)
    red_key = find_column(arr, ['Red'], required=False)
    green_key = find_column(arr, ['Green'], required=False)
    blue_key = find_column(arr, ['Blue'], required=False)

    # 5) Будуємо DataFrame з усіма потрібними полями
    data = {
        'X':                  arr['X'],
        'Y':                  arr['Y'],
        'Z':                  arr['Z'],
        'HeightAboveGround':  arr['HeightAboveGround'],
        'Intensity':          arr['Intensity'],
        'Classification':     arr['Classification'],
        'ReturnNumber':       arr['ReturnNumber'],
        'NumberOfReturns':    arr['NumberOfReturns'],
        'NormalX':            arr[norm_keys[0]],
        'NormalY':            arr[norm_keys[1]],
        'NormalZ':            arr[norm_keys[2]],
        'Eigenvalue0':        arr[eig_keys[0]],
        'Eigenvalue1':        arr[eig_keys[1]],
        'Eigenvalue2':        arr[eig_keys[2]]
    }
    if red_key:   data['Red'] = arr[red_key]
    if green_key: data['Green'] = arr[green_key]
    if blue_key:  data['Blue'] = arr[blue_key]

    df = pd.DataFrame(data)

    # 6) Округлення для зменшення розміру
    round_dict = {
        'X': 3, 'Y': 3, 'Z': 3,
        'HeightAboveGround': 2,
        'Intensity': 0,
        'NormalX': 3, 'NormalY': 3, 'NormalZ': 3,
        'Eigenvalue0': 6, 'Eigenvalue1': 6, 'Eigenvalue2': 6
    }
    # Округлення кольорів якщо є
    if red_key:   round_dict['Red'] = 0
    if green_key: round_dict['Green'] = 0
    if blue_key:  round_dict['Blue'] = 0

    df = df.round(round_dict)

    # 7) Експорт в CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved extended features (rounded) to {output_csv}")
    print(f"Total processing time: {round(time.time() - start_time, 2)} seconds")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python process_tile.py <input_tile.laz> <output_features.csv>")
        sys.exit(1)
    process_tile(sys.argv[1], sys.argv[2])
