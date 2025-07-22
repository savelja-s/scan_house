import os
import glob
import pdal
import json
import pandas as pd
import numpy as np
import joblib
import laspy

# --- Налаштування ---
INPUT_DIR = "outputs/1"
MODEL_PATH = "ml_files/features_1_tail_0_3_1p_trees_classifier_attr.joblib"
OUTPUT_CLASSIFIED_DIR = "outputs/1/test_classified"
OUTPUT_NON_TREE_DIR = "outputs/1/test_non_tree"

os.makedirs(OUTPUT_CLASSIFIED_DIR, exist_ok=True)
os.makedirs(OUTPUT_NON_TREE_DIR, exist_ok=True)

# --- Завантаження моделі ---
clf = joblib.load(MODEL_PATH)

# --- Пошук усіх .laz у директорії ---
laz_files = glob.glob(os.path.join(INPUT_DIR, "*.laz"))


if not laz_files:
    print("Файли з розширенням .laz не знайдено.")
    exit(1)

for INPUT_LAS in laz_files:
    file_name = os.path.splitext(os.path.basename(INPUT_LAS))[0]
    CLASSIFIED_LAS = os.path.join(OUTPUT_CLASSIFIED_DIR, f"{file_name}_classified.laz")
    NON_TREE_LAS = os.path.join(OUTPUT_NON_TREE_DIR, f"{file_name}_non_tree.laz")

    print(f"\n=== Обробка файлу: {INPUT_LAS} ===")
    # --- 1. PDAL pipeline: SMRF + HAG_NN ---
    pipeline = pdal.Pipeline(json.dumps([
        {"type": "readers.las", "filename": INPUT_LAS},
        {"type": "filters.smrf", "scalar": 1.25, "slope": 0.15, "threshold": 0.5, "window": 16.0},
        {"type": "filters.hag_nn"}
    ]))
    pipeline.execute()
    arr = pipeline.arrays[0]

    # --- 2. DataFrame для моделі ---
    df = pd.DataFrame({
        'X': arr['X'],
        'Y': arr['Y'],
        'Z': arr['Z'],
        'HAG': arr['HeightAboveGround'],
        'Intensity': arr['Intensity'],
    })

    # --- 3. Класифікація ---
    df['label'] = clf.predict(df[['HAG', 'Intensity']].values).astype(np.uint8)

    # --- 4. Запис результату у LAS ---
    las = laspy.read(INPUT_LAS)
    if len(las) != len(df):
        raise RuntimeError(f"Point count mismatch: LAS({len(las)}), features({len(df)}) for file {INPUT_LAS}")
    las.classification = df['label'].to_numpy(np.uint8)
    las.write(CLASSIFIED_LAS)
    print(f"Класифікований файл збережено: {CLASSIFIED_LAS}")

    # --- 5. Фільтрація та запис LAS тільки для НЕ дерев (label==0) ---
    non_tree_mask = df['label'] == 0
    if non_tree_mask.any():
        las_out = laspy.create(point_format=las.point_format, file_version=las.header.version)
        las_out.points = las.points[non_tree_mask.to_numpy()]
        las_out.write(NON_TREE_LAS)
        print(f"LAS лише без дерев збережено: {NON_TREE_LAS}")
    else:
        print("У цьому файлі всі точки класифіковані як дерева.")

print("\nПакетна обробка завершена.")
