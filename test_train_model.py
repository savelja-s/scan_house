import pdal
import json
import pandas as pd
import numpy as np
import joblib
import laspy

# --- Налаштування ---
INPUT_LAS = "lidar_files/1.las"
MODEL_PATH = "ml_files/features_1_tail_0_3_1p_trees_classifier_attr.joblib"
CLASSIFIED_LAS = "outputs/1/test/1_classified.laz"
NON_TREE_LAS = "outputs/1/test/1_non_tree.laz"

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

# --- 3. Завантаження моделі та класифікація ---
clf = joblib.load(MODEL_PATH)
df['label'] = clf.predict(df[['HAG', 'Intensity']].values).astype(np.uint8)

# --- 4. Запис результату у LAS ---
las = laspy.read(INPUT_LAS)
if len(las) != len(df):
    raise RuntimeError(f"Point count mismatch: LAS({len(las)}), features({len(df)})")
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
