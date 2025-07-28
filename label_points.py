import pandas as pd
import numpy as np
import laspy

# Шляхи до файлів
csv_path = "test_sklearn/training_data/training_sample.csv"
trees_las_path = "test_sklearn/training_data/training_sample_only_trees.las"
no_trees_las_path = "test_sklearn/training_data/training_sample_without_trees.las"
output_csv_path = "test_sklearn/training_data/training_sample_labeled.csv"

# 1. Завантажуємо CSV з усіма точками
df = pd.read_csv(csv_path)


# 2. Зчитуємо LAS файли (XYZ для ідентифікації)
def read_las_xyz(las_path):
    las = laspy.read(las_path)
    return np.vstack((las.x, las.y, las.z)).T


xyz_trees = read_las_xyz(trees_las_path)
xyz_no_trees = read_las_xyz(no_trees_las_path)

# 3. Перетворюємо координати на множини (округлення до 3 знаків)
trees_set = {tuple(np.round(xyz, 3)) for xyz in xyz_trees}
no_trees_set = {tuple(np.round(xyz, 3)) for xyz in xyz_no_trees}

# 4. Додаємо колонку label (-1 за замовчуванням)
df['label'] = -1

# 5. Проставляємо 1 для дерев, 0 для не-дерев
coords_rounded = np.round(df[['X', 'Y', 'Z']].values, 3)
for idx, xyz in enumerate(coords_rounded):
    tup = tuple(xyz)
    if tup in trees_set:
        df.at[idx, 'label'] = 1
    elif tup in no_trees_set:
        df.at[idx, 'label'] = 0

# 6. Зберігаємо результат
df.to_csv(output_csv_path, index=False)
print(f"Лейблований файл збережено: {output_csv_path}")
