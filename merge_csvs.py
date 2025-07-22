import laspy
import pandas as pd
import numpy as np


def main():
    # === Вхідні файли ===
    file_all = 'ml_files/features_1_tail_0_3_1p.csv'      # Файл з усіма точками (X, Y, Z, HAG, Intensity)
    # file_all = 'ml_files/features_1_0_3_1p_add_info.csv'      # Файл з усіма точками (X, Y, Z, HAG, Intensity)
    file_trees = 'ml_files/features_1_tail_0_3_1p_trees.csv'   # Файл тільки з деревами (X, Y, Z)

    # === Зчитування даних ===
    df_all = pd.read_csv(file_all)
    df_trees = pd.read_csv(file_trees)

    # === Округлення координат для точного зіставлення ===
    precision = 6  # Кількість знаків після коми
    for axis in ['X', 'Y', 'Z']:
        df_all[axis] = df_all[axis].round(precision)
        df_trees[axis] = df_trees[axis].round(precision)

    # === Побудова множини координат дерев для швидкого пошуку ===
    tree_coords = set(tuple(row) for row in df_trees[['X', 'Y', 'Z']].values)

    # === Призначення міток ===
    df_all['label'] = df_all.apply(lambda row: 1 if (row['X'], row['Y'], row['Z']) in tree_coords else 0, axis=1)

    # === Збереження результату ===
    output_file = 'ml_files/features_1_tail_0_3_1p_for_train.csv'
    df_all.to_csv(output_file, index=False)

    print(f"Збережено {output_file} з {len(df_all)} рядками.")


if __name__ == "__main__":
    main()