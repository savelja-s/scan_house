import laspy
import pandas as pd
import numpy as np


def main():
    las = laspy.read("ml_files/features_1_tail_0_3_1p.las")
     # Приводимо до масивів numpy, інакше DataFrame не "розпакує"
    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)
    intensity = np.array(las.intensity)

    # Створюємо DataFrame
    df = pd.DataFrame({
        'X': x,
        'Y': y,
        'Z': z
    })
    df.to_csv("ml_files/features_1_tail_0_3_1p_trees.csv", index=False)


if __name__ == "__main__":
    main()