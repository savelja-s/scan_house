import pandas as pd
import laspy
import numpy as np


def assign_labels(args):
    # 1. Зчитати LAS
    las = laspy.read(args.las)
    las_coords = set(zip(np.array(las.x).round(5), np.array(las.y).round(5), np.array(las.z).round(5)))

    # 2. Зчитати CSV
    df = pd.read_csv(args.input_csv)
    coords = list(zip(df['X'].round(5), df['Y'].round(5), df['Z'].round(5)))
    df['label'] = [1 if xyz in las_coords else 0 for xyz in coords]

    # 3. Балансування (undersampling)
    if args.balance:
        frac = float(args.balance)
        pos = df[df['label'] == 1]
        neg = df[df['label'] == 0]
        n_pos = len(pos)
        n_neg = int(n_pos * frac)
        # Якщо негативних менше ніж потрібно — беремо всі
        n_neg = min(n_neg, len(neg))
        neg_sample = neg.sample(n=n_neg, random_state=42)
        df = pd.concat([pos, neg_sample]).sample(frac=1, random_state=42)  # Перемішати

        print(f"Балансування: {len(pos)} позитивних, {len(neg_sample)} негативних")

    # 4. Записати новий CSV
    df.to_csv(args.output, index=False)
    print(f"Файл з лейблами збережено у: {args.output}")
