import json, pdal
import numpy as np
import pandas as pd

# Читаємо весь набір із SMRF + HAG_NN
pipeline = pdal.Pipeline(json.dumps([
    {"type": "readers.las", "filename": "test.las"},
    {"type": "filters.smrf", "scalar": 1.25, "slope": 0.15, "threshold": 0.5, "window": 16.0},
    {"type": "filters.hag_nn"}
]))
pipeline.execute()
arr = pipeline.arrays[0]

# Визначаємо випадкові індекси (0.5% від усіх)
n_total = arr['X'].size
n_sample = int(n_total * 0.005)
idx = np.random.choice(n_total, size=n_sample, replace=False)

# Формуємо DataFrame тільки з вибірки
df = pd.DataFrame({
    'X': arr['X'][idx],
    'Y': arr['Y'][idx],
    'Z': arr['Z'][idx],
    'HAG': arr['HAG_NN'][idx],
    'Intensity': arr['Intensity'][idx],
})
df.to_csv('features.csv', index=False)
print(f"Збережено features.csv з {len(df)} точок.")
