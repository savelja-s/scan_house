import os
import glob
import time
import json
import joblib
import pandas as pd
import pdal
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import laspy


def human_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f} мс"
    elif seconds < 60:
        return f"{seconds:.2f} сек"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins} хв {secs:.1f} сек"


def find_feature_column(target, columns):
    target_norm = target.strip().lower()
    # Точний збіг
    for c in columns:
        if c.strip().lower() == target_norm:
            return c
    # Спец. для HAG
    if target_norm in {"hag", "hag_nn", "heightaboveground"}:
        for c in columns:
            normc = c.strip().lower()
            if normc.startswith("hag") or "heightaboveground" in normc:
                return c
    # Частковий збіг
    for c in columns:
        if target_norm in c.strip().lower():
            return c
    raise Exception(f"Поле '{target}' відсутнє у даних (маємо {list(columns)})")


def predict_for_file(model_path, las_path, pipeline_json, out_dir, feature_names):
    # 1. Запускаємо pipeline (витягуємо фічі з LAS)
    pj = []
    for step in pipeline_json:
        step_copy = step.copy()
        if step_copy.get("type") == "readers.las":
            step_copy["filename"] = las_path
        pj.append(step_copy)
    pipeline = pdal.Pipeline(json.dumps(pj))
    pipeline.execute()
    arr = pipeline.arrays[0]
    df = pd.DataFrame(arr)

    # 2. Маппінг імен, переіменування (як у попередньому рішенні)
    col_map = {}
    for f in feature_names:
        found_col = find_feature_column(f, df.columns)
        col_map[found_col] = f
    df = df.rename(columns=col_map)

    # 3. Класифікація
    clf = joblib.load(model_path)
    df['label'] = clf.predict(df[feature_names])

    # 4. Додаємо label у LAS та зберігаємо класифікований LAS
    las = laspy.read(las_path)
    if len(las) != len(df):
        raise RuntimeError(f"Point count mismatch: LAS({len(las)}), features({len(df)}) for file {las_path}")
    # Запис у classification (згідно LAS-стандарту — uint8)
    las.classification = df['label'].to_numpy(np.uint8)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(las_path))[0]
    classified_las_path = os.path.join(out_dir, f"{base_name}__{ts}_classified.las")
    las.write(classified_las_path)
    print(f"Класифікований файл збережено: {classified_las_path}")

    # 5. Додатково: зберегти LAS лише для НЕ дерев (label == 0)
    # 5. Додатково: зберегти LAS лише для НЕ дерев (label == 0)
    non_tree_mask = df['label'] == 0
    n_non_tree = non_tree_mask.sum()
    if n_non_tree > 0:
        las_out = laspy.create(point_format=las.point_format, file_version=las.header.version)
        las_out.points = las.points[non_tree_mask.to_numpy()]
        non_tree_las_path = os.path.join(out_dir, f"{base_name}__{ts}_no_trees.las")
        las_out.write(non_tree_las_path)
        print(f"LAS лише без дерев збережено: {non_tree_las_path}")
    else:
        print("У цьому файлі всі точки класифіковані як дерева. LAS без дерев не створюється.")

    # Додатково (за потреби): можна повертати ці шляхи для логування/звіту
    return las_path, classified_las_path


def predict_trees(args):
    # 1. Завантажити модель та фічі
    clf = joblib.load(args.model)
    if hasattr(clf, "feature_names_in_"):
        feature_names = list(clf.feature_names_in_)
    elif hasattr(args, "features") and args.features:
        feature_names = args.features
    else:
        raise Exception("Модель не містить feature_names_in_. Додайте --features або перевчіть модель.")

    # 2. Pipeline
    if args.pipeline:
        with open(args.pipeline, "r") as f:
            pipeline_json = json.load(f)
    else:
        with open("pipelines/last_pipeline.json", "r") as f:
            pipeline_json = json.load(f)

    # 3. Збір списку файлів
    if os.path.isfile(args.input):
        files = [args.input]
    elif os.path.isdir(args.input):
        files = glob.glob(os.path.join(args.input, "*.las")) + glob.glob(os.path.join(args.input, "*.laz"))
    else:
        print(f"❌ Не знайдено файл/папку: {args.input}")
        return

    os.makedirs(args.output, exist_ok=True)

    start = time.time()
    if len(files) == 1 or args.threads == 1:
        results = []
        for f in files:
            try:
                res = predict_for_file(args.model, f, pipeline_json, args.output, feature_names)
                print(f"Файл {res[0]}: знайдено {res[2]} дерев -> {res[1]}")
                results.append(res)
            except Exception as e:
                print(f"❌ Помилка у файлі {f}: {e}")
    else:
        with ProcessPoolExecutor(max_workers=args.threads) as executor:
            futures = [
                executor.submit(predict_for_file, args.model, f, pipeline_json, args.output, feature_names)
                for f in files
            ]
            for fut in futures:
                try:
                    res = fut.result()
                    print(f"Файл {res[0]}: знайдено {res[2]} дерев -> {res[1]}")
                except Exception as e:
                    print(f"❌ Помилка: {e}")

    print(f"\n⏱ Загальний час: {human_time(time.time() - start)}")
