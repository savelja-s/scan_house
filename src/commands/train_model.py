import os
import sys
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib


def normalize_name(name):
    return name.strip().lower()


def get_default_summary_path(input_path):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"{base_name}__{ts}.txt")


def train_model(args):
    # 1. Зчитати дані
    df = pd.read_csv(args.input)
    colnames = [normalize_name(c) for c in df.columns]

    # 2. Знайти ключові поля (кординати та label)
    coord_names = ["x", "y", "z"]
    label_name = "label"
    coord_fields = []
    for cname in coord_names:
        found = [col for col in df.columns if normalize_name(col) == cname]
        if not found:
            print(f"❌ Не знайдено поле координати {cname.upper()} у CSV!", file=sys.stderr)
            sys.exit(1)
        coord_fields.append(found[0])
    label_fields = [col for col in df.columns if normalize_name(col) == label_name]
    if not label_fields:
        print("❌ Не знайдено поле label у CSV!", file=sys.stderr)
        sys.exit(1)
    label_field = label_fields[0]

    # 3. Список фіч: або всі числові, або тільки задані через --features
    exclude = set(coord_fields + [label_field])
    if args.features:
        # Дозволяємо як комою, так і декілька --features
        raw_features = []
        for f in args.features:
            raw_features += [s.strip() for s in f.split(',')]
        feature_fields = []
        for f in raw_features:
            found = [col for col in df.columns if normalize_name(col) == normalize_name(f)]
            if not found:
                print(f"❌ Поле '{f}' не знайдено у CSV!", file=sys.stderr)
                sys.exit(1)
            feature_fields.append(found[0])
    else:
        feature_fields = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not feature_fields:
        print("❌ Не знайдено жодної числової фічі у CSV!", file=sys.stderr)
        sys.exit(1)

    print(f"\nВикористані ознаки для навчання: {', '.join(feature_fields)}")

    # 4. Формування X, y
    X = df[feature_fields]  # <-- залишаємо DataFrame
    y = df[label_field]

    # 5. Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Навчання моделі (RandomForest)
    clf = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)

    # 7. Оцінка
    y_pred = clf.predict(X_test)
    class_report = classification_report(y_test, y_pred, digits=3)
    conf_mat = confusion_matrix(y_test, y_pred)

    print("\n--- Класифікаційний звіт (test) ---")
    print(class_report)

    print("\nМатриця змішування:")
    print(conf_mat)

    importances = clf.feature_importances_
    feat_imp_str = "\nВажливість ознак:\n" + "\n".join(
        f"{name}: {imp:.3f}" for name, imp in sorted(zip(feature_fields, importances), key=lambda x: -x[1])
    )
    print(feat_imp_str)

    print(f"\nЗбереження моделі у: {args.output}")
    joblib.dump(clf, args.output)

    # 8. Summary log, якщо вказано або якщо не вказано, то auto-path у logs/
    summary_path = args.summary or get_default_summary_path(args.input)
    with open(summary_path, "a") as f:
        f.write(f"\n\n==== Тренування {datetime.now()} ====\n")
        f.write(f"Вхідний файл: {args.input}\n")
        f.write(f"Ознаки: {', '.join(feature_fields)}\n")
        f.write(f"Вихідна модель: {args.output}\n\n")
        f.write("--- Класифікаційний звіт (test) ---\n")
        f.write(class_report + "\n\n")
        f.write("Матриця змішування:\n")
        f.write(str(conf_mat) + "\n")
        f.write(feat_imp_str + "\n")
        f.write("=" * 40 + "\n")
    print(f"\n📝 Лог збережено у {summary_path}")
