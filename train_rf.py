#!/usr/bin/env python3
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Train RandomForest on labeled LiDAR features (sklearn).")
    p.add_argument("--csv", required=True, help="Path to training_sample_labeled.csv")
    p.add_argument("--target", default="label", help="Target column name")
    p.add_argument("--drop-cols", nargs="*", default=["X", "Y", "Z"], help="Columns to drop from features")
    p.add_argument("--test-size", type=float, default=0.2, help="Test size for train_test_split")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--n-estimators", type=int, default=400, help="RF n_estimators (ignored if --tune)")
    p.add_argument("--max-depth", type=int, default=None, help="RF max_depth (ignored if --tune)")
    p.add_argument("--class-weight", default="balanced", choices=["balanced", "none"],
                   help="Use 'balanced' or not")
    p.add_argument("--tune", action="store_true", help="Run RandomizedSearchCV to tune hyperparameters")
    p.add_argument("--cv", type=int, default=3, help="CV folds for tuning")
    p.add_argument("--n-iters", type=int, default=30, help="Number of iterations for RandomizedSearchCV")
    p.add_argument("--out-dir", default="models_rf", help="Directory to save model & reports")
    p.add_argument("--plot-importances", action="store_true", help="Save feature importances plot")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # 1. Load & clean
    # ----------------------------
    df = pd.read_csv(args.csv)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in CSV.")

    # Відкидаємо невідомі лейбли
    before = len(df)
    df = df[df[args.target] != -1].copy()
    after = len(df)
    print(f"Dropped {(before - after)} samples with label = -1. Remaining: {after}")

    # Обираємо числові фічі
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Викидаємо таргет і колонки, які ви явно не хочете використовувати
    feature_cols = [c for c in numeric_cols if c != args.target and c not in args.drop_cols]

    X = df[feature_cols].values
    y = df[args.target].astype(int).values

    print(f"Features used ({len(feature_cols)}): {feature_cols}")

    # ----------------------------
    # 2. Train/test split
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # ----------------------------
    # 3. Model
    # ----------------------------
    if args.class_weight == "balanced":
        class_weight = "balanced"
    else:
        class_weight = None

    if args.tune:
        print("Running RandomizedSearchCV...")
        param_dist = {
            "n_estimators": [200, 300, 400, 600, 800, 1000],
            "max_depth": [None, 8, 12, 16, 24, 32],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7]
        }

        base = RandomForestClassifier(
            random_state=args.seed,
            n_jobs=-1,
            class_weight=class_weight
        )

        cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
        search = RandomizedSearchCV(
            base,
            param_distributions=param_dist,
            n_iter=args.n_iters,
            scoring="f1",
            cv=cv,
            random_state=args.seed,
            verbose=1,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print("Best params:", search.best_params_)
    else:
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.seed,
            n_jobs=-1,
            class_weight=class_weight
        )
        model.fit(X_train, y_train)

    # ----------------------------
    # 4. Evaluation
    # ----------------------------
    def eval_split(split_name, Xs, ys):
        yp = model.predict(Xs)
        proba = None
        try:
            proba = model.predict_proba(Xs)[:, 1]
        except Exception:
            pass

        acc = accuracy_score(ys, yp)
        f1 = f1_score(ys, yp)
        metrics = {
            "accuracy": acc,
            "f1": f1
        }
        if proba is not None and len(np.unique(ys)) == 2:
            try:
                auc = roc_auc_score(ys, proba)
                metrics["roc_auc"] = auc
            except Exception:
                pass

        print(f"\n[{split_name}] metrics:", metrics)
        print(classification_report(ys, yp, digits=4))

        cm = confusion_matrix(ys, yp)
        return metrics, cm

    metrics_train, cm_train = eval_split("train", X_train, y_train)
    metrics_test, cm_test = eval_split("test", X_test, y_test)

    # ----------------------------
    # 5. Save artifacts
    # ----------------------------
    model_path = out_dir / "rf_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Feature list saved to: {out_dir / 'feature_columns.json'}")

    report = {
        "train": metrics_train,
        "test": metrics_test,
        "confusion_matrix_train": cm_train.tolist(),
        "confusion_matrix_test": cm_test.tolist(),
        "params": model.get_params()
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {out_dir / 'report.json'}")

    # ----------------------------
    # 6. Feature importances (optional)
    # ----------------------------
    if args.plot_importances:
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1]
        topk = min(40, len(feature_cols))  # щоб не перевантажувати графік
        plt.figure(figsize=(10, 6))
        plt.bar(range(topk), importances[idx][:topk])
        plt.xticks(range(topk), [feature_cols[i] for i in idx[:topk]], rotation=90)
        plt.title("RandomForest feature importances (top-40)")
        plt.tight_layout()
        fig_path = out_dir / "feature_importances.png"
        plt.savefig(fig_path, dpi=200)
        print(f"Feature importances plot saved to: {fig_path}")


if __name__ == "__main__":
    main()
