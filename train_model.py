import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# 1. Завантажуємо розмічені дані
# data = pd.read_csv('ml_files/features_1_tail_0_3_1p_with_label.csv')
data = pd.read_csv('ml_files/features_1_tail_0_3_1p_for_train.csv')
X = data[['HAG', 'Intensity']].values
y = data['label'].values

# 2. Розбиваємо на train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Навчаємо модель
clf = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'
)
clf.fit(X_train, y_train)

# 4. Оцінка на тесті
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['non‑tree','tree']))

# 5. Збереження моделі
joblib.dump(clf, 'ml_files/features_1_tail_0_3_1p_trees_classifier_attr.joblib')
print("Модель збережена як tree_classifier_b.joblib")
