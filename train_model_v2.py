import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import numpy as np

# 1. Завантажуємо дані
# data = pd.read_csv('ml_files/features_1_tail_0_3_1p_with_label.csv')
data = pd.read_csv('ml_files/features_1_tail_0_3_1p_for_train.csv')

# 2. Очищаємо/заповнюємо NaN, якщо є
data = data.fillna(0)

# 3. Вибираємо розширені ознаки
feature_cols = ['HAG', 'Intensity', 'Eigenvalue0', 'Eigenvalue1', 'Eigenvalue2',
                'NormalX', 'NormalY', 'NormalZ', 'Red', 'Green', 'Blue']
# Фільтруємо лише наявні колонки (на випадок, якщо деякі поля відсутні)
feature_cols = [col for col in feature_cols if col in data.columns]

X = data[feature_cols].values
y = data['label'].values

# 4. Масштабуємо ознаки
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# 5. Трен/тест split зі стратифікацією
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Крос‑валідація (швидка)
clf = RandomForestClassifier(
    n_estimators=100, n_jobs=-1, random_state=42, class_weight='balanced'
)
# scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='f1')
# print(f"F1 score на cross-validation: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

# 7. Тренування
clf.fit(X_train, y_train)

# 8. Оцінка на тесті
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred, target_names=['non‑tree','tree']))
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
print('ROC-AUC:', roc_auc_score(y_test, y_proba))

# 9. Візуалізація важливості ознак
importances = clf.feature_importances_
plt.barh(feature_cols, importances)
plt.xlabel('Feature importance')
plt.title('Важливість ознак')
plt.tight_layout()
plt.show()

# 10. Збереження моделі та scaler
joblib.dump({'model': clf, 'features': feature_cols},
            'ml_files/tree_classifier_b_with_scaler_filter.joblib')
print("Модель і scaler збережені як tree_classifier_b_with_scaler.joblib")
