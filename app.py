import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("heart.csv")

print("Original shape:", df.shape)

# Remove duplicates
df = df.drop_duplicates()

print("After removing duplicates:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget distribution:\n", df["target"].value_counts())

# =========================
# FEATURES / TARGET
# =========================
X = df.drop("target", axis=1)
y = df["target"]

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# MODELS
# =========================
log_model = LogisticRegression(max_iter=2000)
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train
log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate_model(name, model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print("ROC-AUC :", round(roc_auc_score(y_test, prob), 4))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
    print("\nClassification Report:\n", classification_report(y_test, pred))

# Evaluate both
evaluate_model("Logistic Regression", log_model, X_test, y_test)
evaluate_model("Random Forest", rf_model, X_test, y_test)

# =========================
# CHOOSE BEST MODEL
# =========================
# For now, we save Random Forest
best_model = rf_model

joblib.dump(best_model, "model.pkl")
print("\nModel saved as model.pkl")