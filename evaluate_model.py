
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import classification_report, accuracy_score

print("Loading dataset and model...")

df = pd.read_csv("data/cleaned_dataset.csv")

features = joblib.load(
    "backend/trained_model/candidate_features.pkl"
)

model = joblib.load(
    "backend/trained_model/candidate_model.pkl"
)

X = df[features].to_numpy(dtype=np.uint8)
y = df["diseases"].astype(str).to_numpy()

with np.load("data/model_split.npz") as split:
    test_idx = split["test_idx"]

X_test = X[test_idx]
y_test = y[test_idx]

print("Evaluating model...")

predictions = model.predict(X_test)

report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0
)

rows = []

for disease, metrics in report.items():

    if disease in [
        "accuracy",
        "macro avg",
        "weighted avg"
    ]:
        continue

    rows.append({
        "Disease": disease,
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1-score"],
        "Test Records": int(metrics["support"])
    })

results = pd.DataFrame(rows)

results.to_csv(
    "data/disease_performance.csv",
    index=False
)

print("\n===== MODEL EVALUATION =====")

print(
    "Accuracy:",
    round(accuracy_score(y_test, predictions) * 100, 2),
    "%"
)

print("\n===== TOP 10 DISEASES =====")

print(
    results.sort_values(
        "F1",
        ascending=False
    ).head(10).to_string(index=False)
)

print("\n===== BOTTOM 10 DISEASES =====")

print(
    results.sort_values(
        "F1",
        ascending=True
    ).head(10).to_string(index=False)
)

print("\nDiseases with zero recall:",
      int((results["Recall"] == 0).sum()))

print("\nFull report saved to:")
print("data/disease_performance.csv")
