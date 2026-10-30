
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score
)

print("Loading dataset...", flush=True)

# Load cleaned dataset
df = pd.read_csv("data/cleaned_dataset.csv")

# Load the existing fixed test split
with np.load("data/model_split.npz") as split:
    test_idx = split["test_idx"]

y_test = df.iloc[test_idx]["diseases"].astype(str).to_numpy()

# All three trained models
models = [
    (
        "Previous Extra Trees",
        "backend/trained_model/candidate_model.pkl",
        "backend/trained_model/candidate_features.pkl"
    ),
    (
        "Tuned Balanced Subsample",
        "backend/trained_model/tuned_candidate.pkl",
        "backend/trained_model/tuned_features.pkl"
    ),
    (
        "Deeper Extra Trees",
        "backend/trained_model/deeper_candidate.pkl",
        "backend/trained_model/deeper_features.pkl"
    )
]

results = []

# Evaluate each model
for name, model_path, features_path in models:

    print("\n====================================")
    print(name)
    print("====================================", flush=True)

    model = joblib.load(model_path)
    features = joblib.load(features_path)

    X_test = df.iloc[test_idx][features].to_numpy(
        dtype=np.uint8
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recalls = recall_score(
        y_test,
        predictions,
        labels=model.classes_,
        average=None,
        zero_division=0
    )

    # Only count classes actually represented in test data
    test_labels, test_counts = np.unique(
        y_test,
        return_counts=True
    )

    supported_labels = set(test_labels)

    zero_recall = sum(
        recall == 0
        for label, recall in zip(model.classes_, recalls)
        if label in supported_labels
    )

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Zero Recall Classes: {zero_recall}")

    results.append({
        "Model": name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Macro F1": round(macro_f1, 4),
        "Weighted F1": round(weighted_f1, 4),
        "Zero Recall Classes": zero_recall
    })

# Final comparison
print("\n\n========== FINAL COMPARISON ==========")

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

print("\nHighest Accuracy:")
print(
    results_df.loc[
        results_df["Accuracy (%)"].idxmax(), "Model"
    ]
)

print("\nHighest Macro F1:")
print(
    results_df.loc[
        results_df["Macro F1"].idxmax(), "Model"
    ]
)

print("\nEvaluation completed successfully!")
print("No existing model files were replaced.")
