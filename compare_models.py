
from pathlib import Path
import gc
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score

DATASET = Path("data/cleaned_dataset.csv")
SPLIT = Path("data/model_split.npz")
OUTPUT = Path("backend/trained_model")

print("Loading dataset...")

df = pd.read_csv(DATASET)

features = df.drop(columns=["diseases"]).columns.tolist()

X = df[features].to_numpy(dtype=np.uint8)
y = df["diseases"].astype(str).to_numpy()

with np.load(SPLIT) as split:
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]

X_train = X[train_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

print("Training records:", len(y_train))
print("Testing records:", len(y_test))

models = [
    (
        "Random Forest",
        RandomForestClassifier(
            n_estimators=40,
            max_depth=18,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42,
            n_jobs=1
        )
    ),
    (
        "Extra Trees",
        ExtraTreesClassifier(
            n_estimators=40,
            max_depth=18,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42,
            n_jobs=1
        )
    )
]

best_score = -1

for name, model in models:

    print(f"\n===== Training {name} =====", flush=True)

    model.fit(X_train, y_train)

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

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    # Select by macro-F1 so smaller classes also matter.
    if macro_f1 > best_score:

        best_score = macro_f1

        OUTPUT.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            model,
            OUTPUT / "candidate_model.pkl"
        )

        joblib.dump(
            features,
            OUTPUT / "candidate_features.pkl"
        )

        print("Best candidate saved separately.")

    del model
    gc.collect()

print("\n===== COMPARISON COMPLETED =====")
print("Existing model.pkl was NOT replaced.")
