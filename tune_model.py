
from pathlib import Path
import gc
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

DATASET = Path("data/cleaned_dataset.csv")
SPLIT = Path("data/model_split.npz")
OUTPUT = Path("backend/trained_model")

print("Loading dataset...", flush=True)

df = pd.read_csv(DATASET)

features = df.drop(columns=["diseases"]).columns.tolist()

X = df[features].to_numpy(dtype=np.uint8)
y = df["diseases"].astype(str).to_numpy()

with np.load(SPLIT) as split:
    train_idx = split["train_idx"]

# Final test set is NOT used during tuning.
X_train_all = X[train_idx]
y_train_all = y[train_idx]

# Keep identical symptom combinations in the same group.
groups = pd.util.hash_pandas_object(
    df.iloc[train_idx][features],
    index=False
).to_numpy()

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=123
)

fit_idx, val_idx = next(
    splitter.split(
        X_train_all,
        y_train_all,
        groups=groups
    )
)

assert set(groups[fit_idx]).isdisjoint(
    set(groups[val_idx])
)

X_fit = X_train_all[fit_idx]
y_fit = y_train_all[fit_idx]

X_val = X_train_all[val_idx]
y_val = y_train_all[val_idx]

print("Training records:", len(y_fit))
print("Validation records:", len(y_val))

settings = [
    {
        "name": "Balanced Extra Trees",
        "n_estimators": 40,
        "max_depth": 18,
        "min_samples_leaf": 3,
        "class_weight": "balanced"
    },
    {
        "name": "Deeper Extra Trees",
        "n_estimators": 60,
        "max_depth": 24,
        "min_samples_leaf": 2,
        "class_weight": None
    },
    {
        "name": "Balanced Subsample Extra Trees",
        "n_estimators": 60,
        "max_depth": 24,
        "min_samples_leaf": 2,
        "class_weight": "balanced_subsample"
    }
]

best_f1 = -1
best_setting = None

for config in settings:

    print(
        "\n===== " + config["name"] + " =====",
        flush=True
    )

    model = ExtraTreesClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        class_weight=config["class_weight"],
        max_features="sqrt",
        random_state=42,
        n_jobs=1
    )

    model.fit(X_fit, y_fit)

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )

    print(f"Validation accuracy: {accuracy * 100:.2f}%")
    print(f"Validation macro F1: {macro_f1:.4f}")

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_setting = config.copy()

    del model
    gc.collect()

print("\n===== BEST VALIDATION CONFIGURATION =====")
print(best_setting)
print("Best validation macro F1:", round(best_f1, 4))

# Refit selected configuration on the full training partition.
# The held-out final test partition remains untouched.

print("\nRefitting selected model...", flush=True)

final_model = ExtraTreesClassifier(
    n_estimators=best_setting["n_estimators"],
    max_depth=best_setting["max_depth"],
    min_samples_leaf=best_setting["min_samples_leaf"],
    class_weight=best_setting["class_weight"],
    max_features="sqrt",
    random_state=42,
    n_jobs=1
)

final_model.fit(X_train_all, y_train_all)

OUTPUT.mkdir(parents=True, exist_ok=True)

joblib.dump(
    final_model,
    OUTPUT / "tuned_candidate.pkl"
)

joblib.dump(
    features,
    OUTPUT / "tuned_features.pkl"
)

print("\nTuned candidate saved successfully.")
print("Existing model.pkl was NOT replaced.")
print("Final test set was NOT evaluated.")
