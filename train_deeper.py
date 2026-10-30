
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier

print("Loading data...", flush=True)

df = pd.read_csv("data/cleaned_dataset.csv")
features = df.drop(columns=["diseases"]).columns.tolist()

with np.load("data/model_split.npz") as split:
    train_idx = split["train_idx"]

X_train = df.iloc[train_idx][features].to_numpy(
    dtype=np.uint8
)
y_train = df.iloc[train_idx]["diseases"].astype(str).to_numpy()

print("Training Deeper Extra Trees...", flush=True)

model = ExtraTreesClassifier(
    n_estimators=60,
    max_depth=24,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight=None,
    random_state=42,
    n_jobs=1
)

model.fit(X_train, y_train)

output = Path("backend/trained_model")
output.mkdir(parents=True, exist_ok=True)

joblib.dump(model, output / "deeper_candidate.pkl")
joblib.dump(features, output / "deeper_features.pkl")

print("\nDeeper model saved successfully!")
print("Existing working model was NOT replaced.")
