
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATASET = Path("data/cleaned_dataset.csv")
OUTPUT = Path("data/model_split.npz")

print("Loading cleaned dataset...")

df = pd.read_csv(DATASET)

X = df.drop(columns=["diseases"]).astype("uint8")
y = df["diseases"].astype(str)

# Identical symptom combinations receive identical group IDs.
groups = pd.util.hash_pandas_object(
    X,
    index=False
).to_numpy()

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

# Verify there is no shared symptom pattern.
train_groups = set(groups[train_idx])
test_groups = set(groups[test_idx])

assert train_groups.isdisjoint(test_groups)

# Check which disease labels occur only in the test set.
train_labels = set(y.iloc[train_idx])
test_labels = set(y.iloc[test_idx])

unseen_labels = test_labels - train_labels

np.savez_compressed(
    OUTPUT,
    train_idx=train_idx,
    test_idx=test_idx
)

print("\n===== SPLIT REPORT =====")
print("Training records:", len(train_idx))
print("Testing records:", len(test_idx))
print("Shared symptom patterns:", len(train_groups & test_groups))
print("Disease labels absent from training:", len(unseen_labels))
print("Saved split:", OUTPUT)

print("\nSplit created successfully.")
