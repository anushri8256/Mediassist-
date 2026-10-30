
from pathlib import Path
import pandas as pd

SOURCE = Path(
    "data/Final_Augmented_dataset_Diseases_and_Symptoms.csv"
)

OUTPUT = Path("data/cleaned_dataset.csv")

print("Loading original dataset...")

df = pd.read_csv(SOURCE)

original_rows = len(df)

# Remove exact duplicate records
df = df.drop_duplicates().copy()

# Remove disease labels with fewer than 10 examples
counts = df["diseases"].value_counts()
valid_diseases = counts[counts >= 10].index

df = df[df["diseases"].isin(valid_diseases)].copy()

# Save a separate cleaned dataset
df.to_csv(OUTPUT, index=False)

print("\n===== CLEANING REPORT =====")
print("Original records:", original_rows)
print("Cleaned records:", len(df))
print("Remaining diseases:", df["diseases"].nunique())
print("Removed records:", original_rows - len(df))
print("Saved to:", OUTPUT)

print("\nOriginal dataset remains unchanged.")
