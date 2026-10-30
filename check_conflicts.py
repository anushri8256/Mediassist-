
import pandas as pd

DATASET = "data/Final_Augmented_dataset_Diseases_and_Symptoms.csv"

print("Loading dataset...")
df = pd.read_csv(DATASET)

target = "diseases"
features = [col for col in df.columns if col != target]

print("\n===== QUALITY REPORT =====")

print("Total records:", len(df))
print("Exact duplicate rows:", df.duplicated().sum())

# Count unique symptom combinations
patterns = df.groupby(
    features,
    dropna=False,
    sort=False
)[target].nunique()

conflicting = patterns[patterns > 1]

print("Unique symptom combinations:", len(patterns))
print(
    "Symptom combinations linked to multiple diseases:",
    len(conflicting)
)

print("\n===== CLASS DISTRIBUTION =====")

counts = df[target].value_counts()

print("Diseases with fewer than 5 records:", (counts < 5).sum())
print("Diseases with fewer than 10 records:", (counts < 10).sum())
print("Diseases with 100+ records:", (counts >= 100).sum())

print("\n===== DUPLICATE-FREE DATASET =====")

clean = df.drop_duplicates()

print("Original records:", len(df))
print("After removing exact duplicates:", len(clean))

print("\nAnalysis completed.")
