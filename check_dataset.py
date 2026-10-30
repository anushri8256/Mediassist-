
import pandas as pd

DATASET = "data/Final_Augmented_dataset_Diseases_and_Symptoms.csv"

print("Loading dataset...")

df = pd.read_csv(DATASET)

print("\n===== DATASET REPORT =====")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print("Total diseases:", df["diseases"].nunique())
print("Duplicate rows:", df.duplicated().sum())
print("Missing values:", df.isnull().sum().sum())

print("\n===== TOP 10 DISEASES =====")
print(df["diseases"].value_counts().head(10))

print("\n===== LEAST REPRESENTED DISEASES =====")
print(df["diseases"].value_counts().tail(10))

print("\n===== SYMPTOM INFORMATION =====")
symptoms = df.drop(columns=["diseases"])

print("Total symptom features:", len(symptoms.columns))
print("Rows with no symptoms:", (symptoms.sum(axis=1) == 0).sum())
print("Average symptoms per row:", round(symptoms.sum(axis=1).mean(), 2))

print("\nAnalysis completed!")
