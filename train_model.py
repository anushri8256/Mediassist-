
import pandas as pd
import joblib
import gc

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =====================================
# 1. PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
)

MODEL_DIR = BASE_DIR / "backend" / "trained_model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "model.pkl"
FEATURES_PATH = MODEL_DIR / "feature_columns.pkl"

# =====================================
# 2. LOAD DATASET
# =====================================

print("Loading MediAssist dataset...", flush=True)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["diseases"]).astype("uint8")
y = df["diseases"]

print("Total records:", len(df))
print("Total symptoms:", X.shape[1])
print("Total diseases:", y.nunique())

# =====================================
# 3. TRAIN / TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

del df
gc.collect()

# =====================================
# 4. TRAIN LIGHTWEIGHT MODEL
# =====================================

model = RandomForestClassifier(
    n_estimators=20,
    max_depth=12,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=1
)

print("Training MediAssist AI model...", flush=True)

model.fit(X_train, y_train)

# =====================================
# 5. TEST MODEL
# =====================================

print("Testing model...", flush=True)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

del predictions, X_train, y_train
gc.collect()

# =====================================
# 6. SAVE TRAINED MODEL
# =====================================

print("Saving trained model...", flush=True)

# Save to temporary files first.
# Final files are replaced only after verification.

TEMP_MODEL = MODEL_DIR / "model_temp.pkl"
TEMP_FEATURES = MODEL_DIR / "features_temp.pkl"

joblib.dump(model, TEMP_MODEL, compress=3)

joblib.dump(
    X.columns.tolist(),
    TEMP_FEATURES
)

print("Model size:", TEMP_MODEL.stat().st_size, "bytes")
print("Features size:", TEMP_FEATURES.stat().st_size, "bytes")

# =====================================
# 7. VERIFY SAVED FILES
# =====================================

print("Verifying saved model...", flush=True)

loaded_model = joblib.load(TEMP_MODEL)
loaded_features = joblib.load(TEMP_FEATURES)

if len(loaded_features) != X.shape[1]:
    raise ValueError("Feature count verification failed!")

# Check whether loaded model can predict
test_prediction = loaded_model.predict(X_test.iloc[:1])

print("Test prediction:", test_prediction[0])

del loaded_model
gc.collect()

# Move verified files to final filenames
TEMP_MODEL.replace(MODEL_PATH)
TEMP_FEATURES.replace(FEATURES_PATH)

print("--------------------------------")
print("Model verification successful!")
print("MediAssist model saved successfully!")
print("Final model size:", MODEL_PATH.stat().st_size, "bytes")
print("Total saved features:", len(loaded_features))
print("Location:", MODEL_DIR)
print("--------------------------------")
