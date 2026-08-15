import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)


VIDEO_FILE = "video_telemetry.csv"
SYSTEM_FILE = "system_events.csv"

MODEL_FILE = "model.pkl"

FEATURES = [
    "eye_gaze_angle",
    "audio_db",
    "tab_switches",
    "gaze_rolling_10s",
    "audio_rolling_10s"
]

TARGET = "is_cheating"

RANDOM_STATE = 42


print("=" * 70)
print("EDUGUARD AI - MODEL TRAINING")
print("=" * 70)


# ------------------------------------------------------------
# 1. CHECK DATASET FILES
# ------------------------------------------------------------

if not os.path.exists(VIDEO_FILE):
    raise FileNotFoundError(
        f"{VIDEO_FILE} was not found."
    )

if not os.path.exists(SYSTEM_FILE):
    raise FileNotFoundError(
        f"{SYSTEM_FILE} was not found."
    )


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

print("\nLoading datasets...")

video_df = pd.read_csv(VIDEO_FILE)
system_df = pd.read_csv(SYSTEM_FILE)

print(
    f"Video telemetry: {video_df.shape}"
)

print(
    f"System events: {system_df.shape}"
)


# ------------------------------------------------------------
# 3. CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

required_video = [
    "timestamp",
    "eye_gaze_angle",
    "audio_db"
]

required_system = [
    "timestamp",
    "tab_switches",
    "is_cheating"
]


for column in required_video:

    if column not in video_df.columns:

        raise ValueError(
            f"Missing column in video telemetry: {column}"
        )


for column in required_system:

    if column not in system_df.columns:

        raise ValueError(
            f"Missing column in system events: {column}"
        )


print("\nRequired columns verified.")


# ------------------------------------------------------------
# 4. CONVERT TIMESTAMPS
# ------------------------------------------------------------

print("\nProcessing timestamps...")

video_df["timestamp"] = pd.to_datetime(
    video_df["timestamp"],
    errors="coerce"
)

system_df["timestamp"] = pd.to_datetime(
    system_df["timestamp"],
    errors="coerce"
)


video_df = video_df.dropna(
    subset=["timestamp"]
).copy()

system_df = system_df.dropna(
    subset=["timestamp"]
).copy()


video_df = video_df.sort_values(
    "timestamp"
).reset_index(drop=True)

system_df = system_df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ------------------------------------------------------------
# 5. ASYNCHRONOUS MERGE
# ------------------------------------------------------------

print("\nMerging asynchronous telemetry...")

merged_df = pd.merge_asof(
    video_df,
    system_df,
    on="timestamp",
    direction="backward"
)


print(
    f"Merged dataset: {merged_df.shape}"
)


# ------------------------------------------------------------
# 6. HANDLE MISSING VALUES
# ------------------------------------------------------------

print("\nHandling missing values...")

merged_df["eye_gaze_angle"] = (
    merged_df["eye_gaze_angle"]
    .ffill()
    .bfill()
)


merged_df["audio_db"] = (
    merged_df["audio_db"]
    .interpolate(
        method="linear"
    )
    .ffill()
    .bfill()
)


merged_df["tab_switches"] = (
    merged_df["tab_switches"]
    .fillna(0)
)


merged_df["is_cheating"] = (
    merged_df["is_cheating"]
    .fillna(0)
)


merged_df["tab_switches"] = (
    merged_df["tab_switches"]
    .astype(int)
)


merged_df["is_cheating"] = (
    merged_df["is_cheating"]
    .astype(int)
)


# ------------------------------------------------------------
# 7. CREATE 10-SECOND FEATURES
# ------------------------------------------------------------

print("\nCreating 10-second rolling features...")


merged_df["gaze_rolling_10s"] = (
    merged_df["eye_gaze_angle"]
    .rolling(
        window=10,
        min_periods=1
    )
    .mean()
)


merged_df["audio_rolling_10s"] = (
    merged_df["audio_db"]
    .rolling(
        window=10,
        min_periods=1
    )
    .max()
)


# ------------------------------------------------------------
# 8. FINAL CLEANING
# ------------------------------------------------------------

merged_df = merged_df.dropna(
    subset=FEATURES + [TARGET]
).reset_index(
    drop=True
)


print(
    f"\nFinal dataset: {merged_df.shape}"
)


# ------------------------------------------------------------
# 9. SHOW CLASS DISTRIBUTION
# ------------------------------------------------------------

print("\nClass distribution:")

print(
    merged_df[TARGET]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# 10. CREATE X AND Y
# ------------------------------------------------------------

X = merged_df[FEATURES]

y = merged_df[TARGET].astype(int)


print("\nFeatures:")

for feature in FEATURES:

    print(
        f"  - {feature}"
    )


# ------------------------------------------------------------
# 11. TRAIN TEST SPLIT
# ------------------------------------------------------------

print("\nCreating train/test split...")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


print(
    f"Training records: {len(X_train)}"
)

print(
    f"Testing records: {len(X_test)}"
)


# ------------------------------------------------------------
# 12. CREATE RANDOM FOREST
# ------------------------------------------------------------

print("\nTraining Random Forest...")


model = RandomForestClassifier(
    n_estimators=50,
    max_depth=12,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)


# ------------------------------------------------------------
# 13. TRAIN MODEL
# ------------------------------------------------------------

model.fit(
    X_train,
    y_train
)


print(
    "Model training completed."
)


# ------------------------------------------------------------
# 14. TEST MODEL
# ------------------------------------------------------------

print("\nEvaluating model...")


predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    predictions
)


precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)


recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)


print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ------------------------------------------------------------
# 15. FEATURE IMPORTANCE
# ------------------------------------------------------------

print("\nFeature importance:")


importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(
    ascending=False
)


print(
    importance
)


# ------------------------------------------------------------
# 16. SAVE MODEL
# ------------------------------------------------------------

print("\nSaving model...")


model_package = {
    "model": model,
    "features": FEATURES,
    "threshold": 0.90
}


joblib.dump(
    model_package,
    MODEL_FILE,
    compress=3
)


print(
    f"\nModel saved successfully as: {MODEL_FILE}"
)


print("\n")
print("=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)