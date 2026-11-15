
from pathlib import Path
import numpy as np
import tensorflow as tf

# ==========================================
# 1. FILE PATHS
# ==========================================

DATA_PATH = Path("outputs/ravdess_mfcc_features.npz")
NORM_PATH = Path("outputs/normalization_parameters.npz")
MODEL_PATH = Path("models/best_speech_emotion_cnn.keras")

# ==========================================
# 2. LOAD DATA
# ==========================================

data = np.load(DATA_PATH, allow_pickle=True)

X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)
groups = data["groups"].astype(np.int64)

emotion_names = data["emotion_names"].tolist()

print("\n========== DATASET DIAGNOSTICS ==========")

print("Feature shape:", X.shape)
print("Label shape:", y.shape)
print("Number of emotions:", len(emotion_names))
print("Emotion mapping:")

for i, emotion in enumerate(emotion_names):
    print(f"{i} = {emotion}")

# ==========================================
# 3. CHECK FEATURE VALUES
# ==========================================

print("\n========== FEATURE CHECK ==========")

print("NaN values:", np.isnan(X).sum())
print("Infinite values:", np.isinf(X).sum())
print("Minimum:", np.min(X))
print("Maximum:", np.max(X))
print("Mean:", np.mean(X))
print("Standard deviation:", np.std(X))

# ==========================================
# 4. CHECK CLASS DISTRIBUTION
# ==========================================

print("\n========== CLASS DISTRIBUTION ==========")

splits = {
    "Training": np.isin(groups, range(1, 19)),
    "Validation": np.isin(groups, range(19, 22)),
    "Testing": np.isin(groups, range(22, 25))
}

for split_name, mask in splits.items():

    print(f"\n{split_name}:")

    for i, emotion in enumerate(emotion_names):
        count = np.sum(y[mask] == i)
        print(f"{emotion}: {count}")

# ==========================================
# 5. CHECK MFCC FEATURES BY EMOTION
# ==========================================

print("\n========== MFCC CLASS STATISTICS ==========")

for i, emotion in enumerate(emotion_names):

    class_features = X[y == i]

    print(
        f"{emotion}: "
        f"Mean={np.mean(class_features):.4f}, "
        f"Std={np.std(class_features):.4f}"
    )

# ==========================================
# 6. LOAD NORMALIZATION PARAMETERS
# ==========================================

normalization = np.load(NORM_PATH)

mean = normalization["mean"]
std = normalization["std"]

# ==========================================
# 7. PREPARE TEST DATA
# ==========================================

test_mask = splits["Testing"]

X_test = X[test_mask]
y_test = y[test_mask]

X_test = (X_test - mean) / std
X_test = X_test[..., np.newaxis]

# ==========================================
# 8. LOAD TRAINED MODEL
# ==========================================

print("\n========== MODEL DIAGNOSTICS ==========")

model = tf.keras.models.load_model(str(MODEL_PATH))

print("Model loaded successfully!")
print("Expected input shape:", model.input_shape)

# ==========================================
# 9. PREDICT TEST EMOTIONS
# ==========================================

probabilities = model.predict(X_test, verbose=0)

predictions = np.argmax(probabilities, axis=1)

print("\n========== PREDICTION DISTRIBUTION ==========")

for i, emotion in enumerate(emotion_names):

    actual_count = np.sum(y_test == i)
    predicted_count = np.sum(predictions == i)

    print(
        f"{emotion}: "
        f"Actual={actual_count}, "
        f"Predicted={predicted_count}"
    )

# ==========================================
# 10. CHECK PREDICTION CONFIDENCE
# ==========================================

print("\n========== PREDICTION CONFIDENCE ==========")

print("Average maximum probability:",
      np.mean(np.max(probabilities, axis=1)))

print("Average probability for each class:")

for i, emotion in enumerate(emotion_names):
    print(
        f"{emotion}: "
        f"{np.mean(probabilities[:, i]):.4f}"
    )

print("\nDIAGNOSTIC COMPLETED!")
