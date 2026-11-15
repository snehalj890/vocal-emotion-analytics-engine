
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ==========================================
# 1. PROJECT PATHS
# ==========================================

DATA_PATH = Path("outputs/ravdess_mfcc_features.npz")
NORMALIZATION_PATH = Path("outputs/normalization_parameters.npz")
MODEL_PATH = Path("models/best_speech_emotion_cnn.keras")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_PATH = OUTPUT_DIR / "confusion_matrix.png"

# ==========================================
# 2. CHECK REQUIRED FILES
# ==========================================

for path in [DATA_PATH, NORMALIZATION_PATH, MODEL_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

# ==========================================
# 3. LOAD DATASET
# ==========================================

data = np.load(DATA_PATH, allow_pickle=True)

X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)
groups = data["groups"].astype(np.int64)

emotion_names = data["emotion_names"].tolist()

print("Dataset loaded successfully!")
print("Total recordings:", len(X))

# ==========================================
# 4. SELECT TEST ACTORS
# ==========================================

test_mask = np.isin(groups, range(22, 25))

X_test = X[test_mask]
y_test = y[test_mask]

print("Test dataset shape:", X_test.shape)
print("Test recordings:", len(y_test))

# ==========================================
# 5. LOAD NORMALIZATION PARAMETERS
# ==========================================

normalization = np.load(NORMALIZATION_PATH)

mean = normalization["mean"]
std = normalization["std"]

X_test = (X_test - mean) / std

# Add CNN channel dimension
X_test = X_test[..., np.newaxis]

# ==========================================
# 6. LOAD TRAINED CNN MODEL
# ==========================================

print("\nLoading trained CNN model...")

model = tf.keras.models.load_model(str(MODEL_PATH))

print("Model loaded successfully!")

# ==========================================
# 7. PREDICT EMOTIONS
# ==========================================

print("\nGenerating predictions...")

predictions = model.predict(X_test, verbose=1)

y_pred = np.argmax(predictions, axis=1)

# ==========================================
# 8. GENERATE CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(emotion_names))
)

print("\nConfusion Matrix:")
print(cm)

# ==========================================
# 9. DISPLAY CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_test,
    y_pred,
    labels=np.arange(len(emotion_names)),
    target_names=emotion_names,
    zero_division=0
)

print("\nClassification Report:")
print(report)

# ==========================================
# 10. PLOT AND SAVE CONFUSION MATRIX
# ==========================================

fig, ax = plt.subplots(figsize=(11, 8))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=emotion_names
)

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format="d",
    colorbar=True
)

plt.title("Speech Emotion Recognition - CNN Confusion Matrix")
plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")

plt.tight_layout()

# Save BEFORE displaying the image
fig.savefig(
    str(IMAGE_PATH),
    dpi=300,
    bbox_inches="tight"
)

print("\nCONFUSION MATRIX SAVED SUCCESSFULLY!")
print("Image location:", IMAGE_PATH.resolve())

# Display image
plt.show()

plt.close(fig)

print("\nPROCESS COMPLETED!")
