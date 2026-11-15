
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. REPRODUCIBILITY
# ==========================================

np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# 2. PROJECT PATHS
# ==========================================

DATA_PATH = Path("outputs/ravdess_mfcc_features.npz")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = str(MODEL_DIR / "best_speech_emotion_cnn.keras")

# ==========================================
# 3. LOAD MFCC DATASET
# ==========================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Feature dataset not found: {DATA_PATH}"
    )

data = np.load(DATA_PATH, allow_pickle=True)

X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)
groups = data["groups"].astype(np.int64)

emotion_names = data["emotion_names"].tolist()

print("\nDATASET LOADED SUCCESSFULLY")
print("--------------------------------")
print("Feature shape:", X.shape)
print("Label shape:", y.shape)
print("Total recordings:", len(X))
print("Emotions:", emotion_names)

# ==========================================
# 4. ACTOR-INDEPENDENT DATA SPLITTING
# ==========================================

train_mask = np.isin(groups, range(1, 19))
val_mask = np.isin(groups, range(19, 22))
test_mask = np.isin(groups, range(22, 25))

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]

print("\nDATASET SPLIT")
print("--------------------------------")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# Verify actor separation
assert set(groups[train_mask]).isdisjoint(set(groups[val_mask]))
assert set(groups[train_mask]).isdisjoint(set(groups[test_mask]))
assert set(groups[val_mask]).isdisjoint(set(groups[test_mask]))

print("Actor separation verified successfully!")

# ==========================================
# 5. FEATURE NORMALIZATION
# ==========================================

mean = X_train.mean(axis=(0, 2), keepdims=True)
std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# Add channel dimension
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

input_shape = X_train.shape[1:]

# Save normalization parameters for future predictions
np.savez(
    OUTPUT_DIR / "normalization_parameters.npz",
    mean=mean,
    std=std
)

# ==========================================
# 6. CLASS WEIGHTS
# ==========================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("\nCLASS WEIGHTS")
print(class_weights)

# ==========================================
# 7. BUILD CNN MODEL
# ==========================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape),

    tf.keras.layers.Conv2D(
        32, (3, 3),
        padding="same",
        activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(
        64, (3, 3),
        padding="same",
        activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Conv2D(
        128, (3, 3),
        padding="same",
        activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.40),

    tf.keras.layers.Dense(
        len(emotion_names),
        activation="softmax"
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN MODEL SUMMARY")
model.summary()

# ==========================================
# 8. TRAINING CALLBACKS
# ==========================================

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# ==========================================
# 9. TRAIN CNN MODEL
# ==========================================

print("\nSTARTING CNN TRAINING...")
print("--------------------------------")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ==========================================
# 10. LOAD BEST MODEL
# ==========================================

best_model = tf.keras.models.load_model(MODEL_PATH)

# ==========================================
# 11. TEST MODEL
# ==========================================

print("\nEVALUATING ON UNSEEN TEST ACTORS...")

test_loss, test_accuracy = best_model.evaluate(
    X_test,
    y_test,
    verbose=1
)

predictions = best_model.predict(X_test, verbose=1)
y_pred = np.argmax(predictions, axis=1)

print("\nFINAL TEST RESULTS")
print("--------------------------------")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# ==========================================
# 12. CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_test,
    y_pred,
    labels=np.arange(len(emotion_names)),
    target_names=emotion_names,
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print("--------------------------------")
print(report)

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

# ==========================================
# 13. SAVE TEST METRICS
# ==========================================

metrics = {
    "test_accuracy": float(test_accuracy),
    "test_loss": float(test_loss),
    "emotion_names": emotion_names
}

with open(
    OUTPUT_DIR / "test_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=4)

# ==========================================
# 14. TRAINING HISTORY GRAPH
# ==========================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "training_history.png",
    dpi=300
)

plt.show()
plt.close()

# ==========================================
# 15. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(emotion_names))
)

fig, ax = plt.subplots(figsize=(10, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=emotion_names
)

disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45
)

plt.title("Speech Emotion Recognition - CNN")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300
)

plt.show()
plt.close()

# ==========================================
# 16. FINAL SUMMARY
# ==========================================

print("\nCNN TRAINING AND EVALUATION COMPLETED!")
print("--------------------------------")
print("Saved model:", MODEL_PATH)
print("Training graph: outputs/training_history.png")
print("Confusion matrix: outputs/confusion_matrix.png")
print("Classification report: outputs/classification_report.txt")
print("Test metrics: outputs/test_metrics.json")
print("Normalization: outputs/normalization_parameters.npz")
