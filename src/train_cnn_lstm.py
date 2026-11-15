
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

# =====================================================
# 1. REPRODUCIBILITY
# =====================================================

np.random.seed(42)
tf.random.set_seed(42)

# =====================================================
# 2. PROJECT PATHS
# =====================================================

DATA_PATH = Path("outputs/ravdess_mfcc_features.npz")

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = str(
    MODEL_DIR / "best_speech_emotion_cnn_lstm.keras"
)

# =====================================================
# 3. LOAD DATASET
# =====================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

data = np.load(DATA_PATH, allow_pickle=True)

X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)
groups = data["groups"].astype(np.int64)

emotion_names = data["emotion_names"].tolist()

print("\n========== DATASET LOADED ==========")
print("Feature shape:", X.shape)
print("Label shape:", y.shape)
print("Total recordings:", len(X))
print("Emotions:", emotion_names)

# Check feature values
if not np.isfinite(X).all():
    raise ValueError("MFCC dataset contains NaN or infinite values.")

# =====================================================
# 4. ACTOR-INDEPENDENT DATA SPLITTING
# =====================================================

train_mask = np.isin(groups, range(1, 19))
val_mask = np.isin(groups, range(19, 22))
test_mask = np.isin(groups, range(22, 25))

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]

# Verify actor separation
assert set(groups[train_mask]).isdisjoint(set(groups[val_mask]))
assert set(groups[train_mask]).isdisjoint(set(groups[test_mask]))
assert set(groups[val_mask]).isdisjoint(set(groups[test_mask]))

print("\n========== DATASET SPLIT ==========")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)
print("Actor separation verified!")

# =====================================================
# 5. PREPARE MFCC FEATURES
# =====================================================

# Original shape: (samples, 40 MFCC, 216 frames)
# New shape: (samples, 216 frames, 40 MFCC)

X_train = np.transpose(X_train, (0, 2, 1))
X_val = np.transpose(X_val, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))

# Calculate normalization using training data only
mean = X_train.mean(axis=(0, 1), keepdims=True)
std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# Save normalization parameters for future predictions
np.savez(
    OUTPUT_DIR / "cnn_lstm_normalization.npz",
    mean=mean,
    std=std
)

print("\nNormalized feature shape:", X_train.shape)

# =====================================================
# 6. CLASS WEIGHTS
# =====================================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("\n========== CLASS WEIGHTS ==========")
print(class_weights)

# =====================================================
# 7. BUILD CNN-LSTM MODEL
# =====================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(X_train.shape[1], X_train.shape[2])
    ),

    # Mild training-time noise augmentation
    tf.keras.layers.GaussianNoise(0.03),

    # CNN feature extraction
    tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=5,
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling1D(pool_size=2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling1D(pool_size=2),

    tf.keras.layers.Dropout(0.30),

    # LSTM temporal learning
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            64,
            return_sequences=True,
            dropout=0.25
        )
    ),

    # Aggregate sequence information
    tf.keras.layers.GlobalAveragePooling1D(),

    # Classification layers
    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.40),

    tf.keras.layers.Dense(
        len(emotion_names),
        activation="softmax"
    )
])

# =====================================================
# 8. COMPILE MODEL
# =====================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

print("\n========== CNN-LSTM MODEL SUMMARY ==========")
model.summary()

# =====================================================
# 9. TRAINING CALLBACKS
# =====================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )
]

# =====================================================
# 10. TRAIN CNN-LSTM MODEL
# =====================================================

print("\n========== STARTING CNN-LSTM TRAINING ==========")

history = model.fit(
    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=60,
    batch_size=32,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1
)

# =====================================================
# 11. LOAD BEST MODEL
# =====================================================

print("\nLoading best CNN-LSTM model...")

best_model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Best model loaded successfully!")

# =====================================================
# 12. EVALUATE ON UNSEEN TEST ACTORS
# =====================================================

print("\n========== TEST EVALUATION ==========")

test_loss, test_accuracy = best_model.evaluate(
    X_test,
    y_test,
    verbose=1
)

predictions = best_model.predict(
    X_test,
    verbose=1
)

y_pred = np.argmax(predictions, axis=1)

print("\nFINAL CNN-LSTM RESULTS")
print("--------------------------------")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# =====================================================
# 13. CLASSIFICATION REPORT
# =====================================================

report = classification_report(
    y_test,
    y_pred,
    labels=np.arange(len(emotion_names)),
    target_names=emotion_names,
    zero_division=0
)

print("\n========== CLASSIFICATION REPORT ==========")
print(report)

with open(
    OUTPUT_DIR / "cnn_lstm_classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

# =====================================================
# 14. SAVE TEST METRICS
# =====================================================

metrics = {
    "model": "CNN-LSTM",
    "test_accuracy": float(test_accuracy),
    "test_loss": float(test_loss),
    "emotion_names": emotion_names
}

with open(
    OUTPUT_DIR / "cnn_lstm_test_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=4)

# =====================================================
# 15. PLOT TRAINING ACCURACY AND LOSS
# =====================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN-LSTM Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN-LSTM Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "cnn_lstm_training_history.png",
    dpi=300,
    bbox_inches="tight"
)


from pathlib import Path

# Create outputs folder if it does not exist
Path("outputs").mkdir(parents=True, exist_ok=True)

# Save CNN-LSTM training graphs
plt.savefig(
    "outputs/cnn_lstm_training_history.png",
    dpi=300,
    bbox_inches="tight"
)

print("Training graph saved successfully!")
print("Location: outputs/cnn_lstm_training_history.png")

plt.show()


plt.show()
plt.close()

# =====================================================
# 16. GENERATE CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(emotion_names))
)

fig, ax = plt.subplots(figsize=(10, 8))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=emotion_names
)

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format="d"
)

plt.title("CNN-LSTM Speech Emotion Recognition")
plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")

plt.tight_layout()

# Save before displaying
fig.savefig(
    OUTPUT_DIR / "cnn_lstm_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

# =====================================================
# 17. FINAL SUMMARY
# =====================================================

print("\n========== CNN-LSTM COMPLETED ==========")

print("Saved model:", MODEL_PATH)

print(
    "Training graph:",
    "outputs/cnn_lstm_training_history.png"
)

print(
    "Confusion matrix:",
    "outputs/cnn_lstm_confusion_matrix.png"
)

print(
    "Classification report:",
    "outputs/cnn_lstm_classification_report.txt"
)

print(
    "Test metrics:",
    "outputs/cnn_lstm_test_metrics.json"
)

print(
    "Normalization:",
    "outputs/cnn_lstm_normalization.npz"
)

print("\nALL TASKS COMPLETED SUCCESSFULLY!")
