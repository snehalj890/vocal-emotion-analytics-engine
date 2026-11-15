
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

# =====================================================
# 1. REPRODUCIBILITY
# =====================================================

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =====================================================
# 2. PROJECT PATHS
# =====================================================

DATA_PATH = Path("outputs/ravdess_mfcc_features.npz")

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = str(
    MODEL_DIR / "best_speech_emotion_cnn_lstm_v2.keras"
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

emotion_names = [
    str(name) for name in data["emotion_names"].tolist()
]

num_classes = len(emotion_names)

print("\n========== DATASET LOADED ==========")
print("Feature shape:", X.shape)
print("Label shape:", y.shape)
print("Number of emotions:", num_classes)
print("Emotions:", emotion_names)

if not np.isfinite(X).all():
    raise ValueError(
        "Dataset contains NaN or infinite values."
    )

# =====================================================
# 4. ACTOR-INDEPENDENT DATA SPLIT
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

train_actors = set(groups[train_mask])
val_actors = set(groups[val_mask])
test_actors = set(groups[test_mask])

assert train_actors.isdisjoint(val_actors)
assert train_actors.isdisjoint(test_actors)
assert val_actors.isdisjoint(test_actors)

print("\n========== DATASET SPLIT ==========")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)
print("Actor-independent split verified!")

# =====================================================
# 5. MFCC FEATURE PREPARATION
# =====================================================

# Convert (samples, 40, 216)
# into (samples, 216, 40)

X_train = np.transpose(X_train, (0, 2, 1))
X_val = np.transpose(X_val, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))

# Normalize using training data only

mean = X_train.mean(
    axis=(0, 1),
    keepdims=True
)

std = X_train.std(
    axis=(0, 1),
    keepdims=True
) + 1e-8

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

np.savez(
    OUTPUT_DIR / "cnn_lstm_v2_normalization.npz",
    mean=mean,
    std=std
)

print("\nNormalized training shape:", X_train.shape)

# =====================================================
# 6. SPEC AUGMENTATION
# =====================================================

def spec_augment(
    features,
    time_masks=2,
    frequency_masks=2,
    max_time_width=20,
    max_frequency_width=6
):
    """
    Apply time masking and frequency masking
    to normalized MFCC features.

    Input shape: (time frames, MFCC coefficients)
    """

    augmented = features.copy()

    time_steps, frequency_bins = augmented.shape

    # Time masking
    for _ in range(time_masks):

        width = np.random.randint(
            0,
            min(max_time_width, time_steps) + 1
        )

        if width > 0:
            start = np.random.randint(
                0,
                time_steps - width + 1
            )

            augmented[start:start + width, :] = 0.0

    # Frequency masking
    for _ in range(frequency_masks):

        width = np.random.randint(
            0,
            min(max_frequency_width, frequency_bins) + 1
        )

        if width > 0:
            start = np.random.randint(
                0,
                frequency_bins - width + 1
            )

            augmented[:, start:start + width] = 0.0

    return augmented


print("\nApplying SpecAugment to training data...")

# Keep original samples and create two augmented
# versions of every training sample.

X_augmented = [X_train]

for copy_number in range(2):

    augmented_copy = np.stack([
        spec_augment(sample)
        for sample in X_train
    ])

    X_augmented.append(augmented_copy)

X_train_final = np.concatenate(
    X_augmented,
    axis=0
)

y_train_final = np.tile(
    y_train,
    3
)

# Shuffle augmented training data

rng = np.random.default_rng(SEED)

indices = rng.permutation(
    len(X_train_final)
)

X_train_final = X_train_final[indices]
y_train_final = y_train_final[indices]

print("Original training samples:", len(X_train))
print("Training samples after augmentation:", len(X_train_final))

# Validation and test sets remain unchanged.

# =====================================================
# 7. CLASS WEIGHTS
# =====================================================

classes = np.unique(y_train_final)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train_final
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(classes, weights)
}

print("\n========== CLASS WEIGHTS ==========")
print(class_weights)

# =====================================================
# 8. BUILD CNN-LSTM MODEL
# =====================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    # CNN feature extraction

    tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=5,
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling1D(
        pool_size=2
    ),

    tf.keras.layers.SpatialDropout1D(
        0.25
    ),

    tf.keras.layers.Conv1D(
        filters=96,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling1D(
        pool_size=2
    ),

    tf.keras.layers.SpatialDropout1D(
        0.30
    ),

    # Bidirectional LSTM

    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            48,
            return_sequences=True,
            dropout=0.30,
            recurrent_dropout=0.0
        )
    ),

    tf.keras.layers.GlobalAveragePooling1D(),

    # Dense classification layers

    tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    ),

    tf.keras.layers.Dropout(0.50),

    tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )
])

# =====================================================
# 9. COMPILE MODEL
# =====================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0003
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

print("\n========== CNN-LSTM V2 MODEL ==========")
model.summary()

# =====================================================
# 10. CALLBACKS
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
        patience=10,
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
# 11. TRAIN MODEL
# =====================================================

print("\n========== CNN-LSTM V2 TRAINING STARTED ==========")

history = model.fit(
    X_train_final,
    y_train_final,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=60,
    batch_size=32,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1
)

# =====================================================
# 12. LOAD BEST MODEL
# =====================================================

best_model = tf.keras.models.load_model(
    MODEL_PATH
)

print("\nBest CNN-LSTM V2 model loaded!")

# =====================================================
# 13. TEST EVALUATION
# =====================================================

print("\n========== TEST EVALUATION ==========")

test_loss, test_accuracy = best_model.evaluate(
    X_test,
    y_test,
    verbose=1
)

probabilities = best_model.predict(
    X_test,
    verbose=1
)

y_pred = np.argmax(
    probabilities,
    axis=1
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n========== FINAL CNN-LSTM V2 RESULTS ==========")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")

# =====================================================
# 14. CLASSIFICATION REPORT
# =====================================================

report = classification_report(
    y_test,
    y_pred,
    labels=np.arange(num_classes),
    target_names=emotion_names,
    zero_division=0
)

print("\n========== CLASSIFICATION REPORT ==========")
print(report)

with open(
    OUTPUT_DIR / "cnn_lstm_v2_classification_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(report)

# =====================================================
# 15. SAVE TEST METRICS
# =====================================================

metrics = {
    "model": "CNN-LSTM Version 2 with SpecAugment",
    "test_accuracy": float(test_accuracy),
    "test_loss": float(test_loss),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "emotion_names": emotion_names,
    "training_samples_after_augmentation": int(
        len(X_train_final)
    )
}

with open(
    OUTPUT_DIR / "cnn_lstm_v2_test_metrics.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )

# =====================================================
# 16. TRAINING ACCURACY AND LOSS GRAPHS
# =====================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5)
)

axes[0].plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

axes[0].plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

axes[0].set_title("CNN-LSTM V2 Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(
    history.history["loss"],
    label="Training Loss"
)

axes[1].plot(
    history.history["val_loss"],
    label="Validation Loss"
)

axes[1].set_title("CNN-LSTM V2 Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()

fig.savefig(
    OUTPUT_DIR / "cnn_lstm_v2_training_history.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

# =====================================================
# 17. CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(num_classes)
)

fig, ax = plt.subplots(
    figsize=(10, 8)
)

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

plt.title(
    "CNN-LSTM V2 Speech Emotion Recognition"
)

plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")

plt.tight_layout()

fig.savefig(
    OUTPUT_DIR / "cnn_lstm_v2_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

# =====================================================
# 18. FINAL SUMMARY
# =====================================================

print("\n========== CNN-LSTM V2 COMPLETED ==========")

print("Saved model:", MODEL_PATH)

print("Training graph:")
print("outputs/cnn_lstm_v2_training_history.png")

print("Confusion matrix:")
print("outputs/cnn_lstm_v2_confusion_matrix.png")

print("Classification report:")
print("outputs/cnn_lstm_v2_classification_report.txt")

print("Test metrics:")
print("outputs/cnn_lstm_v2_test_metrics.json")

print("Normalization parameters:")
print("outputs/cnn_lstm_v2_normalization.npz")

print("\nALL TASKS COMPLETED SUCCESSFULLY!")
