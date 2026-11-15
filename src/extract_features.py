
from pathlib import Path
import numpy as np
import librosa

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_PATH = Path("dataset/raw")
OUTPUT_PATH = Path("outputs")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 22050
DURATION = 5
N_MFCC = 40
HOP_LENGTH = 512
N_FFT = 2048

MAX_SAMPLES = SAMPLE_RATE * DURATION

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Fixed label encoding
EMOTION_LABELS = {
    emotion: index
    for index, emotion in enumerate(EMOTIONS.values())
}

# ==========================================
# 2. FIND SPEECH AUDIO FILES
# ==========================================

audio_files = sorted(DATASET_PATH.rglob("*.wav"))

features = []
labels = []
actor_ids = []
filenames = []

print("\nStarting MFCC feature extraction...")
print("Total WAV files found:", len(audio_files))

# ==========================================
# 3. PROCESS EACH AUDIO RECORDING
# ==========================================

for index, file in enumerate(audio_files):

    parts = file.stem.split("-")

    if len(parts) != 7:
        continue

    modality, channel, emotion_code = parts[:3]
    actor_code = parts[6]

    # Only audio-only speech recordings
    if modality != "03" or channel != "01":
        continue

    if emotion_code not in EMOTIONS:
        continue

    try:

        # Load audio
        audio, sr = librosa.load(
            file,
            sr=SAMPLE_RATE,
            mono=True
        )

        # Remove leading and trailing silence
        audio, _ = librosa.effects.trim(
            audio,
            top_db=25
        )

        # Pad or truncate to fixed duration
        if len(audio) < MAX_SAMPLES:

            audio = np.pad(
                audio,
                (0, MAX_SAMPLES - len(audio))
            )

        else:
            audio = audio[:MAX_SAMPLES]

        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        # Store features and metadata
        features.append(mfcc)

        labels.append(
            EMOTION_LABELS[EMOTIONS[emotion_code]]
        )

        actor_ids.append(int(actor_code))

        filenames.append(file.name)

        # Progress report
        if (index + 1) % 100 == 0:
            print(
                f"Processed {index + 1}/{len(audio_files)} files"
            )

    except Exception as error:

        print(f"Error processing {file.name}: {error}")

# ==========================================
# 4. CONVERT TO NUMPY ARRAYS
# ==========================================

X = np.array(features, dtype=np.float32)
y = np.array(labels, dtype=np.int64)
groups = np.array(actor_ids, dtype=np.int64)
filenames = np.array(filenames)

# ==========================================
# 5. SAVE EXTRACTED FEATURES
# ==========================================

output_file = OUTPUT_PATH / "ravdess_mfcc_features.npz"

np.savez_compressed(
    output_file,
    X=X,
    y=y,
    groups=groups,
    filenames=filenames,
    emotion_names=np.array(list(EMOTIONS.values())),
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC
)

# ==========================================
# 6. DISPLAY EXTRACTION REPORT
# ==========================================

print("\n========== MFCC EXTRACTION REPORT ==========")

print("Total recordings processed:", len(X))
print("Feature array shape:", X.shape)
print("Label array shape:", y.shape)
print("Actor ID array shape:", groups.shape)

print("\nEmotion distribution:")

for emotion, label in EMOTION_LABELS.items():

    count = np.sum(y == label)

    print(f"{emotion.capitalize():12}: {count}")

print("\nUnique actors:", len(np.unique(groups)))

print("\nSaved feature dataset:", output_file)

if len(X) == 1440 and len(np.unique(y)) == 8:
    print("\nSUCCESS: All MFCC features extracted!")
else:
    print("\nWARNING: Check the extraction results.")
