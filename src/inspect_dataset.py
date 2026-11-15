
from pathlib import Path
from collections import Counter

# Dataset directory
DATASET_PATH = Path("dataset/raw")

# RAVDESS emotion labels
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

# Find all audio files
audio_files = list(DATASET_PATH.rglob("*.wav"))

speech_files = []

# Read RAVDESS filenames
for file in audio_files:

    parts = file.stem.split("-")

    if len(parts) == 7:

        modality, vocal_channel, emotion = parts[:3]

        # 03 = Audio-only
        # 01 = Speech
        if modality == "03" and vocal_channel == "01":

            emotion_name = EMOTIONS.get(emotion)

            if emotion_name:
                speech_files.append({
                    "file": file,
                    "emotion": emotion_name
                })

# Calculate emotion distribution
emotion_counts = Counter(
    item["emotion"] for item in speech_files
)

# Display report
print("\n========== RAVDESS DATASET REPORT ==========")

print("Total WAV files found:", len(audio_files))
print("Total speech recordings:", len(speech_files))

print("\nEmotion Distribution:")

for emotion, count in sorted(emotion_counts.items()):
    print(f"{emotion.capitalize():12} : {count}")

print("\nTotal emotion categories:", len(emotion_counts))

# Dataset verification
if len(speech_files) == 1440 and len(emotion_counts) == 8:
    print("\nSUCCESS: Dataset verified!")
else:
    print("\nWARNING: Dataset count differs from expected.")
    print("Please check your dataset folder and filenames.")
