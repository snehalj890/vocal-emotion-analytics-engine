
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. PROJECT PATHS
# ==========================================

DATASET_PATH = Path("dataset/raw")
OUTPUT_PATH = Path("outputs")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. EMOTION MAPPING
# ==========================================

EMOTIONS = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
}

# ==========================================
# 3. FIND AUDIO FILE
# ==========================================

audio_files = sorted(DATASET_PATH.rglob("*.wav"))

if not audio_files:
    raise FileNotFoundError("No WAV audio files found!")

audio_file = audio_files[0]

parts = audio_file.stem.split("-")

if len(parts) != 7:
    raise ValueError("Invalid RAVDESS filename format.")

emotion = EMOTIONS.get(parts[2], "Unknown")

# ==========================================
# 4. LOAD AND PREPROCESS AUDIO
# ==========================================

audio, sample_rate = librosa.load(
    audio_file,
    sr=22050
)

duration = librosa.get_duration(
    y=audio,
    sr=sample_rate
)

print("\n========== AUDIO ANALYSIS ==========")
print("Audio file:", audio_file.name)
print("Emotion:", emotion)
print("Sample rate:", sample_rate, "Hz")
print("Duration:", round(duration, 2), "seconds")
print("Total samples:", len(audio))

# ==========================================
# 5. SPECTROGRAM EXTRACTION
# ==========================================

stft = librosa.stft(audio)

spectrogram = librosa.amplitude_to_db(
    np.abs(stft),
    ref=np.max
)

# ==========================================
# 6. MFCC FEATURE EXTRACTION
# ==========================================

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=40
)

print("MFCC shape:", mfcc.shape)

# ==========================================
# 7. IMPROVED VISUALIZATION
# ==========================================

fig, axes = plt.subplots(
    3, 1,
    figsize=(13, 14),
    constrained_layout=True
)

# Waveform
librosa.display.waveshow(
    audio,
    sr=sample_rate,
    ax=axes[0]
)

axes[0].set_title(
    f"Speech Waveform - {emotion}",
    pad=12
)

axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Amplitude")

# Spectrogram
img1 = librosa.display.specshow(
    spectrogram,
    sr=sample_rate,
    x_axis="time",
    y_axis="hz",
    ax=axes[1]
)

axes[1].set_title("Speech Spectrogram", pad=12)
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Frequency (Hz)")

fig.colorbar(
    img1,
    ax=axes[1],
    pad=0.02,
    format="%+2.0f dB"
)

# MFCC visualization
# Skip the first coefficient only for display.
mfcc_display = mfcc[1:, :]

vmin, vmax = np.percentile(
    mfcc_display,
    [5, 95]
)

img2 = librosa.display.specshow(
    mfcc_display,
    sr=sample_rate,
    x_axis="time",
    ax=axes[2],
    vmin=vmin,
    vmax=vmax
)

axes[2].set_title(
    "MFCC Features (Coefficients 2-40)",
    pad=12
)

axes[2].set_xlabel("Time (seconds)")
axes[2].set_ylabel("MFCC Coefficients")

fig.colorbar(
    img2,
    ax=axes[2],
    pad=0.02
)

# ==========================================
# 8. SAVE VISUALIZATION
# ==========================================

output_file = OUTPUT_PATH / "audio_analysis.png"

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

print("\nVisualization saved at:", output_file)
print("\nAudio analysis completed successfully!")

plt.show()
