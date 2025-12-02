import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import librosa
import scipy.signal
import json
import os
import sys

# ==========================================
# 1. CONFIGURATION (Must Match V12 Preprocessor)
# ==========================================
CONFIG = {
    # --- Input File ---
    # Supports Audio: .wav, .mp3, .flac, .ogg, .m4a, .aac, .wma
    # Supports Video: .mp4, .avi, .mov, .mkv, .webm, .flv (extracts audio automatically)
    "AUDIO_FILE_PATH": r"C:\Users\benpr\Downloads\ScreenRecording_11-22-2025 20-36-11_1.mp3",  # <--- CHANGE THIS

    # --- Paths ---
    "MODEL_PATH": r"D:\ChirpBirdClassifier\models\best_model_simple.pth",
    "CLASS_MAPPING_PATH": r"D:\ChirpBirdClassifier\models\class_mapping.pth",
    # This file is created by your preprocessor. It's needed for correct colors.
    "STATS_PATH": r"C:\BirdData\Spectrograms\normalization_stats.json",

    # --- Audio Settings (V12) ---
    "SAMPLE_RATE": 32000,
    "DURATION": 5.0,
    "FMIN": 300,
    "FMAX": 14000,
    "N_FFT": 2048,
    "HOP_LENGTH": 320,
    "N_MELS": 224,  # High Res

    # --- PCEN Settings (V12) ---
    "PCEN_TIME_CONSTANT": 0.060,
    "PCEN_GAIN": 0.8,
    "PCEN_BIAS": 10.0,
    "PCEN_POWER": 0.25,
    "PCEN_EPS": 1e-6,

    # --- Model Settings ---
    "MODEL_NAME": 'efficientnet_b0',
    "IMG_SIZE": 224,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}


# ==========================================
# 2. DSP & FEATURE EXTRACTION (Matches V12)
# ==========================================
def load_normalization_stats():
    """Loads global stats to ensure colors match training data."""
    if os.path.exists(CONFIG["STATS_PATH"]):
        print(f"Loaded normalization stats from {CONFIG['STATS_PATH']}")
        with open(CONFIG["STATS_PATH"], 'r') as f:
            return json.load(f)
    else:
        print("⚠️ WARNING: normalization_stats.json not found!")
        print("Using local min/max estimation. Results may be less accurate.")
        return None


def process_audio_to_v12_image(file_path, stats):
    """
    Loads audio/video file, extracts audio, converts to V12 spectrogram.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return None

    print(f"Processing file: {os.path.basename(file_path)}")

    # 1. Load Audio (Handles Video & Audio formats)
    # Librosa uses ffmpeg backend (via audioread) to handle mp4, avi, mov, mp3, etc.
    try:
        y, sr = librosa.load(file_path, sr=CONFIG["SAMPLE_RATE"], mono=True)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        print("💡 TIP: For video files (.mp4, .avi) or compressed audio (.mp3, .m4a),")
        print("        ensure FFmpeg is installed and added to your system PATH.")
        return None

    # 2. Bandpass Filter (Butterworth High-pass)
    sos = scipy.signal.butter(10, CONFIG["FMIN"], 'hp', fs=sr, output='sos')
    y = scipy.signal.sosfilt(sos, y)

    # 3. Enforce 5 Seconds (Pad or Crop)
    target_length = int(CONFIG["SAMPLE_RATE"] * CONFIG["DURATION"])
    if len(y) > target_length:
        y = y[:target_length]  # Crop to first 5s
    else:
        padding = target_length - len(y)
        y = np.pad(y, (0, padding), 'constant')

    # 4. Generate Features (PCEN + Deltas)
    # Mel Spectrogram
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=CONFIG["N_FFT"], hop_length=CONFIG["HOP_LENGTH"],
        n_mels=CONFIG["N_MELS"], fmin=CONFIG["FMIN"], fmax=CONFIG["FMAX"]
    )

    # PCEN (Channel 0: Energy)
    pcen = librosa.pcen(
        S * (2 ** 31), sr=sr, hop_length=CONFIG["HOP_LENGTH"],
        time_constant=CONFIG["PCEN_TIME_CONSTANT"], gain=CONFIG["PCEN_GAIN"],
        bias=CONFIG["PCEN_BIAS"], power=CONFIG["PCEN_POWER"], eps=CONFIG["PCEN_EPS"]
    )

    # Deltas (Channel 1: Velocity, Channel 2: Acceleration)
    delta = librosa.feature.delta(pcen)
    delta2 = librosa.feature.delta(pcen, order=2)

    # Stack Channels
    features = np.stack([pcen, delta, delta2], axis=0)  # Shape: (3, n_mels, time)

    # 5. Normalize & Convert to Image
    img_chans = []
    for ch in range(3):
        # Select stats source
        if stats and stats.get('ready'):
            key = f'ch{ch}'
            min_val = stats[key]['min']
            max_val = stats[key]['max']
        else:
            # Fallback: Use local stats (2nd/98th percentile)
            flat = features[ch].flatten()
            min_val = np.percentile(flat, 2.0)
            max_val = np.percentile(flat, 98.0)

        denom = max_val - min_val + 1e-8
        norm = (features[ch] - min_val) / denom
        norm = np.clip(norm, 0, 1)
        img_chans.append(norm)

    # Merge to RGB
    merged = np.stack(img_chans, axis=-1)  # Shape: (n_mels, time, 3)
    merged = (merged * 255).astype(np.uint8)

    # Flip frequency axis (Low freq at bottom)
    merged = np.flip(merged, axis=0)

    return Image.fromarray(merged)


# ==========================================
# 3. MODEL & INFERENCE
# ==========================================
def load_trained_model(model_path, num_classes, device):
    print(f"Loading model from {model_path}...")

    # Initialize architecture (Must match training)
    if CONFIG["MODEL_NAME"] == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif CONFIG["MODEL_NAME"] == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Model name in CONFIG not supported by this snippet.")

    # Load Weights
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return None

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return None

    model = model.to(device)
    model.eval()
    return model


def main():
    # Check Config Paths
    if CONFIG["AUDIO_FILE_PATH"] == r"path_to_your_test_file.mp4":
        print("⚠️ Please edit the script and set 'AUDIO_FILE_PATH' to your real file.")
        return

    # 1. Load Class Mapping
    if os.path.exists(CONFIG["CLASS_MAPPING_PATH"]):
        idx_to_class = torch.load(CONFIG["CLASS_MAPPING_PATH"])
        num_classes = len(idx_to_class)
    else:
        print("❌ Error: Class mapping file not found. Cannot map IDs to names.")
        return

    # 2. Load Stats & Process Audio
    stats = load_normalization_stats()
    image = process_audio_to_v12_image(CONFIG["AUDIO_FILE_PATH"], stats)
    if image is None: return

    # 3. Preprocess for Model (Resize & Normalize)
    preprocess = transforms.Compose([
        transforms.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(image).unsqueeze(0).to(CONFIG["DEVICE"])

    # 4. Load Model & Predict
    model = load_trained_model(CONFIG["MODEL_PATH"], num_classes, CONFIG["DEVICE"])
    if model is None: return

    print("\nRunning inference...")
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    # 5. Output Results
    probs = probabilities.cpu().numpy()[0]
    top5_indices = probs.argsort()[-5:][::-1]

    print("\n--- RESULTS ---")
    print(f"{'Species':<20} | {'Probability':<10}")
    print("-" * 35)
    for idx in top5_indices:
        species = idx_to_class[idx]
        score = probs[idx] * 100
        print(f"{species:<20} | {score:.2f}%")


if __name__ == "__main__":
    main()