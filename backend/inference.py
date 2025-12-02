import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import librosa
import scipy.signal
import json
import os
import io
import tempfile
import shutil

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "SAMPLE_RATE": 32000,
    "DURATION": 5.0,
    "FMIN": 300,
    "FMAX": 14000,
    "N_FFT": 2048,
    "HOP_LENGTH": 320,
    "N_MELS": 224,
    "PCEN_TIME_CONSTANT": 0.060,
    "PCEN_GAIN": 0.8,
    "PCEN_BIAS": 10.0,
    "PCEN_POWER": 0.25,
    "PCEN_EPS": 1e-6,
    "MODEL_NAME": 'efficientnet_b0',
    "IMG_SIZE": 224,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}

# Paths - Update these to match your system structure
# Assuming models are in d:\ChirpBirdClassifier\ChirpPlatform\models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model_simple.pth")
CLASS_MAPPING_PATH = os.path.join(MODELS_DIR, "class_mapping.pth")
STATS_PATH = r"C:\BirdData\Spectrograms\normalization_stats.json" # Keep as is or make relative if possible

# Global variables to hold loaded model and mapping
model = None
idx_to_class = None
normalization_stats = None

def load_normalization_stats():
    """Loads global stats."""
    global normalization_stats
    if os.path.exists(STATS_PATH):
        print(f"Loaded normalization stats from {STATS_PATH}")
        with open(STATS_PATH, 'r') as f:
            normalization_stats = json.load(f)
    else:
        print("⚠️ WARNING: normalization_stats.json not found! Using local estimation.")
        normalization_stats = None

def load_resources():
    """Loads model and class mapping into memory."""
    global model, idx_to_class
    
    print(f"Loading resources on {CONFIG['DEVICE']}...")

    # Load Class Mapping
    if os.path.exists(CLASS_MAPPING_PATH):
        idx_to_class = torch.load(CLASS_MAPPING_PATH, weights_only=False) # weights_only=False for safety if needed, or True if safe
    else:
        print(f"❌ Error: Class mapping file not found at {CLASS_MAPPING_PATH}")
        return False

    # Initialize Model
    num_classes = len(idx_to_class)
    if CONFIG["MODEL_NAME"] == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        print(f"❌ Error: Unsupported model name {CONFIG['MODEL_NAME']}")
        return False

    # Load Weights
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        return False

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=CONFIG["DEVICE"]))
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return False

    model = model.to(CONFIG["DEVICE"])
    model.eval()
    
    load_normalization_stats()
    
    print("✅ Resources loaded successfully.")
    return True

def process_audio_to_image(audio_bytes: bytes, filename: str = None, offset: float = 0.0, duration: float = None):
    """
    Converts audio bytes to V12 spectrogram image.
    """
    try:
        # Save to temporary file to help librosa detect format (especially mp3)
        suffix = os.path.splitext(filename)[1] if filename else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            y, sr = librosa.load(tmp_path, sr=CONFIG["SAMPLE_RATE"], mono=True, offset=offset, duration=duration)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return None

    # Bandpass Filter
    sos = scipy.signal.butter(10, CONFIG["FMIN"], 'hp', fs=sr, output='sos')
    y = scipy.signal.sosfilt(sos, y)

    # Enforce 5 Seconds
    target_length = int(CONFIG["SAMPLE_RATE"] * CONFIG["DURATION"])
    if len(y) > target_length:
        y = y[:target_length]
    else:
        padding = target_length - len(y)
        y = np.pad(y, (0, padding), 'constant')

    # Generate Features
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=CONFIG["N_FFT"], hop_length=CONFIG["HOP_LENGTH"],
        n_mels=CONFIG["N_MELS"], fmin=CONFIG["FMIN"], fmax=CONFIG["FMAX"]
    )

    pcen = librosa.pcen(
        S * (2 ** 31), sr=sr, hop_length=CONFIG["HOP_LENGTH"],
        time_constant=CONFIG["PCEN_TIME_CONSTANT"], gain=CONFIG["PCEN_GAIN"],
        bias=CONFIG["PCEN_BIAS"], power=CONFIG["PCEN_POWER"], eps=CONFIG["PCEN_EPS"]
    )

    delta = librosa.feature.delta(pcen)
    delta2 = librosa.feature.delta(pcen, order=2)

    features = np.stack([pcen, delta, delta2], axis=0)

    # Normalize
    img_chans = []
    for ch in range(3):
        if normalization_stats and normalization_stats.get('ready'):
            key = f'ch{ch}'
            min_val = normalization_stats[key]['min']
            max_val = normalization_stats[key]['max']
        else:
            flat = features[ch].flatten()
            min_val = np.percentile(flat, 2.0)
            max_val = np.percentile(flat, 98.0)

        denom = max_val - min_val + 1e-8
        norm = (features[ch] - min_val) / denom
        norm = np.clip(norm, 0, 1)
        img_chans.append(norm)

    merged = np.stack(img_chans, axis=-1)
    merged = (merged * 255).astype(np.uint8)
    merged = np.flip(merged, axis=0)

    return Image.fromarray(merged)

def predict(audio_bytes: bytes, filename: str = None, offset: float = 0.0, duration: float = None):
    """
    Runs inference on audio bytes.
    Returns: list of (species, probability) tuples, and base64 image string (or image object).
    """
    if model is None:
        if not load_resources():
            return None, None

    image = process_audio_to_image(audio_bytes, filename, offset, duration)
    if image is None:
        return None, None

    # Preprocess for Model
    preprocess = transforms.Compose([
        transforms.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(image).unsqueeze(0).to(CONFIG["DEVICE"])

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    probs = probabilities.cpu().numpy()[0]
    top5_indices = probs.argsort()[-5:][::-1]

    results = []
    for idx in top5_indices:
        species = idx_to_class[idx]
        score = float(probs[idx])
        results.append({"species": species, "probability": score})

    return results, image
