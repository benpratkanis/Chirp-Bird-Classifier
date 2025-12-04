import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import librosa
import scipy.signal
import json
import os
import tempfile

class BirdClassifier:
    def __init__(self):
        self.config = {
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
        
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, "models")
        self.model_path = os.path.join(self.models_dir, "best_model_simple.pth")
        self.class_mapping_path = os.path.join(self.models_dir, "class_mapping.pth")
        # Use env var or default relative path for stats
        self.stats_path = os.getenv("NORMALIZATION_STATS_PATH", os.path.join(self.models_dir, "normalization_stats.json"))
        
        self.model = None
        self.idx_to_class = None
        self.normalization_stats = None

    def load_normalization_stats(self):
        if os.path.exists(self.stats_path):
            print(f"Loaded normalization stats from {self.stats_path}")
            with open(self.stats_path, 'r') as f:
                self.normalization_stats = json.load(f)
        else:
            print("Warning: normalization_stats.json not found! Using local estimation.")
            self.normalization_stats = None

    def load_resources(self):
        print(f"Loading resources on {self.config['DEVICE']}...")

        if os.path.exists(self.class_mapping_path):
            self.idx_to_class = torch.load(self.class_mapping_path, weights_only=False)
        else:
            print(f"Error: Class mapping file not found at {self.class_mapping_path}")
            return False

        num_classes = len(self.idx_to_class)
        if self.config["MODEL_NAME"] == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.model = models.efficientnet_b0(weights=weights)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        else:
            print(f"Error: Unsupported model name {self.config['MODEL_NAME']}")
            return False

        if not os.path.exists(self.model_path):
            print(f"Error: Model file not found at {self.model_path}")
            return False

        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.config["DEVICE"]))
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False

        self.model = self.model.to(self.config["DEVICE"])
        self.model.eval()
        
        self.load_normalization_stats()
        print("Resources loaded successfully.")
        return True

    def process_audio_to_image(self, audio_bytes: bytes, filename: str = None, offset: float = 0.0, duration: float = None):
        try:
            suffix = os.path.splitext(filename)[1] if filename else ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name

            try:
                y, sr = librosa.load(tmp_path, sr=self.config["SAMPLE_RATE"], mono=True, offset=offset, duration=duration)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        except Exception as e:
            print(f"Error loading audio: {e}")
            return None

        sos = scipy.signal.butter(10, self.config["FMIN"], 'hp', fs=sr, output='sos')
        y = scipy.signal.sosfilt(sos, y)

        target_length = int(self.config["SAMPLE_RATE"] * self.config["DURATION"])
        if len(y) > target_length:
            y = y[:target_length]
        else:
            padding = target_length - len(y)
            y = np.pad(y, (0, padding), 'constant')

        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=self.config["N_FFT"], hop_length=self.config["HOP_LENGTH"],
            n_mels=self.config["N_MELS"], fmin=self.config["FMIN"], fmax=self.config["FMAX"]
        )

        pcen = librosa.pcen(
            S * (2 ** 31), sr=sr, hop_length=self.config["HOP_LENGTH"],
            time_constant=self.config["PCEN_TIME_CONSTANT"], gain=self.config["PCEN_GAIN"],
            bias=self.config["PCEN_BIAS"], power=self.config["PCEN_POWER"], eps=self.config["PCEN_EPS"]
        )

        delta = librosa.feature.delta(pcen)
        delta2 = librosa.feature.delta(pcen, order=2)

        features = np.stack([pcen, delta, delta2], axis=0)

        img_chans = []
        for ch in range(3):
            if self.normalization_stats and self.normalization_stats.get('ready'):
                key = f'ch{ch}'
                min_val = self.normalization_stats[key]['min']
                max_val = self.normalization_stats[key]['max']
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

    def predict(self, audio_bytes: bytes, filename: str = None, offset: float = 0.0, duration: float = None):
        if self.model is None:
            if not self.load_resources():
                return None, None

        image = self.process_audio_to_image(audio_bytes, filename, offset, duration)
        if image is None:
            return None, None

        preprocess = transforms.Compose([
            transforms.Resize((self.config["IMG_SIZE"], self.config["IMG_SIZE"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        input_tensor = preprocess(image).unsqueeze(0).to(self.config["DEVICE"])

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        probs = probabilities.cpu().numpy()[0]
        top5_indices = probs.argsort()[-5:][::-1]

        results = []
        for idx in top5_indices:
            species = self.idx_to_class[idx]
            score = float(probs[idx])
            results.append({"species": species, "probability": score})

        return results, image

# Singleton instance for easy import if needed, but main.py will instantiate it.
classifier = BirdClassifier()

