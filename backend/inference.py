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
import csv

class BirdClassifier:
    def __init__(self):
        self.config = {
            "SAMPLE_RATE": 32000,
            "DURATION": 3.0,
            "FMIN": 300,
            "FMAX": 14000,
            "N_FFT": 2048,
            "HOP_LENGTH": 250,
            "N_MELS": 384,
            "PCEN_TIME_CONSTANT": 0.060,
            "PCEN_GAIN": 0.85,
            "PCEN_BIAS": 10.0,
            "PCEN_POWER": 0.25,
            "PCEN_EPS": 1e-6,
            "MODEL_NAME": 'efficientnet_b4',
            "IMG_SIZE": 384,
            "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, "models")
        self.model_path = os.path.join(self.models_dir, "best_model_b4_mixup.pth")
        self.class_mapping_path = os.path.join(self.models_dir, "ClassMapping.txt")
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
            self.idx_to_class = {}
            try:
                with open(self.class_mapping_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Assumes columns: "Index (Class ID)", "Species Name"
                        idx = int(row["Index (Class ID)"])
                        species = row["Species Name"]
                        self.idx_to_class[idx] = species
            except Exception as e:
                print(f"Error parsing class mapping: {e}")
                return False
        else:
            print(f"Error: Class mapping file not found at {self.class_mapping_path}")
            return False

        num_classes = len(self.idx_to_class)
        if self.config["MODEL_NAME"] == 'efficientnet_b4':
            try:
                weights = models.EfficientNet_B4_Weights.DEFAULT
                self.model = models.efficientnet_b4(weights=weights)
                self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
            except AttributeError:
                 # Fallback for older torchvision versions if B4 weights enum not found, though unlikely given environment
                print("Warning: EfficientNet_B4_Weights not found, using pretrained=True if possible or default init.")
                self.model = models.efficientnet_b4(pretrained=True)
                self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        else:
            print(f"Error: Unsupported model name {self.config['MODEL_NAME']}")
            return False

        if not os.path.exists(self.model_path):
            print(f"Error: Model file not found at {self.model_path}")
            return False

        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.config["DEVICE"], weights_only=True))
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
                # Load with fixed SR
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

        # 1. Load full audio to check duration and prepare for segmentation
        try:
            suffix = os.path.splitext(filename)[1] if filename else ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name

            try:
                # Load audio with offset and duration if provided
                y_full, sr = librosa.load(tmp_path, sr=self.config["SAMPLE_RATE"], mono=True, offset=offset, duration=duration)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        except Exception as e:
            print(f"Error loading audio for prediction: {e}")
            return None, None

        total_duration = len(y_full) / sr
        target_duration = self.config["DURATION"]
        
        # 2. Check Minimum Duration
        if total_duration < target_duration:
            print(f"Error: Audio too short ({total_duration:.2f}s). Minimum required is {target_duration}s.")
            return None, None

        # --- Signal Processing on FULL AUDIO (Preserve PCEN Context) ---
        # 1. Melspectrogram
        S = librosa.feature.melspectrogram(
            y=y_full, sr=sr, n_fft=self.config["N_FFT"], hop_length=self.config["HOP_LENGTH"],
            n_mels=self.config["N_MELS"], fmin=self.config["FMIN"], fmax=self.config["FMAX"]
        )

        # 2. PCEN
        pcen = librosa.pcen(
            S * (2 ** 31), sr=sr, hop_length=self.config["HOP_LENGTH"],
            time_constant=self.config["PCEN_TIME_CONSTANT"], gain=self.config["PCEN_GAIN"],
            bias=self.config["PCEN_BIAS"], power=self.config["PCEN_POWER"], eps=self.config["PCEN_EPS"]
        )

        # 3. Deltas
        delta = librosa.feature.delta(pcen)
        delta2 = librosa.feature.delta(pcen, order=2)

        full_features = np.stack([pcen, delta, delta2], axis=0)
        # full_features shape: [3, n_mels, n_time_frames]
        
        # --- Sliding Window on FEATURES ---
        n_frames_total = full_features.shape[2]
        
        # Calculate window parameters in FRAMES
        # Window = 3.0s * 32000 / 250 = 384 frames
        frames_per_window = int((target_duration * sr) / self.config["HOP_LENGTH"])
        
        # Stride = 2.0s * 32000 / 250 = 256 frames
        stride_sec = 2.0
        frames_stride = int((stride_sec * sr) / self.config["HOP_LENGTH"])
        
        starts = []
        current_start = 0
        while current_start + frames_per_window <= n_frames_total:
            starts.append(current_start)
            current_start += frames_stride
            
        # Handle case where last segment is missing or if audio is exactly length
        # If we haven't covered the end, and we have enough for a partial (which we'd pad), 
        # but here we enforce min duration so we should be good.
        # If the loop didn't run (e.g. total < window, but we checked min duration), 
        # or if we want to capture the tail?
        # The training pipeline usually just takes valid segments.
        # Let's ensure at least one segment if total >= duration (which it is).
        if not starts:
            starts.append(0)
        elif n_frames_total > starts[-1] + frames_per_window:
             # If there's a significant chunk left, maybe take the last window aligned to end?
             # For now, standard stride is fine.
             pass

        accumulated_probs = None
        count = 0
        images = []

        preprocess = transforms.Compose([
            transforms.Resize((self.config["IMG_SIZE"], self.config["IMG_SIZE"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        print(f"Processing {len(starts)} segments from {n_frames_total} frames...")

        for start_frame in starts:
            end_frame = start_frame + frames_per_window
            
            # Extract feature crop
            crop = full_features[:, :, start_frame:end_frame]
            
            # Padding Logic (if crop is smaller than window, though loop logic prevents this mostly)
            if crop.shape[2] < frames_per_window:
                pad_amt = frames_per_window - crop.shape[2]
                crop = np.pad(crop, ((0, 0), (0, 0), (0, pad_amt)), mode='constant')

            img_chans = []
            for ch in range(3):
                if self.normalization_stats and self.normalization_stats.get('ready'):
                    key = f'ch{ch}'
                    min_val = self.normalization_stats[key]['min']
                    max_val = self.normalization_stats[key]['max']
                else:
                    flat = crop[ch].flatten()
                    min_val = np.percentile(flat, 2.0)
                    max_val = np.percentile(flat, 98.0)

                denom = max_val - min_val + 1e-8
                norm = (crop[ch] - min_val) / denom
                norm = np.clip(norm, 0, 1)
                img_chans.append(norm)

            merged = np.stack(img_chans, axis=-1)
            merged = (merged * 255).astype(np.uint8)
            merged = np.flip(merged, axis=0)
            
            image = Image.fromarray(merged)
            images.append(image)

            # Inference
            input_tensor = preprocess(image).unsqueeze(0).to(self.config["DEVICE"])

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            probs = probabilities.cpu().numpy()[0]
            
            if accumulated_probs is None:
                accumulated_probs = probs
            else:
                accumulated_probs += probs
            count += 1

        # Average
        avg_probs = accumulated_probs / count
        top5_indices = avg_probs.argsort()[-5:][::-1]

        results = []
        for idx in top5_indices:
            species = self.idx_to_class[idx]
            score = float(avg_probs[idx])
            results.append({"species": species, "probability": score})

        return results, images

# Singleton instance for easy import if needed, but main.py will instantiate it.
classifier = BirdClassifier()

