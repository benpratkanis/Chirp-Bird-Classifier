import csv
import gc
import json
import multiprocessing
import os
import sys
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import scipy.signal
from PIL import Image
from tqdm import tqdm

# ==========================================
# GLOBAL CONFIGURATION (OPTIMIZED & RELAXED)
# ==========================================
CONFIG = {
    # --- Path Settings ---
    "RAW_AUDIO_DIR": r"C:\BirdData\SORTED_DATA",
    "PROCESSED_IMG_DIR": r"D:\ChirpBirdClassifier\Spectrograms_HighRes_384_Mem",
    "STATS_FILE": "normalization_stats.json",

    # Changed extension to .txt for speed
    "CHECKPOINT_FILE": "processing_checkpoint.txt",
    "AUDIT_LOG": "skipped_files_audit.csv",
    "ERROR_LOG": "processing_errors.log",

    # --- Calibration ---
    "CALIBRATION_RATIO": 0.2,
    "CALIBRATION_MAX": 6000,

    # --- Audio ---
    "SAMPLE_RATE": 32000,
    "FMIN": 300,
    "FMAX": 14000,

    # --- Spectrogram (Targeting 384x384 Square) ---
    "N_FFT": 2048,
    "N_MELS": 384,  # High Frequency Resolution

    # CALCULATED HOP:
    # 3.0s * 32000sr = 96000 samples.
    # 96000 / 384 pixels = 250 hop length.
    "HOP_LENGTH": 250,
    "SAMPLE_LEN_SEC": 3.0,

    # --- Segmentation ---
    # Increased to 12 to capture more from busy clips
    "MAX_SEGMENTS_PER_FILE": 12,

    # --- Detection Thresholds (RELAXED) ---
    # Lowered to 800Hz to catch doves/pigeons
    "BIRD_BAND_MIN_HZ": 800,
    # Raised to 13500Hz for high-pitch warblers
    "BIRD_BAND_MAX_HZ": 13500,
    # Lowered to 0.12 (was 0.15) to be more sensitive to faint calls
    "MIN_ENERGY_THRESHOLD": 0.12,

    # --- PCEN (Color Channels) ---
    "PCEN_TIME_CONSTANT": 0.060,
    "PCEN_GAIN": 0.85,  # Slight boost to contrast
    "PCEN_BIAS": 10.0,
    "PCEN_POWER": 0.25,
    "PCEN_EPS": 1e-6,

    # --- Normalization ---
    "NORM_PERCENTILE_LOW": 5.0,
    "NORM_PERCENTILE_HIGH": 99.5,

    # --- System ---
    "NUM_WORKERS": max(1, (os.cpu_count() or 4) - 1),
    "CHECKPOINT_INTERVAL": 1,  # Unused in new append-only mode
    "MAX_TASKS_PER_CHILD": 20
}


# ==========================================
# UTILITIES
# ==========================================
class CheckpointManager:
    """
    Optimized Append-Only Checkpoint System.
    Prevents the slowdown caused by rewriting large JSON files.
    """

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.processed = set()
        self.load()

    def load(self):
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    # Read lines efficiently
                    self.processed = set(line.strip() for line in f if line.strip())
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")

    def mark_done(self, file_path_str):
        if file_path_str not in self.processed:
            self.processed.add(file_path_str)
            # Instant append to disk - very fast
            with open(self.filepath, 'a') as f:
                f.write(f"{file_path_str}\n")


class NormalizationStats:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.stats = {'ch0': {}, 'ch1': {}, 'ch2': {}, 'ready': False}
        self.load()

    def load(self):
        if self.filepath.exists():
            with open(self.filepath, 'r') as f: self.stats = json.load(f)

    def save_from_batches(self, batch_data):
        if not batch_data: return
        print("Calculating stats...")
        for ch in range(3):
            key = f'ch{ch}'
            lows = [x[ch][0] for x in batch_data]
            highs = [x[ch][1] for x in batch_data]
            self.stats[key] = {'min': float(np.median(lows)), 'max': float(np.median(highs))}
        self.stats['ready'] = True
        with open(self.filepath, 'w') as f:
            json.dump(self.stats, f, indent=2)


# ==========================================
# CORE LOGIC
# ==========================================
def load_audio(filepath):
    try:
        # Load with fixed SR
        y, sr = librosa.load(filepath, sr=CONFIG["SAMPLE_RATE"], mono=True)
        if len(y) < CONFIG["HOP_LENGTH"]: return None, "Too Short"
        return y, "OK"
    except Exception as e:
        return None, str(e)


def compute_features(y):
    # 1. Melspectrogram
    S = librosa.feature.melspectrogram(
        y=y, sr=CONFIG["SAMPLE_RATE"], n_fft=CONFIG["N_FFT"],
        hop_length=CONFIG["HOP_LENGTH"], n_mels=CONFIG["N_MELS"],
        fmin=CONFIG["FMIN"], fmax=CONFIG["FMAX"]
    )

    # 2. PCEN
    pcen = librosa.pcen(
        S * (2 ** 31), sr=CONFIG["SAMPLE_RATE"], hop_length=CONFIG["HOP_LENGTH"],
        time_constant=CONFIG["PCEN_TIME_CONSTANT"], gain=CONFIG["PCEN_GAIN"],
        bias=CONFIG["PCEN_BIAS"], power=CONFIG["PCEN_POWER"], eps=CONFIG["PCEN_EPS"]
    )

    # 3. Deltas
    delta = librosa.feature.delta(pcen)
    delta2 = librosa.feature.delta(pcen, order=2)

    return np.stack([pcen, delta, delta2], axis=0)


def detect_bird_segments(features):
    """
    Intelligent scanner with FAILSAFE.
    """
    pcen = features[0]
    n_mels, n_frames = pcen.shape
    frames_window = int((CONFIG["SAMPLE_LEN_SEC"] * CONFIG["SAMPLE_RATE"]) / CONFIG["HOP_LENGTH"])

    # Map Hz to Mel Bins
    mel_freqs = librosa.mel_frequencies(n_mels=CONFIG["N_MELS"], fmin=CONFIG["FMIN"], fmax=CONFIG["FMAX"])
    bird_bins = np.where((mel_freqs >= CONFIG["BIRD_BAND_MIN_HZ"]) & (mel_freqs <= CONFIG["BIRD_BAND_MAX_HZ"]))[0]

    if len(bird_bins) == 0: return []

    # Calculate energy ONLY in bird range
    bird_energy_profile = np.mean(pcen[bird_bins, :], axis=0)

    # Dynamic Thresholding
    noise_floor = np.median(bird_energy_profile)
    threshold = noise_floor + CONFIG["MIN_ENERGY_THRESHOLD"]

    # Strategy 1: Find Peaks
    peaks, _ = scipy.signal.find_peaks(bird_energy_profile, height=threshold, distance=frames_window)

    segments = []

    if len(peaks) > 0:
        peak_heights = bird_energy_profile[peaks]
        sorted_indices = np.argsort(peak_heights)[::-1]
        peaks = peaks[sorted_indices]

        for peak_idx in peaks:
            if len(segments) >= CONFIG["MAX_SEGMENTS_PER_FILE"]: break
            half_window = frames_window // 2
            start = max(0, peak_idx - half_window)
            end = start + frames_window
            if end > n_frames:
                end = n_frames
                start = max(0, end - frames_window)
            segments.append((start, end))

    # Strategy 2: Failsafe (If no peaks, grab loudest moment)
    if not segments:
        max_idx = np.argmax(bird_energy_profile)
        half_window = frames_window // 2
        start = max(0, max_idx - half_window)
        end = start + frames_window
        if end > n_frames:
            end = n_frames
            start = max(0, end - frames_window)
        segments.append((start, end))

    return segments


def worker(args):
    filepath, species, out_dir, mode, global_stats = args

    try:
        # --- OPTIMIZATION 1: File Size Check ---
        # Skip files > 150MB to prevent RAM swapping/Thrashing
        if os.path.getsize(filepath) > 150 * 1024 * 1024:
            return ('skip', "File too large (>150MB)", 0.0, str(filepath))

        y, status = load_audio(filepath)
        if y is None: return ('skip', status, 0.0, str(filepath))

        # Ultra-low check for dead silence files
        if np.sqrt(np.mean(y ** 2)) < 0.0001:
            return ('skip', "Digital Silence", 0.0, str(filepath))

        features = compute_features(y)

        # --- OPTIMIZATION 2: Aggressive cleanup ---
        del y  # Free raw audio immediately

        if mode == 'stats':
            stats_tuple = []
            for ch in range(3):
                flat = features[ch].flatten()
                stats_tuple.append((
                    np.percentile(flat, CONFIG["NORM_PERCENTILE_LOW"]),
                    np.percentile(flat, CONFIG["NORM_PERCENTILE_HIGH"])
                ))
            return ('stat', tuple(stats_tuple), str(filepath))

        elif mode == 'generate':
            segments = detect_bird_segments(features)

            if not segments:
                del features
                return ('skip', "No Bird Calls", 0.0, str(filepath))

            filename = Path(filepath).stem
            save_dir = Path(out_dir) / species
            save_dir.mkdir(parents=True, exist_ok=True)

            saved = 0
            for idx, (start, end) in enumerate(segments):
                crop = features[:, :, start:end]

                # Padding Logic
                target_w = int((CONFIG["SAMPLE_LEN_SEC"] * CONFIG["SAMPLE_RATE"] // CONFIG["HOP_LENGTH"]))
                if crop.shape[2] < target_w:
                    pad_amt = target_w - crop.shape[2]
                    crop = np.pad(crop, ((0, 0), (0, 0), (0, pad_amt)), mode='constant')

                # Normalize & Colorize
                img_chans = []
                for ch in range(3):
                    stats = global_stats[f'ch{ch}']
                    denom = stats['max'] - stats['min'] + 1e-8
                    norm = (crop[ch] - stats['min']) / denom
                    norm = np.clip(norm, 0, 1)
                    img_chans.append(norm)

                merged = np.stack(img_chans, axis=-1)
                merged = (merged * 255).astype(np.uint8)
                merged = np.flip(merged, axis=0)

                Image.fromarray(merged).save(save_dir / f"{filename}_{idx}.png")
                saved += 1

            # --- OPTIMIZATION 3: Force Garbage Collection ---
            del features
            del crop
            del merged
            gc.collect()

            return ('done', saved, 0.0, str(filepath))

    except Exception as e:
        return ('error', str(e), 0.0, str(filepath))


def run():
    print("BIRD AUDIO PREPROCESSOR (OPTIMIZED + RELAXED FILTERS)")
    Path(CONFIG["PROCESSED_IMG_DIR"]).mkdir(exist_ok=True)
    norm_stats = NormalizationStats(Path(CONFIG["PROCESSED_IMG_DIR"]) / CONFIG["STATS_FILE"])
    checkpoint = CheckpointManager(Path(CONFIG["PROCESSED_IMG_DIR"]) / CONFIG["CHECKPOINT_FILE"])

    raw_path = Path(CONFIG["RAW_AUDIO_DIR"])
    all_files = [f for ext in ['*.wav', '*.mp3', '*.ogg', '*.flac'] for f in raw_path.rglob(ext)]
    print(f"Found {len(all_files)} files.")

    # 1. Calibration
    if not norm_stats.stats.get('ready'):
        print("Calibrating...")
        subset = np.random.choice(all_files, min(len(all_files), CONFIG['CALIBRATION_MAX']), replace=False)
        tasks = [(f, "", "", 'stats', None) for f in subset]
        results = []
        with multiprocessing.Pool(CONFIG["NUM_WORKERS"]) as pool:
            for res in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks)):
                if res[0] == 'stat': results.append(res[1])
        norm_stats.save_from_batches(results)

    # 2. Generation
    print("Generating High-Res Spectrograms...")

    # Filter using set lookup (very fast)
    to_process = [f for f in all_files if str(f) not in checkpoint.processed]
    print(f"Remaining files to process: {len(to_process)}")

    tasks = [(f, f.parent.name, CONFIG["PROCESSED_IMG_DIR"], 'generate', norm_stats.stats) for f in to_process]

    with multiprocessing.Pool(CONFIG["NUM_WORKERS"], maxtasksperchild=CONFIG["MAX_TASKS_PER_CHILD"]) as pool:
        for res in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks)):
            status, msg, val, fpath = res

            if status == 'done':
                # Immediate append to text file (Low I/O cost)
                checkpoint.mark_done(fpath)

            elif status == 'skip' and "No Bird" not in msg:
                # Log non-bird skips (e.g., silence, too large)
                with open(CONFIG["AUDIT_LOG"], "a", newline='') as f:
                    csv.writer(f).writerow([fpath, msg])


if __name__ == '__main__':
    multiprocessing.freeze_support()
    run()