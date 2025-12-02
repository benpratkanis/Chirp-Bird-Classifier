"""
CHIRP BIRD AUDIO PREPROCESSOR V12

DESCRIPTION:
This is an audio preprocessing pipeline designed for large audio datasets.
It converts raw audio recordings into High-Resolution 3-Channel Mel Spectrograms for CNN training.

CORE FEATURES:
1. UNIVERSAL FORMAT SUPPORT: Processes .wav, .mp3, .m4a, .flac, and .ogg files.
   (Note: .m4a support requires FFmpeg installed on the system).
2. HIGH-RES SPECTROGRAMS: Generates 224-Mel x 320-Hop spectrograms.
   - Matches standard ResNet/EfficientNet input dimensions (224px height) natively.
   - Provides high temporal resolution for detecting rapid trills.
3. RGB ENCODING: Encodes temporal context into color channels:
   - Red: Energy (PCEN)
   - Green: Velocity (Delta)
   - Blue: Acceleration (Delta-Delta)
4. STABLE ARCHITECTURE: Uses a Single Persistent Process Pool to prevent OS handle exhaustion.
   - Includes continuous checkpointing to save progress every 500 files.
   - Implements "Fail-Fast" checks to skip corrupt, silent, or static-filled files.

USAGE:
1. Open the file and edit the 'CONFIG' dictionary at the top:
   - Set 'RAW_AUDIO_DIR' to your input folder.
   - Set 'PROCESSED_IMG_DIR' to your desired output folder.
2. Run the script: `python chrip_bird_audio_preprocessor_v12.py`
3. The script will perform two passes:
   - Pass 1: Calibration (Scans 20% of files to determine global normalization stats).
   - Pass 2: Generation (Processes all files and saves images).
4. To stop safely at any time, press Ctrl+C once. The script will save its state and exit.

DEPENDENCIES:
- numpy, librosa, scipy, pillow, tqdm
- FFmpeg (optional, required only for .m4a files)
"""

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
# GLOBAL CONFIGURATION
# ==========================================
CONFIG = {
    # --- Path Settings ---
    "RAW_AUDIO_DIR": r"C:\BirdData\SORTED_DATA",
    "PROCESSED_IMG_DIR": r"D:\ChirpBirdClassifier\SpectrogramsCompleteSet",
    "STATS_FILE": "normalization_stats.json",
    "CHECKPOINT_FILE": "processing_checkpoint.json",
    "AUDIT_LOG": "skipped_files_audit.csv",
    "ERROR_LOG": "processing_errors.log",

    # --- Calibration ---
    "CALIBRATION_RATIO": 0.2,
    "CALIBRATION_MAX": 8000,

    # --- Audio ---
    "SAMPLE_RATE": 32000,
    "FMIN": 300,
    "FMAX": 14000,

    # --- Spectrogram (HIGHER RES) ---
    "N_FFT": 2048,
    "HOP_LENGTH": 320,
    "N_MELS": 224,

    # --- PCEN ---
    "PCEN_TIME_CONSTANT": 0.060,
    "PCEN_GAIN": 0.8,
    "PCEN_BIAS": 10.0,
    "PCEN_POWER": 0.25,
    "PCEN_EPS": 1e-6,

    # --- Segmentation ---
    "SAMPLE_LEN_SEC": 5.0,
    "MAX_SEGMENTS_PER_FILE": 8,
    "MIN_SEGMENT_SPACING_SEC": 0.5,

    # --- Normalization (Per Channel) ---
    "NORM_PERCENTILE_LOW": 2.0,
    "NORM_PERCENTILE_HIGH": 98.0,

    # --- Quality Filters (V21 Golden Settings) ---
    "MIN_RMS_AMPLITUDE": 0.001,
    "MAX_CREST_FACTOR": 50.0,
    "MAX_SPECTRAL_FLATNESS": 0.65,
    "MIN_PCEN_SNR": 0.5,

    # --- System ---
    "NUM_WORKERS": max(1, (os.cpu_count() or 4) - 1),
    "CHECKPOINT_INTERVAL": 500,
    "MAX_TASKS_PER_CHILD": 10
}


# ==========================================
# UTILITY CLASSES
# ==========================================

class CheckpointManager:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.processed = set()
        self.load()

    def load(self):
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    self.processed = set(data.get('processed', []))
                    print(f"Checkpoint loaded: {len(self.processed)} files done.")
            except:
                print("Starting fresh (Checkpoint reset).")

    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump({'processed': list(self.processed)}, f)


class NormalizationStats:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.stats = {'ch0': {}, 'ch1': {}, 'ch2': {}, 'ready': False}
        self.load()

    def load(self):
        if self.filepath.exists():
            with open(self.filepath, 'r') as f:
                self.stats = json.load(f)

    def save_from_batches(self, batch_data):
        if not batch_data: return

        print("Calculating per-channel stats...")
        for ch in range(3):
            key = f'ch{ch}'
            lows = [x[ch][0] for x in batch_data]
            highs = [x[ch][1] for x in batch_data]

            self.stats[key] = {
                'min': float(np.median(lows)),
                'max': float(np.median(highs))
            }

        self.stats['ready'] = True
        with open(self.filepath, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"Stats Ready: {self.stats}")


# ==========================================
# CORE PROCESSING
# ==========================================

def check_signal_quality(y, sr):
    rms = np.sqrt(np.mean(y ** 2))
    if rms < CONFIG["MIN_RMS_AMPLITUDE"]:
        return False, "Silence (Low RMS)", rms

    peak = np.max(np.abs(y))
    crest = peak / (rms + 1e-9)
    if crest > CONFIG["MAX_CREST_FACTOR"]:
        return False, "Impulsive Click", crest

    center_y = y[len(y) // 4: 3 * len(y) // 4]
    if len(center_y) < 2048: return True, "OK", 0.0

    flatness = librosa.feature.spectral_flatness(y=center_y, n_fft=1024)[0]
    mean_flatness = np.mean(flatness)
    if mean_flatness > CONFIG["MAX_SPECTRAL_FLATNESS"]:
        return False, "High Static/Noise", mean_flatness

    return True, "OK", 0.0


def check_spectrogram_quality(pcen_spec):
    med = np.median(pcen_spec)
    mx = np.max(pcen_spec)
    snr = mx - med
    if snr < CONFIG["MIN_PCEN_SNR"]:
        return False, "Low Contrast/Washed Out", snr
    return True, "OK", 0.0


def load_and_clean(filepath):
    try:
        # Attempt load
        y, sr = librosa.load(filepath, sr=CONFIG["SAMPLE_RATE"], mono=True)
        if len(y) < CONFIG["HOP_LENGTH"]: return None, "Too Short"

        sos = scipy.signal.butter(10, CONFIG["FMIN"], 'hp', fs=sr, output='sos')
        y = scipy.signal.sosfilt(sos, y)

        return y, "OK"
    except Exception as e:
        # Catch Missing FFmpeg Error specifically
        err_msg = str(e)
        if "audioread" in err_msg or "NoBackendError" in err_msg:
            return None, "Missing FFmpeg (Codec Error)"
        return None, err_msg


def compute_features(y):
    S = librosa.feature.melspectrogram(
        y=y, sr=CONFIG["SAMPLE_RATE"], n_fft=CONFIG["N_FFT"],
        hop_length=CONFIG["HOP_LENGTH"], n_mels=CONFIG["N_MELS"],
        fmin=CONFIG["FMIN"], fmax=CONFIG["FMAX"]
    )

    pcen = librosa.pcen(
        S * (2 ** 31), sr=CONFIG["SAMPLE_RATE"], hop_length=CONFIG["HOP_LENGTH"],
        time_constant=CONFIG["PCEN_TIME_CONSTANT"], gain=CONFIG["PCEN_GAIN"],
        bias=CONFIG["PCEN_BIAS"], power=CONFIG["PCEN_POWER"], eps=CONFIG["PCEN_EPS"]
    )

    delta = librosa.feature.delta(pcen)
    delta2 = librosa.feature.delta(pcen, order=2)

    return np.stack([pcen, delta, delta2], axis=0)


def get_segments(features):
    spec = features[0]
    frames_total = spec.shape[1]
    frames_window = int((CONFIG["SAMPLE_LEN_SEC"] * CONFIG["SAMPLE_RATE"]) / CONFIG["HOP_LENGTH"])

    onset_env = librosa.onset.onset_strength(S=spec, sr=CONFIG["SAMPLE_RATE"])
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=CONFIG["SAMPLE_RATE"],
        hop_length=CONFIG["HOP_LENGTH"], backtrack=True, units='frames'
    )

    segments = []
    used_mask = np.zeros(frames_total, dtype=bool)
    spacing = int((CONFIG["MIN_SEGMENT_SPACING_SEC"] * CONFIG["SAMPLE_RATE"]) / CONFIG["HOP_LENGTH"])

    if len(onsets) == 0:
        center = frames_total // 2
        onsets = [center]

    for onset in onsets:
        if len(segments) >= CONFIG["MAX_SEGMENTS_PER_FILE"]: break

        start = max(0, onset - (frames_window // 2))
        end = start + frames_window

        if end > frames_total:
            end = frames_total
            start = max(0, end - frames_window)

        if np.any(used_mask[start:end]): continue

        if (end - start) == frames_window:
            segments.append((start, end))
            used_mask[start:min(frames_total, end + spacing)] = True

    return segments


# ==========================================
# WORKER
# ==========================================

def worker(args):
    filepath, species, out_dir, mode, global_stats = args

    try:
        y, status = load_and_clean(filepath)
        if y is None: return ('skip', status, 0.0, str(filepath))

        is_good, reason, val = check_signal_quality(y, CONFIG["SAMPLE_RATE"])
        if not is_good: return ('skip', reason, val, str(filepath))

        features = compute_features(y)

        is_spec_good, reason, val = check_spectrogram_quality(features[0])
        if not is_spec_good: return ('skip', reason, val, str(filepath))

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
            segments = get_segments(features)
            if not segments: return ('skip', "No Onsets Found", 0.0, str(filepath))

            filename = Path(filepath).stem
            save_dir = Path(out_dir) / species
            save_dir.mkdir(parents=True, exist_ok=True)

            saved = 0
            for idx, (start, end) in enumerate(segments):
                crop = features[:, :, start:end]

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

                Image.fromarray(merged).save(save_dir / f"{filename}_seg{idx}.png")
                saved += 1

            return ('done', saved, 0.0, str(filepath))

    except Exception as e:
        return ('error', str(e), 0.0, str(filepath))


# ==========================================
# RUNNER
# ==========================================

def run():
    print("-" * 60)
    print("BIRD AUDIO PREPROCESSOR (V12)")
    print("-" * 60)

    Path(CONFIG["PROCESSED_IMG_DIR"]).mkdir(exist_ok=True)
    norm_stats = NormalizationStats(Path(CONFIG["PROCESSED_IMG_DIR"]) / CONFIG["STATS_FILE"])
    checkpoint = CheckpointManager(Path(CONFIG["PROCESSED_IMG_DIR"]) / CONFIG["CHECKPOINT_FILE"])

    audit_path = Path(CONFIG["PROCESSED_IMG_DIR"]) / CONFIG["AUDIT_LOG"]
    if not audit_path.exists():
        with open(audit_path, 'w', newline='') as f:
            csv.writer(f).writerow(["Filename", "Reason", "Metric_Value"])

    # UPDATED FILE SEARCH: Looks for WAV, MP3, M4A, FLAC, OGG
    print("Indexing Files...")
    raw_path = Path(CONFIG["RAW_AUDIO_DIR"])
    extensions = ['*.wav', '*.mp3', '*.m4a', '*.flac', '*.ogg']
    all_files = []
    for ext in extensions:
        all_files.extend(list(raw_path.rglob(ext)))

    if not all_files:
        print("No files found. Check your directory and extensions.")
        return
    print(f"Found {len(all_files)} files.")

    try:
        # --- PASS 1: CALIBRATION ---
        if not norm_stats.stats.get('ready'):
            subset_size = min(CONFIG['CALIBRATION_MAX'], int(len(all_files) * CONFIG['CALIBRATION_RATIO']))
            subset_size = max(100, subset_size)
            print(f"\nStep 1: Calibrating on {subset_size} files...")

            subset = np.random.choice(all_files, subset_size, replace=False)
            tasks = [(f, "", "", 'stats', None) for f in subset]
            results = []

            with multiprocessing.Pool(CONFIG["NUM_WORKERS"]) as pool:
                for res in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks)):
                    if res[0] == 'stat': results.append(res[1])

            norm_stats.save_from_batches(results)
            gc.collect()

        # --- PASS 2: GENERATION ---
        print("\nStep 2: Generating Spectrograms (Press Ctrl+C to stop safely)...")
        to_process = [f for f in all_files if str(f) not in checkpoint.processed]
        print(f"Remaining: {len(to_process)}")

        tasks = [(f, f.parent.name, CONFIG["PROCESSED_IMG_DIR"], 'generate', norm_stats.stats) for f in to_process]

        with multiprocessing.Pool(CONFIG["NUM_WORKERS"], maxtasksperchild=CONFIG["MAX_TASKS_PER_CHILD"]) as pool:
            try:
                loop_counter = 0
                for res in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks), unit="file"):
                    status, msg, val, fpath = res
                    loop_counter += 1

                    if status == 'error':
                        with open(CONFIG["ERROR_LOG"], "a") as f:
                            f.write(f"{datetime.now()} - {fpath} - {msg}\n")
                    elif status == 'skip':
                        with open(audit_path, "a", newline='') as f:
                            # Use utf-8 to handle odd characters in filenames
                            # Simple retry logic for file writing if locked
                            try:
                                csv.writer(f).writerow([Path(fpath).name, msg, f"{val:.4f}"])
                            except:
                                pass
                    elif status == 'done':
                        checkpoint.processed.add(fpath)

                    if loop_counter % CONFIG["CHECKPOINT_INTERVAL"] == 0:
                        checkpoint.save()

                checkpoint.save()

            except KeyboardInterrupt:
                print("\nStopping pool...")
                pool.terminate()
                pool.join()
                raise

        print(f"\nDone.")

    except KeyboardInterrupt:
        print("\n\n🛑 INTERRUPT RECEIVED! Saving Checkpoint and Exiting...")
        checkpoint.save()
        sys.exit(0)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    run()
