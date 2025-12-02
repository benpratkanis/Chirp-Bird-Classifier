# 🐦 Chirp: Bird Call Classification

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c?style=for-the-badge&logo=pytorch)
![Librosa](https://img.shields.io/badge/Audio-Librosa-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

**Chirp** is a deep learning pipeline designed to detect and classify bird species from raw audio field recordings. Unlike standard classifiers that crush audio into low-resolution images, Chirp utilizes **high-resolution 384px square spectrograms**, **PCEN adaptive gain**, and **EfficientNet-B4** architecture to capture the subtle temporal details of bird vocalizations.

---

## 🎧 The Problem
Bird calls are complex, rapid-fire acoustic events. Traditional AI approaches often:
1.  **Distort Time:** Squishing 5 seconds of audio into a 224px square blurs rapid trills.
2.  **Train on Noise:** Standard onset detection often captures wind, rain, or static.
3.  **Lose Detail:** Low-resolution Mel Spectrograms miss harmonic stacks.

## ⚡ The Solution: Our Methodology

### 1. The "RGB" Audio Engine
We don't just use grayscale spectrograms. We treat audio like color images to give the model more context:
* **Red Channel (Energy):** **PCEN** (Per-Channel Energy Normalization) is used instead of standard Log-Mel. This suppresses constant background noise (wind/insects) while highlighting rapid transient events (chirps).
* **Green Channel (Velocity):** Delta features (first derivative of energy).
* **Blue Channel (Acceleration):** Delta-Delta features (second derivative).

### 2. Physics-Based Segmentation
* **Square Inputs:** We process audio in **3.0-second chunks**. At our sample rate and hop length, this produces a mathematically perfect **384x384** square image. This ensures **zero aspect ratio distortion** when feeding the CNN.
* **Band-Limited Scanning:** The preprocessor ignores low-frequency rumble (wind) and high-frequency hiss, triggering only on energy in the **1kHz - 12kHz** "Bird Vocalization Band."
* **Failsafe Extraction:** Includes logic to rescue faint calls from quiet recordings by analyzing the noise floor relative to local peaks.

### 3. Model Architecture
* **Backbone:** **EfficientNet-B4** (Pretrained on ImageNet). Selected for its ability to handle higher resolution textures ($384 \times 384$).
* **Regularization Strategy:**
    * **MixUp:** Blends images and labels (e.g., 40% Robin + 60% Sparrow) to force the model to learn features rather than memorizing training data.
    * **SpecAugment:** Randomly masks vertical (time) and horizontal (frequency) strips during training to improve robustness against signal loss.
    * **Label Smoothing:** Prevents the model from becoming over-confident in its predictions.

---

## 📊 Visual Pipeline

```mermaid
graph LR
A[Raw Audio .WAV] --> B(Band-Pass Filter)
B --> C{Bird Activity Detected?}
C -- Yes --> D[Generate 384x384 Spectrogram]
C -- No --> E[Skip / Log]
D --> F[PCEN + Deltas Encoding]
F --> G[EfficientNet-B4]
G --> H[Species Prediction]
