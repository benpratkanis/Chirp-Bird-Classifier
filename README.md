# 🐦 Chirp: Bird Call Classification

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c?style=for-the-badge&logo=pytorch)
![Librosa](https://img.shields.io/badge/Audio-Librosa-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

-----

**Chirp** is a deep learning pipeline designed to detect and classify bird species from raw audio field recordings. Unlike standard classifiers that crush audio into low-resolution images, Chirp utilizes **high-resolution 384px square spectrograms**, **PCEN adaptive gain**, and **EfficientNet-B4** architecture to capture the subtle temporal details of bird vocalizations.

Unlike standard classifiers that compress audio into low-resolution images, Chirp utilizes high-resolution spectrograms, PCEN adaptive gain, and the EfficientNet-B4 architecture to capture the subtle temporal details of bird vocalizations.

-----

## 🎧 The Problem

Bird calls are complex, rapid-fire acoustic events. Traditional AI approaches often fail to capture these nuances due to three main limitations:

  * **Time Distortion:** Squishing 5 seconds of audio into a small $224 \times 224$ square blurs rapid trills.
  * **Noise Training:** Standard onset detection often captures wind, rain, or static, polluting the dataset.
  * **Loss of Detail:** Low-resolution Mel Spectrograms miss harmonic stacks crucial for species identification.

-----

## ⚡ The Solution: Our Methodology

### 1\. The "RGB" Audio Engine

We don't just use grayscale spectrograms. We treat audio like color images to provide the model with depth and context:

  * 🔴 **Red Channel (Energy):** Uses **PCEN** (Per-Channel Energy Normalization) instead of standard Log-Mel. This suppresses constant background noise (wind/insects) while highlighting rapid transient events (chirps).
  * 🟢 **Green Channel (Velocity):** Delta features (the first derivative of energy).
  * 🔵 **Blue Channel (Acceleration):** Delta-Delta features (the second derivative).

### 2\. Physics-Based Segmentation

  * **Square Inputs:** We process audio in $3.0$-second chunks. At our specific sample rate and hop length, this produces a mathematically perfect $384 \times 384$ square image. This ensures **zero aspect ratio distortion** when feeding the CNN.
  * **Band-Limited Scanning:** The preprocessor ignores low-frequency rumble and high-frequency hiss, triggering only on energy in the **$1\text{kHz} - 12\text{kHz}$** "Bird Vocalization Band."
  * **Failsafe Extraction:** Includes logic to rescue faint calls from quiet recordings by analyzing the noise floor relative to local peaks.

### 3\. Model Architecture

[Image of EfficientNet architecture]

  * **Backbone:** **EfficientNet-B4** (Pretrained on ImageNet). Selected for its ability to handle higher resolution textures ($384 \times 384$).
  * **Regularization Strategy:**
      * **MixUp:** Blends images and labels (e.g., $40\%$ Robin + $60\%$ Sparrow) to force the model to learn features rather than memorizing training data.
      * **SpecAugment:** Randomly masks vertical (time) and horizontal (frequency) strips during training to improve robustness against signal loss.
      * **Label Smoothing:** Prevents the model from becoming over-confident in its predictions.

-----

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
```

-----

## 🚀 Getting Started

Chirp is fully containerized. Whether you are developing features or deploying the model, we use Docker to manage the React frontend, FastAPI backend, and dependencies.

### Prerequisites

  * Docker Desktop installed and running.
  * Git

### 1\. Clone the Repository

```bash
git clone https://github.com/benpratkanis/Chirp-Bird-Classifier.git
cd chirp
```

### 2\. 🛠️ Run in Development Mode

Use this mode for active coding. It enables Hot Module Replacement (HMR) for the React frontend and auto-reloading for the FastAPI backend.

**Start the stack:**

```bash
docker-compose up --build
```

**Access the application:**

  * **Frontend (Vite):** `http://localhost:5173`
  * **Backend Docs (Swagger UI):** `http://localhost:8000/docs`

**Stop the stack:**
Press `Ctrl+C` in your terminal.

### 3\. 🐳 Run in Production Mode

Use this mode to simulate the final deployment environment. This uses `docker-compose.prod.yml` to serve the frontend via a static server (e.g., Nginx) and runs the backend with production workers (Uvicorn).

**Start the stack (Detached mode):**

```bash
docker-compose -f docker-compose.prod.yml up -d --build
```

**Access the application:**

  * **Frontend:** `http://localhost` (Served on port 80)

**Stop and remove containers:**

```bash
docker-compose -f docker-compose.prod.yml down
```

-----
