import sys
import os
import torch
import numpy as np

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from inference import BirdClassifier

def test_classifier():
    print("Initializing classifier...")
    classifier = BirdClassifier()
    
    print("Loading resources...")
    success = classifier.load_resources()
    if not success:
        print("FAILED: Could not load resources.")
        sys.exit(1)
    
    print("Resources loaded.")
    
    # Check model architecture
    if classifier.config["MODEL_NAME"] != 'efficientnet_b4':
        print(f"FAILED: Incorrect model name in config: {classifier.config['MODEL_NAME']}")
        sys.exit(1)
        
    # Check class mapping
    if not classifier.idx_to_class:
        print("FAILED: Class mapping is empty.")
        sys.exit(1)
    
    print(f"Class mapping loaded with {len(classifier.idx_to_class)} classes.")
    
    # Dummy inference
    print("Running dummy inference...")
    # Generate 3 seconds of white noise at 32kHz
    sr = 32000
    duration = 3.0
    audio = np.random.uniform(-1, 1, int(sr * duration)).astype(np.float32)
    
    # Convert to bytes (simulate file read)
    import io
    import soundfile as sf
    
    with io.BytesIO() as bio:
        sf.write(bio, audio, sr, format='WAV')
        audio_bytes = bio.getvalue()
        
    results, image = classifier.predict(audio_bytes, filename="test.wav")
    
    if results is None:
        print("FAILED: Prediction returned None.")
        sys.exit(1)
        
    print("Prediction successful!")
    print("Top result:", results[0])
    print("Image size:", image.size)
    
    if image.size != (384, 384):
        print(f"FAILED: Incorrect image size: {image.size}")
        sys.exit(1)

    print("ALL TESTS PASSED")

if __name__ == "__main__":
    test_classifier()
