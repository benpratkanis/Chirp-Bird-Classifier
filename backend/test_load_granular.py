import sys
import os
import torch
import numpy as np
import time

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
    
    # Test Model Direct Inference
    print("Testing model forward pass with random tensor...")
    try:
        dummy_input = torch.randn(1, 3, 384, 384).to(classifier.config["DEVICE"])
        with torch.no_grad():
            output = classifier.model(dummy_input)
        print("Model forward pass successful. Output shape:", output.shape)
    except Exception as e:
        print(f"FAILED: Model forward pass error: {e}")
        sys.exit(1)

    # Test Audio Processing
    print("Testing audio processing...")
    try:
        # Generate 3 seconds of white noise
        sr = 32000
        duration = 3.0
        audio = np.random.uniform(-1, 1, int(sr * duration)).astype(np.float32)
        
        # Convert to bytes
        import io
        import soundfile as sf
        with io.BytesIO() as bio:
            sf.write(bio, audio, sr, format='WAV')
            audio_bytes = bio.getvalue()
            
        print("Calling process_audio_to_image...")
        start_time = time.time()
        image = classifier.process_audio_to_image(audio_bytes, filename="test.wav")
        end_time = time.time()
        
        if image is None:
            print("FAILED: process_audio_to_image returned None.")
            sys.exit(1)
            
        print(f"Audio processing successful in {end_time - start_time:.2f}s. Image size: {image.size}")
        
    except Exception as e:
        print(f"FAILED: Audio processing error: {e}")
        sys.exit(1)

    print("ALL TESTS PASSED")

if __name__ == "__main__":
    test_classifier()
