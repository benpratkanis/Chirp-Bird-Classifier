import torch
import os

CLASS_MAPPING_PATH = r"d:\ChirpBirdClassifier\ChirpPlatform\models\class_mapping.pth"

if os.path.exists(CLASS_MAPPING_PATH):
    idx_to_class = torch.load(CLASS_MAPPING_PATH, weights_only=False)
    print("Classes found:")
    for idx, name in idx_to_class.items():
        print(f"{idx}: {name}")
else:
    print("Class mapping file not found.")
