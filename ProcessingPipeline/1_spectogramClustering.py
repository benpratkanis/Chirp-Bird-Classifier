"""
AI-Powered Spectrogram Cleanup Script (02b_cluster_cleanup_pytorch.py)

This script uses unsupervised clustering (K-Means) to find and remove "junk" samples
(like static or electronic hums) from the processed spectrogram dataset.

This version is built entirely on the PyTorch stack. It uses a pre-trained
ResNet-18 model from torchvision as a fast, intelligent feature extractor.

It works in two modes:
1. "review": Analyzes all images, groups them into clusters, and saves
   a few samples from each cluster for you to manually review.
2. "delete": Deletes all images that belong to the clusters you identify
   as "bad" (junk).

WORKFLOW:
1. Run once in "review" mode.
2. Manually check the folders in "cleanup_review".
3. Write down the numbers of the "junk" clusters (e.g., [7, 23, 41]).
4. Change the "BAD_CLUSTER_IDS" list to your junk cluster numbers.
5. Change MODE to "delete" and run again to perform the cleanup.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import joblib
from sklearn.cluster import KMeans
import shutil
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---

CONFIG = {
    # --- Mode Settings ---
    # "review":  Analyzes and saves sample clusters for you to look at.
    # "delete":  Deletes all files from the clusters listed in BAD_CLUSTER_IDS.
    "MODE": "review",  # <-- START WITH "review"
    
    # --- Paths ---
    "PROCESSED_IMG_DIR": r"D:\ChirpBirdClassifier\Spectrograms_HighRes_384_Mem",
    "REVIEW_OUTPUT_DIR": r"D:\ChirpBirdClassifier\SpectrogramsReview", # Stores sample images
    "CLUSTERING_DATA_FILE": r"D:\ChirpBirdClassifier\cluster_data_pytorch.pkl", # Caches AI results - CANNOT BE SAME AS REVIEW OUTPUT - Trial by error :( / Must have "\cluster_data_pytorch.pkl"
    
    # --- AI Clustering Settings ---
    "N_CLUSTERS": 200,         # <-- ***INCREASED TO 150***
                               # This forces the AI to be "pickier" and create
                               # smaller, purer clusters. This is the key to
                               # separating junk files from good files.
    "SAMPLES_PER_CLUSTER": 30, # How many samples to save for manual review.
    "BATCH_SIZE": 64,          # Batch size for feature extraction (faster on GPU)
    
    # --- Image Settings (Must match preprocesser output) ---
    # We will resize to 256x256 to fit the standard ResNet input
    "IMG_SIZE": 384,
    
    # --- !!! DELETE MODE SETTING !!! ---
    # AFTER REVIEW, set MODE="delete" and list the junk cluster numbers here.
    "BAD_CLUSTER_IDS": [] # Example: [10, 35, 42] - START THIS LIST EMPTY
}

# --- END OF CONFIGURATION ---

# Check for CUDA availability
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def get_feature_extractor():
    """
    Loads the ResNet-18 model pre-trained on ImageNet and removes its top
    classification layer. This will be our "feature extractor".
    """
    print("Loading pre-trained PyTorch model (ResNet-18)...")
    
    # Load the base model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # "Chop off" the final classification layer (fc) by replacing it
    # with an identity layer that just passes the features through.
    model.fc = nn.Identity()
    
    model = model.to(DEVICE)
    model.eval() # Set model to evaluation mode (disables dropout, etc.)
    return model


# Define the image transformations
# These MUST match the validation transforms from your training script
preprocess_transform = transforms.Compose([
    transforms.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ImageDataset(Dataset):
    """A simple PyTorch Dataset to load the image paths."""
    def __init__(self, filepaths, transform=None):
        self.filepaths = filepaths
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"  [Error] Failed to load image {img_path}: {e}")
            # Return a blank tensor if image is corrupt
            return torch.zeros((3, CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]))


def extract_features(model, all_filepaths):
    """
    Processes all images through the CNN and returns a giant list
    of feature vectors.
    """
    print(f"Extracting features from {len(all_filepaths)} images...")
    
    dataset = ImageDataset(all_filepaths, transform=preprocess_transform)
    # Use DataLoader for batching (much faster on GPU)
    loader = DataLoader(
        dataset, 
        batch_size=CONFIG["BATCH_SIZE"], 
        shuffle=False, 
        num_workers=4,  # Use multiple CPU cores to load data
        pin_memory=True
    )
    
    all_features = []
    
    # Run feature extraction with torch.no_grad() for efficiency
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features", unit="batch"):
            batch = batch.to(DEVICE)
            
            # Get the feature vectors (e.g., [64, 512] for ResNet18)
            features = model(batch)
            
            # Move features to CPU and store as numpy arrays
            all_features.append(features.cpu().numpy())
            
    # Concatenate all batches into one giant numpy array
    all_features = np.concatenate(all_features, axis=0)
    return all_features, all_filepaths


def cluster_features(features):
    """
    Runs K-Means clustering on the extracted features to group
    similar images together. (This part is identical to the TF version)
    """
    n_clusters = CONFIG["N_CLUSTERS"]
    print(f"Clustering {len(features)} images into {n_clusters} groups...")
    
    # Run K-Means
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10, # Explicitly set n_init
        verbose=1  # Show progress
    )
    cluster_labels = kmeans.fit_predict(features)
    
    print("Clustering complete.")
    return cluster_labels


def review_clusters(all_filepaths, cluster_labels):
    """
    REVIEW MODE: Saves sample images from each cluster into a
    review directory for manual inspection.
    """
    print("Saving sample images for review...")
    review_dir = Path(CONFIG["REVIEW_OUTPUT_DIR"])
    
    # Clean up old review folders
    if review_dir.exists():
        shutil.rmtree(review_dir)
    
    # Create a dictionary to hold filepaths for each cluster
    clusters = {i: [] for i in range(CONFIG["N_CLUSTERS"])}
    for filepath, label in zip(all_filepaths, cluster_labels):
        clusters[label].append(filepath)

    # Save samples
    for cluster_id, files in clusters.items():
        cluster_folder = review_dir / f"cluster_{cluster_id:02d}"
        cluster_folder.mkdir(parents=True, exist_ok=True)
        
        # Get up to SAMPLES_PER_CLUSTER random samples
        sample_files = np.random.choice(
            files, 
            min(len(files), CONFIG["SAMPLES_PER_CLUSTER"]), 
            replace=False
        )
        
        # Copy the sample files
        for f in sample_files:
            shutil.copy(f, cluster_folder)
            
    print(f"--- Review folders have been created in: {review_dir} ---")
    print("Please inspect these folders and identify the 'junk' cluster IDs.")


def delete_clusters(all_filepaths, cluster_labels):
    """
    DELETE MODE: Deletes all files belonging to the clusters
    listed in CONFIG["BAD_CLUSTER_IDS"].
    """
    bad_ids = CONFIG["BAD_CLUSTER_IDS"]
    if not bad_ids:
        print("No BAD_CLUSTER_IDS specified. Nothing to delete.")
        return

    print(f"--- DELETE MODE ACTIVATED ---")
    print(f"Will delete all files from clusters: {bad_ids}")
    
    deleted_count = 0
    total_count = 0
    
    # Iterate and delete
    for filepath, label in tqdm(zip(all_filepaths, cluster_labels), total=len(all_filepaths), desc="Deleting files"):
        if label in bad_ids:
            try:
                os.remove(filepath)
                deleted_count += 1
            except Exception as e:
                print(f"  [Error] Could not delete {filepath}: {e}")
        total_count += 1
        
    print("\n--- Deletion Complete! ---")
    print(f"Total images inspected: {total_count}")
    print(f"Total junk images deleted: {deleted_count}")
    print(f"Remaining clean images: {total_count - deleted_count}")


def main():
    """
    Main function to run the clustering and cleanup.
    """
    
    # --- 1. Gather all image filepaths ---
    img_dir = Path(CONFIG["PROCESSED_IMG_DIR"])
    all_filepaths = [str(p) for p in img_dir.glob("*/*.png")]
    
    if not all_filepaths:
        print(f"Error: No .png files found in {img_dir}.")
        print("Please run 01_process_audio.py first.")
        return
        
    print(f"Found {len(all_filepaths)} images to analyze.")

    # --- 2. Load or Run Clustering ---
    cluster_data_file = Path(CONFIG["CLUSTERING_DATA_FILE"])
    
    # Check if we have already run the (slow) AI part
    if cluster_data_file.exists() and CONFIG["MODE"] == "delete":
        # Only load from cache if we are in delete mode
        print(f"Loading cached clustering data from {cluster_data_file}...")
        data = joblib.load(cluster_data_file)
        processed_filepaths = data["filepaths"]
        cluster_labels = data["labels"]
        
        # Simple check
        if len(processed_filepaths) != len(all_filepaths):
            print("Warning: File list has changed. Re-running clustering...")
            if cluster_data_file.exists():
                cluster_data_file.unlink() # Delete old data
            main() # Restart the process
            return
            
    else:
        # No cached data, or in "review" mode
        if CONFIG["MODE"] == "review" and cluster_data_file.exists():
            print("Review mode: Deleting old cache to re-run clustering...")
            cluster_data_file.unlink()
            
        print("Starting full feature extraction and clustering...")
        
        # 1. Load the AI model
        model = get_feature_extractor()
        
        # 2. Extract features (This is the SLOW part)
        features, processed_filepaths = extract_features(model, all_filepaths)
        
        # 3. Cluster the features
        cluster_labels = cluster_features(features)
        
        # 4. Save the results to speed up next time
        print(f"Saving clustering results to {cluster_data_file}...")
        cluster_data_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"filepaths": processed_filepaths, "labels": cluster_labels}, 
            cluster_data_file
        )

    # --- 3. Run the selected MODE ---
    if CONFIG["MODE"] == "review":
        review_clusters(processed_filepaths, cluster_labels)
    elif CONFIG["MODE"] == "delete":
        delete_clusters(processed_filepaths, cluster_labels)
    else:
        print(f"Error: Unknown MODE '{CONFIG['MODE']}'. Set to 'review' or 'delete'.")


if __name__ == "__main__":
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    main()