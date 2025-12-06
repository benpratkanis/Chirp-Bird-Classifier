import os
import multiprocessing  # Added missing import
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
# The folder containing your current heavy PNGs/images
SOURCE_DIR = r"D:\ChirpBirdClassifier\SpectrogramsHighResBalanced"

# The new folder where the small JPGs will be saved
DEST_DIR = r"D:\ChirpBirdClassifier\Spectrograms_384_JPG"

# Target size for the A100 training script (384x384 is standard for EfficientNetB4)
TARGET_SIZE = 384

# Quality 85 is usually indistinguishable from PNG for ML, but 10x smaller
JPEG_QUALITY = 85


# ==========================================
# PROCESSING FUNCTION
# ==========================================
def process_file(args):
    """
    Worker function to process a single image.
    Args: (source_path, dest_path)
    """
    src_path, dst_path = args

    try:
        # Skip if file already exists (allows you to resume if script crashes)
        if os.path.exists(dst_path):
            return

        with Image.open(src_path) as img:
            # Convert to RGB (handles PNG transparency or Grayscale issues)
            img = img.convert('RGB')

            # High-quality resize
            img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)

            # Save as optimized JPEG
            img.save(dst_path, "JPEG", quality=JPEG_QUALITY, optimize=True)

    except Exception as e:
        print(f"Error processing {src_path}: {e}")


# ==========================================
# MAIN SCRIPT
# ==========================================
def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory not found: {SOURCE_DIR}")
        return

    print(f"Scanning files in '{SOURCE_DIR}'...")

    tasks = []

    # 1. Walk through source directory to build task list and replicate folders
    for root, dirs, files in os.walk(SOURCE_DIR):
        # Calculate relative path (e.g., "Robin")
        rel_path = os.path.relpath(root, SOURCE_DIR)

        # Create corresponding destination folder
        dest_folder = os.path.join(DEST_DIR, rel_path)
        os.makedirs(dest_folder, exist_ok=True)

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                src_file_path = os.path.join(root, file)

                # Change extension to .jpg for the destination
                file_name_no_ext = os.path.splitext(file)[0]
                dest_file_path = os.path.join(dest_folder, file_name_no_ext + ".jpg")

                tasks.append((src_file_path, dest_file_path))

    total_files = len(tasks)
    print(f"Found {total_files} images.")
    print(f"Targeting: {TARGET_SIZE}x{TARGET_SIZE} px | JPG Quality: {JPEG_QUALITY}")
    print(f"Using {cpu_count()} CPU cores...")

    # 2. Process in parallel
    # chunksize helps reduce overhead for massive lists of small tasks
    chunk_size = max(1, total_files // (cpu_count() * 4))

    with Pool(cpu_count()) as p:
        # imap_unordered is generally faster than map for this type of work
        list(tqdm(p.imap_unordered(process_file, tasks, chunksize=chunk_size), total=total_files, unit="img"))

    print("\n------------------------------------------------")
    print("Processing Complete!")
    print(f"New dataset location: {DEST_DIR}")
    print("------------------------------------------------")


if __name__ == '__main__':
    # Windows requires this for multiprocessing
    multiprocessing.freeze_support() if hasattr(multiprocessing, 'freeze_support') else None
    main()