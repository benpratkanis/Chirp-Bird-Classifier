import os
import shutil
import random
import matplotlib.pyplot as plt
import argparse

# --- CONFIGURATION ---
# You can change these defaults or input them when the script runs
DEFAULT_SOURCE_DIR = r"D:\ChirpBirdClassifier\Spectrograms_HighRes_384_Mem"
DEFAULT_DEST_DIR = r"D:\ChirpBirdClassifier\SpectrogramsHighResBalanced"
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def scan_dataset(source_dir):
    """
    Scans the source directory and returns a dictionary:
    { 'SpeciesName': ['file1.png', 'file2.png', ...] }
    """
    print(f"Scanning {source_dir}...")
    dataset = {}

    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' not found.")
        return None

    try:
        # Get all subfolders (species)
        species_list = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

        for species in species_list:
            species_path = os.path.join(source_dir, species)
            # Get all valid image files
            files = [f for f in os.listdir(species_path) if f.lower().endswith(VALID_EXTENSIONS)]
            if files:
                dataset[species] = files

        return dataset
    except Exception as e:
        print(f"An error occurred during scanning: {e}")
        return None


def analyze_distribution(dataset):
    """
    Prints text statistics and generates a bar chart.
    """
    print("\n--- DATASET STATISTICS ---")
    counts = {species: len(files) for species, files in dataset.items()}
    sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1]))

    total_files = sum(counts.values())
    total_species = len(counts)
    min_files = min(counts.values())
    max_files = max(counts.values())
    avg_files = total_files / total_species

    print(f"Total Species: {total_species}")
    print(f"Total Images:  {total_files}")
    print(f"Min Images:    {min_files} ({list(sorted_counts.keys())[0]})")
    print(f"Max Images:    {max_files} ({list(sorted_counts.keys())[-1]})")
    print(f"Average:       {avg_files:.2f}")
    print("-" * 30)

    # Show individual counts (limit to top/bottom 5 if list is huge, or show all)
    print("Distribution per species:")
    for species, count in sorted_counts.items():
        print(f"{species}: {count}")

    # Visualization
    names = list(sorted_counts.keys())
    values = list(sorted_counts.values())

    plt.figure(figsize=(12, 6))
    plt.bar(names, values, color='skyblue')
    plt.xlabel('Species')
    plt.ylabel('Number of Spectrograms')
    plt.title('Data Distribution by Species')
    plt.xticks(rotation=90)  # Rotate names so they fit
    plt.tight_layout()

    print("\nDisplaying graph... (Close the graph window to continue)")
    plt.show()

    return sorted_counts


def balance_and_copy(dataset, source_dir, dest_dir):
    """
    Interactive function to determine thresholds and copy files.
    """
    print("\n--- BALANCING STRATEGY ---")
    print("We need to decide which species to drop and how many images to keep per species.")

    # 1. Threshold Input
    while True:
        try:
            min_limit = int(input(
                "STEP 1: What is the MINIMUM number of unique images required to keep a species? (Enter 0 to keep all): "))
            break
        except ValueError:
            print("Please enter a valid integer.")

    # Filter dataset based on threshold
    qualified_species = {s: f for s, f in dataset.items() if len(f) >= min_limit}
    dropped_species = [s for s in dataset if s not in qualified_species]

    print(f"\nBased on minimum limit of {min_limit}:")
    print(f"Keeping: {len(qualified_species)} species")
    print(f"Dropping: {len(dropped_species)} species ({', '.join(dropped_species)})")

    if len(qualified_species) == 0:
        print("No species meet the criteria. Exiting.")
        return

    # 2. Target Count Input
    while True:
        try:
            target_input = input(
                f"STEP 2: How many images do you want per species? (Enter 'min' for {min([len(f) for f in qualified_species.values()])}, or a specific number): ")
            if target_input.lower() == 'min':
                target_count = min([len(f) for f in qualified_species.values()])
                break
            else:
                target_count = int(target_input)
                if target_count > 0:
                    break
                print("Number must be greater than 0.")
        except ValueError:
            print("Please enter a valid integer or 'min'.")

    # 3. Upsampling Choice
    upsample = False
    max_avail = max([len(f) for f in qualified_species.values()])
    if target_count > min([len(f) for f in qualified_species.values()]):
        print(f"\nSome species have fewer than {target_count} images.")
        print("Upsampling will duplicate images in small folders to reach the target count.")
        upsample_input = input("Do you want to enable Upsampling? (y/n): ")
        upsample = upsample_input.lower() == 'y'

    # 4. Execute
    print(f"\n--- CREATING NEW DATASET ---")
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    print(f"Strategy: {target_count} images per species.")
    print(f"Upsampling: {'Enabled (Duplicates allowed)' if upsample else 'Disabled (Capped at max available)'}")

    confirm = input("Proceed to generate files? (y/n): ")
    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return

    if os.path.exists(dest_dir):
        print(f"Warning: Destination folder '{dest_dir}' already exists.")
        overwrite = input("Do you want to assume it's empty and write into it? (y/n): ")
        if overwrite.lower() != 'y':
            return

    for species, files in qualified_species.items():
        # Create species folder in destination
        dest_species_path = os.path.join(dest_dir, species)
        os.makedirs(dest_species_path, exist_ok=True)

        # Determine files to copy
        current_count = len(files)
        selected_files = []

        if current_count >= target_count:
            # Undersample: Randomly pick unique files
            selected_files = random.sample(files, target_count)
        else:
            if upsample:
                # Upsample: Take all originals + random duplicates
                selected_files.extend(files)  # Take all originals
                needed = target_count - current_count
                selected_files.extend(random.choices(files, k=needed))  # Random duplicates
            else:
                # Cap at max available (no upsampling)
                selected_files = files[:]

        print(f"Processing {species}: Copying {len(selected_files)} images (Originals: {current_count})...")

        src_species_path = os.path.join(source_dir, species)

        # Use a dictionary to track how many times a filename has been used to handle naming duplicates
        name_tracker = {}

        for filename in selected_files:
            name_tracker[filename] = name_tracker.get(filename, 0) + 1

            src_file = os.path.join(src_species_path, filename)

            # If it's the first time seeing this file, keep original name
            if name_tracker[filename] == 1:
                dest_filename = filename
            else:
                # If it's a duplicate, append a counter
                root, ext = os.path.splitext(filename)
                dest_filename = f"{root}_copy_{name_tracker[filename]}{ext}"

            dest_file = os.path.join(dest_species_path, dest_filename)
            shutil.copy2(src_file, dest_file)

    print("\nDone! Your balanced dataset is ready.")


if __name__ == "__main__":
    # 1. Setup Paths
    print("Bird Data Balancer")
    print("------------------")
    src = input(f"Enter source directory (Press Enter for default: {DEFAULT_SOURCE_DIR}): ").strip()
    if not src: src = DEFAULT_SOURCE_DIR

    # 2. Scan
    data = scan_dataset(src)

    if data:
        # 3. Analyze & Visualize
        analyze_distribution(data)

        # 4. Balance
        dst = input(f"Enter destination directory (Press Enter for default: {DEFAULT_DEST_DIR}): ").strip()
        if not dst: dst = DEFAULT_DEST_DIR

        balance_and_copy(data, src, dst)