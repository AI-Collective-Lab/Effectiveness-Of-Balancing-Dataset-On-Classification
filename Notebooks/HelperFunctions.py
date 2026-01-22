import os
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from tqdm import tqdm
def analyze_split(split_name, split_path):
    print(f"\n==============================================")
    print(f"       ANALYSIS FOR: {split_name} SET")
    print(f"==============================================")

    class_names = []
    image_counts = []

    # Iterate through the class folders within the split (e.g., 'train/Apple___Apple_scab')
    for class_name in os.listdir(split_path):
        class_folder_path = os.path.join(split_path, class_name)

        if os.path.isdir(class_folder_path):
            # Count the number of files (images) in the folder
            # We assume files are images and not other hidden files
            count = len([name for name in os.listdir(class_folder_path) if
                         os.path.isfile(os.path.join(class_folder_path, name))])

            # Store the results
            class_names.append(class_name)
            image_counts.append(count)

    # --- Create DataFrame ---
    df = pd.DataFrame({
        'Class Name': class_names,
        'Image Count': image_counts
    })

    total_images = df['Image Count'].sum()
    if total_images == 0:
        print(f"No images found in the {split_name} set at: {split_path}")
        return

    df['Percentage'] = (df['Image Count'] / total_images) * 100
    df = df.sort_values(by='Image Count', ascending=False).reset_index(drop=True)

    # --- Display Results ---
    print(f"Total images in {split_name} set: {total_images}")
    #print(df)

    # --- Imbalance Check ---
    min_count = df['Image Count'].min()
    max_count = df['Image Count'].max()

    # Calculate the ideal count for a perfectly balanced set
    ideal_count = total_images / len(df)

    # Simple check: are any classes less than 50% of the average size?
    is_imbalanced = (min_count < (0.5 * ideal_count))

    print("\n--- Imbalance Summary ---")
    print(f"Number of Classes: {len(df)}")
    print(f"Average Images per Class (Ideal): {ideal_count:.0f}")
    print(f"Smallest Class Size: {min_count} ({df['Class Name'].iloc[-1]})")
    print(f"Largest Class Size: {max_count} ({df['Class Name'].iloc[0]})")

    if is_imbalanced:
        print(f"\n⚠️ **Conclusion: The {split_name} set appears to be significantly imbalanced.**")
        print(f"The smallest class ({min_count}) is less than half the average count ({ideal_count:.0f}).")
    else:
        print(f"\n✅ **Conclusion: The {split_name} set appears to be reasonably balanced.**")

    return df,ideal_count,split_name,total_images


def visualize_split_results(df, ideal_count, split_name, total_images,show=False):
    # 1. Create specific figure and axes objects
    fig, ax = plt.subplots(figsize=(16, 6))

    # 2. Use the 'ax' object for plotting (cleaner and thread-safe)
    ax.bar(df['Class Name'], df['Image Count'], color='teal')
    ax.axhline(y=ideal_count, color='r', linestyle='--', label='Ideal Average Count')

    ax.set_xlabel("Disease/Condition Class")
    ax.set_ylabel("Number of Images")
    ax.set_title(f"Distribution of Images in the {split_name} Set (Total: {total_images})")

    # Set ticks on the axes object
    ax.set_xticks(range(len(df['Class Name'])))
    ax.set_xticklabels(df['Class Name'], rotation=90, fontsize=8)

    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)

    # 3. Return the FIGURE object, not the module
    return fig


# --- 3. FILTERING AND RELOCATION FUNCTION ---
def copy_valid_samples(dataset, split_name, class_names,clean_base_dir,class_idx_to_name):
    """Checks file existence and copies valid files to a new directory structure."""

    # Define the target split folder (e.g., 'Filtered_Plant_Dataset/train')
    target_split_dir = os.path.join(clean_base_dir, split_name)
    os.makedirs(target_split_dir, exist_ok=True)

    total_copied = 0
    total_missing = 0

    print(f"\nProcessing {split_name} set...")

    # Iterate through all samples scanned by ImageFolder
    for source_path, label_idx in dataset.samples:
        class_name = class_idx_to_name[label_idx]

        # Check if the file path exists on disk
        if os.path.exists(source_path):
            # Create the target class folder (e.g., 'Filtered_Plant_Dataset/train/Apple___scab')
            target_class_dir = os.path.join(target_split_dir, class_name)
            os.makedirs(target_class_dir, exist_ok=True)

            # Define the destination path
            filename = os.path.basename(source_path)
            target_path = os.path.join(target_class_dir, filename)

            # Copy the file to the new location (copy2 preserves metadata)
            try:
                shutil.copy2(source_path, target_path)
                total_copied += 1
            except Exception as e:
                print(f"Warning: Could not copy {source_path}. Error: {e}")
        else:
            total_missing += 1

    print(f"--- {split_name} Summary ---")
    print(f"Total valid images copied: {total_copied}")
    print(f"Total missing files skipped: {total_missing}")
    if total_missing > 0:
        print("⚠️ NOTE: The original dataset structure had missing file entries.")


def merge_data(source_split_path, target_combined_path, action='copy'):
    """Iterates through a source split (train/valid) and moves/copies files
    into a consolidated structure."""

    print(f"\nProcessing data from: {os.path.basename(source_split_path)}...")

    # Iterate through all class folders in the source split
    for class_name in os.listdir(source_split_path):
        source_class_folder = os.path.join(source_split_path, class_name)

        # Skip if not a directory (e.g., if there are hidden files)
        if not os.path.isdir(source_class_folder):
            continue

        # Define the target class folder in the combined structure
        target_class_folder = os.path.join(target_combined_path, class_name)

        # Create the target class folder if it doesn't exist
        os.makedirs(target_class_folder, exist_ok=True)

        # Count files processed for logging
        count = 0

        # Iterate through images in the source class folder
        for filename in os.listdir(source_class_folder):
            source_file = os.path.join(source_class_folder, filename)
            target_file = os.path.join(target_class_folder, filename)

            # Ensure it's a file
            if os.path.isfile(source_file):
                # We use copy to keep the original file in the train/valid folders
                # (safer than move/rename)
                try:
                    shutil.copy2(source_file, target_file)  # copy2 preserves metadata
                    count += 1
                except Exception as e:
                    print(f"Error processing {source_file}: {e}")

        print(f"  -> Merged {count} images into class '{class_name}'")

def ScanFilesForFilePathsAndLabels(COMBINED_DIR):
    filepaths = []
    labels = []
    for class_name in os.listdir(COMBINED_DIR):
        class_folder = os.path.join(COMBINED_DIR, class_name)

        if os.path.isdir(class_folder):
            for filename in os.listdir(class_folder):
                file_path = os.path.join(class_folder, filename)
                if os.path.isfile(file_path):
                    filepaths.append(file_path)
                    labels.append(class_name)

    if not filepaths:
        print("❌ ERROR: No files found. Check your COMBINED_DIR path.")
        exit()
    return filepaths,labels

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def calculate_normalization_stats(dataset_path, img_size=128):
    dataset = datasets.ImageFolder(dataset_path, transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ]))

    # Using the workers count you identified earlier
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    mean = 0.0
    std = 0.0
    total_images = 0

    print(f"📊 Calculating stats for dataset at: {dataset_path}")

    # Wrap loader with tqdm for a progress bar
    # unit="batch" shows the progress in terms of image batches
    for images, _ in tqdm(loader, desc="Computing Mean/Std", unit="batch"):
        batch_samples = images.size(0)

        # Reshape to (batch_size, channels, height * width)
        images = images.view(batch_samples, images.size(1), -1)

        # Sum the mean and std across the batch for each channel
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    # Final division to get the average across all images
    mean /= total_images
    std /= total_images

    print(f"\n✅ Calculation Complete!")
    print(f"Mean: {mean.tolist()}")
    print(f"Std:  {std.tolist()}")

    return mean, std

def copy_files_to_split(target_dir, file_list, split_name):
    """Copies files into the target split directory, maintaining class subfolders."""
    count = 0
    print(f"\nCopying files for {split_name} split...")

    for i, source_path in enumerate(file_list):
        # The class name is the parent directory name of the source file
        class_name = os.path.basename(os.path.dirname(source_path))

        # 1. Define and create the class folder in the new split directory
        target_class_dir = os.path.join(target_dir, class_name)
        os.makedirs(target_class_dir, exist_ok=True)

        # 2. Define the destination path
        filename = os.path.basename(source_path)
        target_path = os.path.join(target_class_dir, filename)

        # 3. Copy the file
        try:
            shutil.copy2(source_path, target_path)
            count += 1
        except Exception as e:
            print(f"Error copying {source_path}: {e}")

    print(f"✅ {split_name} Split Complete. Total files copied: {count}")
    return count