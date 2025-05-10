import os
from PIL import Image

from config.params import *


def get_file_map(folder: str) -> dict:
    """
    Create a dictionary mapping base filenames (without extension) to full filenames.

    Args:
        folder (str): Path to the folder containing files.

    Returns:
        dict: Mapping of base filenames to full filenames.
    """
    return {
        os.path.splitext(f)[0]: f
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    }


def clean_unmatched_files_and_mismatched_sizes(folder1: str, folder2: str) -> None:
    """
    Remove unmatched files and files with mismatched image sizes from two folders.

    Args:
        folder1 (str): Path to the first folder (e.g., images).
        folder2 (str): Path to the second folder (e.g., masks).
    """
    # Get file mappings for both folders
    files1 = get_file_map(folder1)
    files2 = get_file_map(folder2)

    # Identify common and unique files
    common_files = set(files1.keys()) & set(files2.keys())
    only_in_1 = set(files1.keys()) - set(files2.keys())
    only_in_2 = set(files2.keys()) - set(files1.keys())

    # Delete unmatched files from folder1
    for base_name in only_in_1:
        path = os.path.join(folder1, files1[base_name])
        os.remove(path)
        print(f"Deleted unmatched file from {folder1}: {path}")

    # Delete unmatched files from folder2
    for base_name in only_in_2:
        path = os.path.join(folder2, files2[base_name])
        os.remove(path)
        print(f"Deleted unmatched file from {folder2}: {path}")

    # Check for size mismatches in common files
    mismatched = []
    for base_name in common_files:
        path1 = os.path.join(folder1, files1[base_name])
        path2 = os.path.join(folder2, files2[base_name])

        try:
            with Image.open(path1) as img1, Image.open(path2) as img2:
                if img1.size != img2.size:
                    mismatched.append(base_name)
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            mismatched.append(base_name)

    # Delete mismatched size files from both folders
    for base_name in mismatched:
        path1 = os.path.join(folder1, files1.get(base_name, ""))
        path2 = os.path.join(folder2, files2.get(base_name, ""))

        if os.path.exists(path1):
            os.remove(path1)
            print(f"Deleted mismatched size file from {folder1}: {path1}")
        if os.path.exists(path2):
            os.remove(path2)
            print(f"Deleted mismatched size file from {folder2}: {path2}")

    # Print summary
    print(f"\nDeleted {len(only_in_1)} unmatched files from {folder1}")
    print(f"Deleted {len(only_in_2)} unmatched files from {folder2}")
    print(f"Deleted {len(mismatched)} mismatched size file pairs")


if __name__ == "__main__":
    # Clean training and validation datasets
    clean_unmatched_files_and_mismatched_sizes(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    clean_unmatched_files_and_mismatched_sizes(VAL_IMG_DIR, VAL_MASK_DIR)