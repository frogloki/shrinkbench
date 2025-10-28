import os
import requests
from tqdm import tqdm
import torchvision

# ==============================================================================
# Configuration
# ==============================================================================
# Base directory where all datasets will be stored
BASE_DATA_DIR = "./data"

# URLs and file info
PLACES365_URL = "http://data.csail.mit.edu/places/places365/places365_standard.tar.gz"
PLACES365_FILENAME = "places365_standard.tar.gz"

# ==============================================================================
# Helper Functions
# ==============================================================================

def download_file(url, destination_path):
    """Downloads a file with a progress bar."""
    print(f"Downloading {os.path.basename(destination_path)} from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte

            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(destination_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()

            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("ERROR, something went wrong during download.")
                return False
        print("Download complete.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False




# ==============================================================================
# Dataset Handlers
# ==============================================================================

def setup_torchvision_datasets(root_dir):
    """Downloads MNIST, CIFAR10, and CIFAR100 using torchvision."""
    datasets = {
        "MNIST": torchvision.datasets.MNIST,
        "CIFAR10": torchvision.datasets.CIFAR10,
        "CIFAR100": torchvision.datasets.CIFAR100,
    }

    print("\n--- Setting up torchvision datasets (MNIST, CIFAR10, CIFAR100) ---")
    for name, dset_class in datasets.items():
        print(f"\nChecking/Downloading {name}...")
        # Download train and test sets
        dset_class(root=os.path.join(root_dir, name), train=True, download=True)
        dset_class(root=os.path.join(root_dir, name), train=False, download=True)
        print(f"{name} is ready.")

def handle_imagenet(root_dir):
    """Provides instructions for downloading ImageNet manually."""
    imagenet_dir = os.path.join(root_dir, "ImageNet")
    os.makedirs(imagenet_dir, exist_ok=True)

    train_archive = os.path.join(imagenet_dir, "ILSVRC2012_img_train.tar")
    val_archive = os.path.join(imagenet_dir, "ILSVRC2012_img_val.tar")

    print("\n--- Handling ImageNet ILSVRC 2012 ---")
    print("=" * 60)
    print("⚠️  ACTION REQUIRED: ImageNet cannot be downloaded automatically. ⚠️")
    print("=" * 60)
    print("You must download it manually after registering and agreeing to the terms.")
    print("\n1. Go to the official website: https://image-net.org/challenges/LSVRC/2012/index.php")
    print("2. Create an account and request access to the downloads.")
    print("3. Once approved, download the following two files:")
    print(f"   - Training images (Task 1 & 2): ILSVRC2012_img_train.tar (~138 GB)")
    print(f"   - Validation images (all tasks): ILSVRC2012_img_val.tar (~6.3 GB)")
    print("\n4. Place the downloaded files in this directory:")
    print(f"   -> {os.path.abspath(imagenet_dir)}")
    print("\nAfter placing the files, you will need to extract them. The standard procedure is:")
    print(f"   - For training set: `cd {os.path.abspath(imagenet_dir)} && mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train && tar -xvf ILSVRC2012_img_train.tar && find . -name \"*.tar\" | while read NAME ; do mkdir -p \"${{NAME%.tar}}\"; tar -xvf \"$NAME\" -C \"${{NAME%.tar}}\"; rm -f \"$NAME\"; done`")
    print(f"   - For validation set: `cd {os.path.abspath(imagenet_dir)} && mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar`")
    print("   - You may also need a script to move validation images into labeled subfolders.")
    print("-" * 60)

    if os.path.exists(train_archive) and os.path.exists(val_archive):
        print("\nSUCCESS: Found ImageNet archives. Please proceed with the extraction steps above.")
    else:
        print("\nSTATUS: ImageNet archive files not found. Please follow the manual download steps.")

def setup_places365(root_dir):
    """Downloads and extracts the Places365 standard small dataset."""
    places_dir = os.path.join(root_dir, "Places365")
    archive_path = os.path.join(places_dir, PLACES365_FILENAME)
    
    # The extracted content is typically a folder named 'places365_standard'
    extracted_content_path = os.path.join(places_dir, 'data_256') 

    print("\n--- Setting up Places365 (Standard Small 256x256) ---")

    if os.path.exists(extracted_content_path) or os.path.exists(os.path.join(places_dir, 'places365_standard')):
        print("Places365 already appears to be downloaded and extracted. Skipping.")
        return

    os.makedirs(places_dir, exist_ok=True)

    if not os.path.exists(archive_path):
        if not download_file(PLACES365_URL, archive_path):
            print("Failed to download Places365. Aborting this step.")
            return

    extract_archive(archive_path, places_dir)
    print("Places365 is ready.")
    # Optional: Clean up the large archive file after extraction
    # print("Cleaning up archive file...")
    # os.remove(archive_path)


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    print("Starting dataset download and setup process.")
    print(f"All data will be stored in: {os.path.abspath(BASE_DATA_DIR)}")
    print("=" * 60)
    # print("WARNING: This script will download a large amount of data (~25 GB automatically,")
    # print("and instructs you to download an additional ~150 GB manually for ImageNet).")
    # print("Please ensure you have sufficient disk space and a stable internet connection.")
    # print("=" * 60)
    # input("Press Enter to continue or Ctrl+C to cancel...")

    # Create the base directory if it doesn't exist
    os.makedirs(BASE_DATA_DIR, exist_ok=True)

    # Process each dataset
    setup_torchvision_datasets(BASE_DATA_DIR)

    print("\n\nAll tasks complete. Please check the output above for any manual steps required.")