import os
import urllib.request
import sys

def download_with_progress(url, dest):
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = int(downloaded * 100 / total_size) if total_size > 0 else 0
        sys.stdout.write(f"\rDownloading... {percent}%")
        sys.stdout.flush()
        if downloaded >= total_size:
            print("\nDownload complete.")

    urllib.request.urlretrieve(url, dest, reporthook=show_progress)

# COCO dataset URLs
base_url = "http://images.cocodataset.org/"
files = {
    "train2017": "zips/train2017.zip",
    "val2017": "zips/val2017.zip",
    "annotations": "annotations/annotations_trainval2017.zip",
}

# Directory to save files
save_dir = r"D:\coco_dataset"
os.makedirs(save_dir, exist_ok=True)

# Download each file with progress
for name, url_suffix in files.items():
    url = base_url + url_suffix
    dest_path = os.path.join(save_dir, os.path.basename(url_suffix))
    print(f"\nDownloading {name} to {dest_path}")
    download_with_progress(url, dest_path)
