import urllib.request
import zipfile
import os

url = "https://github.com/robaita/introduction_to_machine_learning/raw/main/dataset.zip"
zip_path = "dataset.zip"
extract_dir = "dataset"

print(f"Downloading dataset from {url}...")
urllib.request.urlretrieve(url, zip_path)

print("Extracting dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Cleaning up...")
os.remove(zip_path)

print("Dataset downloaded and extracted successfully.")
