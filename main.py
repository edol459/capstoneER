# Description: Automatically collects and stores images from DuckDuckGo search

import os
import requests
from io import BytesIO
from PIL import Image
from duckduckgo_search import DDGS




# configuration
# list of search phrases
SEARCH_PHRASES = [
    "happy face person",
    "sad face person",
    "angry face person",
    "surprised face person",
    "fearful face person",
    "neutral face person"
]

# number of images to download per phrase
MAX_IMAGES = 20

# Folder to save all data
DATASET_DIR = "emotion_dataset"

# Resize images?
RESIZE_IMAGES = True
TARGET_SIZE = (128, 128)  # pixels (width, height)


# Create dataset folder if not exists
os.makedirs(DATASET_DIR, exist_ok=True)

def download_images():
    print("Starting image collection...")
    with DDGS() as ddgs:
        for phrase in SEARCH_PHRASES:
            print(f"\nSearching for: '{phrase}'")
            folder_name = phrase.replace(" ", "_")
            folder_path = os.path.join(DATASET_DIR, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            # Search for image URLs
            results = ddgs.images(phrase, max_results=MAX_IMAGES)
            count = 0

            for i, r in enumerate(results):
                img_url = r.get("image")
                if not img_url:
                    continue
                try:
                    # Download image data
                    response = requests.get(img_url, timeout=10)
                    img = Image.open(BytesIO(response.content)).convert("RGB")

                    # Optional: resize
                    if RESIZE_IMAGES:
                        img = img.resize(TARGET_SIZE)

                    # Save image
                    save_path = os.path.join(folder_path, f"{i+1}.jpg")
                    img.save(save_path)
                    count += 1
                    print(f"Saved: {save_path}")

                except Exception as e:
                    print(f" Skipped one: {e}")
                    continue

            print(f"Collected {count} images for '{phrase}'")

    print("\nAll image downloads completed successfully!")

# Run the script
if __name__ == "__main__":
    download_images()
