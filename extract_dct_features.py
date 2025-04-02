import os
import cv2
import json
import numpy as np
from scipy.fftpack import dct
from tqdm import tqdm

# Define paths
DATASET_METADATA_FILE = "/scratch/data/m22aie221/workspace/VeriVid/dataset_metrics.json"
OUTPUT_ROOT = "/scratch/data/m22aie221/workspace/VeriVid/preprocessed/frequency_dct/"
DATASET_CONFIG_FILE = "/scratch/data/m22aie221/workspace/VeriVid/dataset_config.json"

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ensure output root exists
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ✅ Load Dataset Configuration
def load_dataset_config():
    """ Load dataset configuration including max_frames limit """
    try:
        with open(DATASET_CONFIG_FILE, "r") as f:
            config = json.load(f)
        return config.get("max_frames", 201)  # Default to 301 if not specified
    except (FileNotFoundError, json.JSONDecodeError):
        raise ValueError("❌ Error: dataset_config.json is missing or corrupted!")

def load_metadata():
    """ Load dataset metadata from JSON. """
    with open(DATASET_METADATA_FILE, "r") as f:
        return json.load(f)

def apply_dct(image_path, output_path):
    """ Computes 2D DCT on an image and saves it as a PNG frame. """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
    
    if img is None:
        print(f"Error loading image: {image_path}")
        return

    # Convert to float32 for precision
    img = np.float32(img)

    # Apply 2D DCT
    dct_img = dct(dct(img.T, norm='ortho').T, norm='ortho')

    # Convert to log scale for better visibility
    dct_log = np.log(np.abs(dct_img) + 1)

    # Normalize to [0, 255] for saving
    dct_norm = cv2.normalize(dct_log, None, 0, 255, cv2.NORM_MINMAX)
    dct_img_uint8 = np.uint8(dct_norm)

    # Save the transformed image
    cv2.imwrite(output_path, dct_img_uint8)

def process_videos():
    """ Process videos and apply DCT to extracted frames. """
    metadata = load_metadata()
    num_frames = load_dataset_config()
    total_videos = metadata["train"]["total_videos"] + metadata["test"]["total_videos"]
    print(f"📊 Total train/test videos to process: {total_videos}")

    for category in ["facial", "non-facial"]:
        for label in ["real", "fake"]:
            for video_name, video_data in metadata["videos"][category][label].items():
                
                if video_data.get("train") != "yes" and video_data.get("test") != "yes":
                    continue

                video_path = video_data["path"]
                relative_path = os.path.relpath(video_path, "/scratch/data/m22aie221/workspace/dataset")
                
                frame_folder = os.path.join("/scratch/data/m22aie221/workspace/VeriVid/preprocessed/frames/", os.path.dirname(relative_path), os.path.splitext(video_name)[0])
                output_video_folder = os.path.join(OUTPUT_ROOT, os.path.dirname(relative_path), os.path.splitext(video_name)[0])
                os.makedirs(output_video_folder, exist_ok=True)

                if not os.path.exists(frame_folder):
                    print(f"⚠️ Skipping {video_name}, frame folder not found: {frame_folder}")
                    continue

                frame_count = 0
                for frame_name in sorted(os.listdir(frame_folder)):
                    if not frame_name.endswith(".png"):
                        continue
                    frame_path = os.path.join(frame_folder, frame_name)
                    output_path = os.path.join(output_video_folder, frame_name)

                    apply_dct(frame_path, output_path)
                    frame_count = frame_count + 1
                    if frame_count > num_frames:
                        break
                    #apply_fft(frame_path, output_path)

    print("🚀 DCT transformation completed!")

# ✅ Run DCT Extraction
if __name__ == "__main__":
    process_videos()
