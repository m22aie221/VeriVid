import os
import json
import cv2
import glob
# File Paths
# DATASET_CONFIG_FILE = "/scratch/data/m22aie221/workspace/VeriVid/dataset_config.json"
# DATASET_METADATA_FILE = "/scratch/data/m22aie221/workspace/VeriVid/dataset_metrics.json"
# OUTPUT_ROOT = "/scratch/data/m22aie221/workspace/VeriVid/preprocessed/frames"

# Default base directory
DEFAULT_BASE_DIR = "/scratch/data/m22aie221/workspace/VeriVid"

# Global variables
BASE_DIR = ""
DATASET_CONFIG_FILE = ""
DATASET_METADATA_FILE = ""
OUTPUT_ROOT = ""

def parse_args():
    """Parses command-line arguments with a default BASE_DIR."""
    parser = argparse.ArgumentParser(description="Set BASE_DIR dynamically.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default=DEFAULT_BASE_DIR,  # Default value
        help=f"Base directory path (default: {DEFAULT_BASE_DIR})"
    )
    return parser.parse_args()

def update_global_paths(base_dir):
    """Updates global paths dynamically based on BASE_DIR."""
    global BASE_DIR, DATASET_CONFIG_FILE, DATASET_METADATA_FILE, OUTPUT_ROOT

    BASE_DIR = base_dir
    DATASET_CONFIG_FILE = os.path.join(BASE_DIR, "dataset_config.json")
    DATASET_METADATA_FILE = os.path.join(BASE_DIR, "dataset_metrics.json")
    OUTPUT_ROOT = os.path.join(BASE_DIR, "preprocessed", "frames")


# ✅ Load Dataset Configuration
def load_dataset_config():
    """ Load dataset configuration including max_frames limit """
    try:
        with open(DATASET_CONFIG_FILE, "r") as f:
            config = json.load(f)
        return config.get("max_frames", 201)  # Default to 301 if not specified
    except (FileNotFoundError, json.JSONDecodeError):
        raise ValueError("❌ Error: dataset_config.json is missing or corrupted!")

# ✅ Load Dataset Metadata
def load_metadata():
    """ Load dataset metadata from JSON """
    try:
        with open(DATASET_METADATA_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        raise ValueError("❌ Error: dataset_metrics.json is corrupted or not formatted correctly!")
    except FileNotFoundError:
        raise FileNotFoundError("❌ Error: dataset_metrics.json not found!")

# ✅ Extract Frames from MP4
def extract_frames(video_path, output_dir, max_frames):
    """ Extract frames from a video and save as images """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break  # Stop if the video is finished

        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

        if frame_count >= max_frames:
            break

    cap.release()
    print(f"✅ Extracted {frame_count} frames from {video_path} to {output_dir}")
    return frame_count

def count_existing_frames(video_folder):
    """Count the number of frame images (.png, .jpg) in the output folder."""
    return len(glob.glob(os.path.join(video_folder, "*.png"))) + len(glob.glob(os.path.join(video_folder, "*.jpg")))

# ✅ Process Only Train/Test Videos
def process_videos():
    """ Process videos based on dataset_metrics.json and extract frames for train/test videos only """
    metadata = load_metadata()
    max_frames = load_dataset_config()  # Get max_frames from dataset_config.json

    total_videos = metadata["train"]["total_videos"] + metadata["test"]["total_videos"]
    print(f"📊 Total train/test videos to process: {total_videos}")
    print(f"frames to extract {max_frames}")

    for category in ["facial", "non-facial"]:
        for label in ["real", "fake"]:
            for video_name, video_data in metadata["videos"][category][label].items():
                
                # ✅ Process only if train="yes" or test="yes"
                if video_data.get("train") != "yes" and video_data.get("test") != "yes":
                    continue

                video_path = video_data["path"]
                relative_path = os.path.relpath(video_path, "/scratch/data/m22aie221/workspace/dataset")  # Maintain structure
                
                # Create output folder with MP4 name (without extension)
                video_folder = os.path.join(OUTPUT_ROOT, os.path.dirname(relative_path), os.path.splitext(video_name)[0])
                os.makedirs(video_folder, exist_ok=True)


                existing_frame_count = count_existing_frames(video_folder)
                print(f"{video_folder}: {existing_frame_count}")

                if existing_frame_count > 0:
                    print(f"⏩ Skipping {video_name}, frames already extracted ({existing_frame_count}).")
                    continue  # Skip reprocessing


                frame_count = extract_frames(video_path, video_folder, max_frames)

                # ✅ Update JSON status
                metadata["videos"][category][label][video_name]["frame"] = frame_count

    # ✅ Save Updated JSON
    with open(DATASET_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

    print("🚀 Frame extraction completed and JSON updated!")


# ✅ Main Execution
def main():
    args = parse_args()
    update_global_paths(args.base_dir)  # Update global paths
    process_videos()


# ✅ Run Frame Extraction
if __name__ == "__main__":
    main()
