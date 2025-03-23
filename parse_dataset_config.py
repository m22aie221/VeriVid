import os
import json

# Paths
DATASET_CONFIG_FILE = "/scratch/data/m22aie221/workspace/VeriVid/dataset_config.json"
OUTPUT_JSON_FILE = "/scratch/data/m22aie221/workspace/VeriVid/dataset_metrics.json"

# ✅ Load Dataset Configuration
def load_dataset_config():
    """ Load dataset_config.json and verify structure """
    if os.path.exists(DATASET_CONFIG_FILE):
        with open(DATASET_CONFIG_FILE, "r") as f:
            return json.load(f)
    raise FileNotFoundError("❌ dataset_config.json not found!")

# ✅ Recursively Find .MP4 Files
def find_mp4_files(paths):
    """ Scan multiple dataset paths for .mp4 files """
    mp4_files = []
    for root_path in paths:
        if not os.path.exists(root_path):
            print(f"⚠️ Warning: Path does not exist - {root_path}")
            continue

        for root, _, files in os.walk(root_path):
            for file in files:
                if file.endswith(".mp4"):
                    mp4_files.append(os.path.join(root, file))
    
    # ✅ Debugging: Print found files
    if not mp4_files:
        print(f"⚠️ Warning: No .mp4 files found in {paths}")

    return mp4_files

# ✅ Generate Metadata JSON
def generate_metadata():
    dataset_config = load_dataset_config()

    metadata = {
        "summary": {
            "total_videos": 0,
            "facial": {"real": 0, "fake": 0},
            "non-facial": {"real": 0, "fake": 0}
        },
        "trial_num": 0,
        "videos": {"facial": {"real": {}, "fake": {}}, "non-facial": {"real": {}, "fake": {}}}
    }

    # ✅ Ensure we correctly read dataset paths
    dataset_paths = dataset_config.get("dataset_paths", {})
    
    for category, labels in dataset_paths.items():
        for label, paths in labels.items():
            collected_videos = find_mp4_files(paths)

            # ✅ Debugging: Print number of files found
            print(f"📌 {category}/{label} → Found {len(collected_videos)} videos.")

            for video_path in collected_videos:
                video_name = os.path.basename(video_path)
                metadata["videos"][category][label][video_name] = {
                    "path": video_path,
                    "frame": "none",
                    "spatial": "none",
                    "optical": "none",
                    "depth": "none",
                    "train": "none",
                    "test": "none"
                }
                metadata["summary"][category][label] += 1
                metadata["summary"]["total_videos"] += 1
                #print(video_path)

    # ✅ Save JSON File
    with open(OUTPUT_JSON_FILE, "w") as f:
        json.dump(metadata, f, indent=4)
        f.flush()
    print("✅ dataset_metrics.json updated!")
    print_dataset_metrics(metadata)

# ✅ Print Dataset Metrics
def print_dataset_metrics(metadata):
    summary = metadata["summary"]

    print("\n📊 **Dataset Statistics:**")
    print(f"📌 Total Videos: {summary['total_videos']}")
    print(f"   ✅ Facial Real: {summary['facial']['real']}")
    print(f"   ❌ Facial Fake: {summary['facial']['fake']}")
    print(f"   ✅ Non-Facial Real: {summary['non-facial']['real']}")
    print(f"   ❌ Non-Facial Fake: {summary['non-facial']['fake']}")

# ✅ Run the script
if __name__ == "__main__":
    generate_metadata()
