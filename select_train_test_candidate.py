import os
import json
import random
from collections import defaultdict
import numpy as np

# File Paths
# DATASET_METADATA_FILE = "/scratch/data/m22aie221/workspace/VeriVid/dataset_metrics.json"
# DATASET_CONFIG_FILE = "/scratch/data/m22aie221/workspace/VeriVid/dataset_config.json"


import argparse
import os

# Default base directory
DEFAULT_BASE_DIR = "/scratch/data/m22aie221/workspace/VeriVid"

# Global variables initialized with empty strings
DATASET_METADATA_FILE = ""
DATASET_CONFIG_FILE = ""

def update_global_paths(base_dir):
    """Updates global variables based on BASE_DIR."""
    global DATASET_METADATA_FILE, DATASET_CONFIG_FILE

    # Ensure that base_dir is not empty
    base_dir = base_dir.strip()

    # Dynamically update the paths based on the base directory
    DATASET_METADATA_FILE = os.path.join(base_dir, "dataset_metrics.json")
    DATASET_CONFIG_FILE = os.path.join(base_dir, "dataset_config.json")

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Set BASE_DIR dynamically.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default=DEFAULT_BASE_DIR,  # Use default if base_dir is not passed
        help=f"Base directory path (default: {DEFAULT_BASE_DIR})"
    )
    return parser.parse_args()

def load_json(file_path):
    """ Load JSON file """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"❌ Error: {file_path} is corrupted or not formatted correctly!")
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Error: {file_path} not found!")

def group_videos_by_folder(metadata):
    """ Group videos by folder for fair representation in test set """
    folder_mapping = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for category in ["facial", "non-facial"]:
        for label in ["real", "fake"]:
            for video_name, video_data in metadata["videos"].get(category, {}).get(label, {}).items():
                folder_path = os.path.dirname(video_data["path"])
                folder_mapping[category][label][folder_path].append((video_name, video_data))

    return folder_mapping

def test_already_selected(metadata):
    """ Check if test set is already selected in dataset_metrics.json """
    return "test" in metadata and "total_videos" in metadata["test"]

def select_fixed_test_set(folder_mapping, metadata, split_config):
    """ Selects the test set only once based on the split configuration """
    if test_already_selected(metadata):
        print("✅ Test set already selected. Skipping test selection.")
        return  

    print("🔹 Selecting fixed test set...")

    for category in ["facial", "non-facial"]:
        for label in ["real", "fake"]:
            new_test_count = split_config["test_new"][category][label]
            available_videos = list(metadata["videos"][category][label].items())

            # Ensure we don't select more than available
            if len(available_videos) < new_test_count:
                print(f"⚠️ Warning: Not enough {category} {label} videos for test. Adjusting count.")
                new_test_count = len(available_videos)

            random.shuffle(available_videos)
            selected_videos = available_videos[:new_test_count]

            # Mark test videos
            for video_name, video_data in selected_videos:
                metadata["videos"][category][label][video_name]["test"] = "yes"

    print("✅ Test set selected and fixed in dataset_metrics.json.")


def select_new_test_set(folder_mapping, metadata, split_config):
    print("🔹 Selecting new test set...")

    for category in ["facial", "non-facial"]:
        for label in ["real", "fake"]:
            test_count = split_config["test"][category][label]
            available_videos = list(metadata["videos"][category][label].items())

            # Ensure we don't select more than available
            if len(available_videos) < test_count:
                print(f"⚠️ Warning: Not enough {category} {label} videos for test. Adjusting count.")
                test_count = len(available_videos)

            random.shuffle(available_videos)
            selected_videos = available_videos[:test_count]

            # Mark test videos
            for video_name, video_data in selected_videos:
                metadata["videos"][category][label][video_name]["test"] = "yes"

    print("✅ Test set selected and fixed in dataset_metrics.json.")


def select_train_set(metadata, split_config, trial_no=1):
    """ Select training samples from remaining videos after test selection """

    # Dictionary to store available video counts
    available_videos_count = {
        "facial": {"real": 0, "fake": 0},
        "non-facial": {"real": 0, "fake": 0}
    }

    # Dictionary to store total video counts for validation
    total_videos_count = {
        "facial": {"real": 0, "fake": 0},
        "non-facial": {"real": 0, "fake": 0}
    }

    # Iterate through the dataset and count videos correctly
    for category in ["facial", "non-facial"]:
        for label in ["real", "fake"]:
            video_entries = metadata["videos"].get(category, {}).get(label, {})

            # Total count for reference
            total_videos_count[category][label] = len(video_entries)

            # Count videos available for training (where train = "none" and test = "none")
            available_videos_count[category][label] = sum(
                1 for video_data in video_entries.values()
                if video_data.get("train") == "none" and video_data.get("test") == "none"
            )

    # Print the total dataset statistics
    print("\n📊 **Dataset Statistics:**")
    for category in total_videos_count:
        for label in total_videos_count[category]:
            print(f"📌 {category}/{label} → Found {total_videos_count[category][label]} videos.")

    # Print the available training video counts
    print("\n📊 **Available Training Videos:**")
    for category in available_videos_count:
        for label in available_videos_count[category]:
            print(f"📌 {category}/{label} → {available_videos_count[category][label]} available for training.")


    
    for category in ["facial", "non-facial"]:
        for label in ["real", "fake"]:
            train_count = split_config["train"][category][label]

            # ✅ Corrected Filtering Condition
            available_videos = [
                (video_name, video_data)
                for video_name, video_data in metadata["videos"].get(category, {}).get(label, {}).items()
                if (video_data.get("test") != "yes" and video_data.get("train") not in ["done", "yes"])  
            ]

            print(f"available_videos video len: {len(available_videos)}, train_count: {train_count}")

            # ✅ Handle Case When There Aren't Enough Videos
            if len(available_videos) < train_count:
                print(f"⚠️ Warning: Not enough {category} {label} videos for training. Adjusting count.")
                train_count = len(available_videos)

            # ✅ Extract only video names
            video_list = sorted([vid[0] for vid in available_videos])

            if len(video_list) == 0:
                print(f"⚠️ Warning: No available videos for {category} {label}. Skipping selection.")
                selected_videos = []
            else:
                np.random.seed(trial_no)  # ✅ Ensure deterministic selection
                selected_video_names = list(np.random.choice(video_list, min(len(video_list), train_count), replace=False))

                # ✅ Retrieve full (video_name, video_data) tuples
                selected_videos = [(video_name, metadata["videos"][category][label][video_name]) for video_name in selected_video_names]

            print(f"available_videos video len: {len(available_videos)}, train_count: {train_count}, selected_videos: {len(selected_videos)}")

            # ✅ Mark Selected Videos for Training
            for video_name, video_data in selected_videos:
                metadata["videos"][category][label][video_name]["train"] = "yes"

    print("✅ Training set selected dynamically.")

def mark_processing_fields(metadata):
    """ Mark processing fields for each video as 'none' """
    for category in ["facial", "non-facial"]:
        for label in ["real", "fake"]:
            for video_name, video_data in metadata["videos"].get(category, {}).get(label, {}).items():
                metadata["videos"][category][label][video_name]["frame"] = 0
                metadata["videos"][category][label][video_name]["spatial"] = "none"
                metadata["videos"][category][label][video_name]["optical"] = "none"
                metadata["videos"][category][label][video_name]["depth"] = "none"

    print("✅ Processing fields marked as 'none'.")

def compute_final_split_metrics(metadata):
    """ Compute final train/test split based on 'yes'/'no' selections """
    train_counts  = {"total_videos": 0, "facial": {"real": 0, "fake": 0}, "non-facial": {"real": 0, "fake": 0}}
    replay_counts = {"total_videos": 0, "facial": {"real": 0, "fake": 0}, "non-facial": {"real": 0, "fake": 0}}

    test_counts = {"total_videos": 0, "facial": {"real": 0, "fake": 0}, "non-facial": {"real": 0, "fake": 0}}

    for category in ["facial", "non-facial"]:
        for label in ["real", "fake"]:
            for video_name, video_data in metadata["videos"].get(category, {}).get(label, {}).items():
                if video_data.get("train") == "yes":
                    train_counts[category][label] += 1
                    train_counts["total_videos"] += 1
                elif video_data.get("train") == "done":
                    replay_counts[category][label] += 1
                    replay_counts["total_videos"] += 1                    
                if video_data.get("test") == "yes":
                    test_counts[category][label] += 1
                    test_counts["total_videos"] += 1

    print("train_counts: ", train_counts)
    print("replay_counts: ", replay_counts)
    print("test_counts: ", test_counts)
    metadata["train"] = train_counts
    metadata["test"] = test_counts
    print("✅ Final train/test split metrics computed.")

def update_dataset_json(metadata):
    """ Save the updated dataset with train/test selections """
    with open(DATASET_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)
        f.flush()

    print("✅ Updated dataset_metrics.json with final train/test split.")

def print_final_metrics(metadata):
    """ Print final train/test split metrics """
    print("\n📊 **Final Dataset Split Metrics:**")
    print(f"📌 Total Videos: {metadata['summary']['total_videos']}")
    print(f"   ✅ Train: {metadata['train']['total_videos']} | Test: {metadata['test']['total_videos']}")
    print(f"   ✅ Facial Real Train: {metadata['train']['facial']['real']} | Test: {metadata['test']['facial']['real']}")
    print(f"   ❌ Facial Fake Train: {metadata['train']['facial']['fake']} | Test: {metadata['test']['facial']['fake']}")
    print(f"   ✅ Non-Facial Real Train: {metadata['train']['non-facial']['real']} | Test: {metadata['test']['non-facial']['real']}")
    print(f"   ❌ Non-Facial Fake Train: {metadata['train']['non-facial']['fake']} | Test: {metadata['test']['non-facial']['fake']}")

def main():
    args = parse_args()
    # Update global paths based on the provided or default base_dir
    update_global_paths(args.base_dir)
    metadata = load_json(DATASET_METADATA_FILE)
    split_config = load_json(DATASET_CONFIG_FILE)["split_config"]

    # Step 1: Group videos by their folders for fair test set selection
    folder_mapping = group_videos_by_folder(metadata)

    # Step 2: Select and fix the test set (only once)
    select_fixed_test_set(folder_mapping, metadata, split_config)
    select_new_test_set(folder_mapping, metadata, split_config)
    # Step 3: Select the training set from remaining videos
    trial_num = metadata.get("trial_num", 0)
    print("trial_num no:", trial_num)
    select_train_set(metadata, split_config, trial_num)

    # Step 4: Mark processing fields as 'none'
    # mark_processing_fields(metadata)

    # Step 5: Compute final train/test split metrics
    compute_final_split_metrics(metadata)

    # Step 6: Save updated JSON
    update_dataset_json(metadata)

    # Step 7: Print final metrics
    print_final_metrics(metadata)

if __name__ == "__main__":
    main()
