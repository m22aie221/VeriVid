import os
import json
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.fftpack import dct

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Paths
# ✅ Base Paths
MODEL_NAME = "VeriVidTrans"
# BASE_DIR = "/scratch/data/m22aie221/workspace/VeriVid"
# RESULTS_DIR = os.path.join(BASE_DIR, "results_trans")
# DATASET_METADATA_FILE = os.path.join(BASE_DIR, "dataset_metrics.json")
# MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)
# PLOT_DIR = os.path.join(BASE_DIR, "plots")
# FEATURE_ROOT = os.path.join(BASE_DIR, "features_times")
# # ✅ New frequency feature path
# FREQ_FEATURE_ROOT = os.path.join(BASE_DIR, "frequency_dct")
DEFAULT_BASE_DIR = "/scratch/data/m22aie221/workspace/VeriVid"


# Define global variables (to be updated dynamically)
BASE_DIR = ""
MODEL_NAME = "VeriVidTrans"
RESULTS_DIR = ""
DATASET_METADATA_FILE = ""
MODEL_PATH = ""
PLOT_DIR = ""
FEATURE_ROOT = ""
FREQ_FEATURE_ROOT = ""

BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 20
VAL_SPLIT = 0.2  # 20% validation split
REPLAY_SIZE = 100  # Number of past videos to replay

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
    global BASE_DIR, RESULTS_DIR, DATASET_METADATA_FILE, MODEL_PATH, PLOT_DIR, FEATURE_ROOT, FREQ_FEATURE_ROOT

    BASE_DIR = base_dir
    RESULTS_DIR = os.path.join(BASE_DIR, "results_trans")
    DATASET_METADATA_FILE = os.path.join(BASE_DIR, "dataset_metrics.json")
    MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)
    PLOT_DIR = os.path.join(BASE_DIR, "plots")
    FEATURE_ROOT = os.path.join(BASE_DIR, "features_times")
    FREQ_FEATURE_ROOT = os.path.join(BASE_DIR, "frequency_dct")



def safe_load_features(path):
    """Load a single feature file from the given directory.
       Assumes only one .npy file is present.
    """
    try:
        files = [f for f in os.listdir(path) if f.endswith(".npy")]
        if len(files) == 0:
            print(f"❌ Warning: No feature file found in {path}.")
            return np.zeros((764,))  # Return a zero-vector instead
        # Load the first .npy file found
        feature_array = np.load(os.path.join(path, files[0]))
        if feature_array.ndim == 0:
            print(f"❌ Warning: Feature in {path} is empty or invalid.")
            return np.zeros((764,))  # Return a zero-vector instead
        return feature_array
    except Exception as e:
        print(f"❌ Error loading features from {path}: {e}")
        return np.zeros((764,))  # Return a zero-vector instead


# ✅ Load Dataset Metadata
def load_metadata():
    with open(DATASET_METADATA_FILE, "r") as f:
        return json.load(f)

# ✅ Updated Load Features from .npy Files
def load_features(metadata):
    video_feature_map, labels = {}, {}

    for category in ["facial", "non-facial"]:
        for label in ["real", "fake"]:
            for video_name, video_data in metadata["videos"][category][label].items():
                vid_no_ext = video_name.replace(".mp4", "")
                
                if not any([video_data.get("train") == "yes", video_data.get("train") == "done", video_data.get("test") == "yes"]):
                    continue  # Only pick train/test videos

                # ✅ Preserve Full Hierarchy
                relative_video_path = os.path.relpath(video_data["path"], "/scratch/data/m22aie221/workspace/dataset").replace(".mp4", "")

                print(f" {vid_no_ext}: {relative_video_path}")

                feature_paths = {
                    "spatial": os.path.join(FEATURE_ROOT, "spatial", relative_video_path),
                    "optical": os.path.join(FEATURE_ROOT, "optical", relative_video_path),
                    "depth": os.path.join(FEATURE_ROOT, "depth", relative_video_path),
                    "frequency_dct": os.path.join(FEATURE_ROOT, "frequency_dct", relative_video_path)
                }

                video_feature_map[vid_no_ext] = feature_paths
                labels[vid_no_ext] = 1 if label == "real" else 0  # Binary classification (Real=1, Fake=0)

    return video_feature_map, labels

# ✅ Select Training Videos (Including Replay Buffer)
def select_train_videos(metadata, video_feature_map, replay_size=50, trial_no=1):
    train_videos, replay_videos = [], []

    for category in ["facial", "non-facial"]:
        for label in ["real", "fake"]:
            for vid, data in metadata["videos"][category][label].items():
                vid_no_ext = vid.replace(".mp4", "")  # ✅ Remove .mp4 for matching

                # ✅ First Training: Use "train": "yes"
                if data.get("train") == "yes" and vid_no_ext in video_feature_map:
                    print(f"📌 Added to train: {vid_no_ext}")
                    train_videos.append(vid_no_ext)

                # ✅ Replay Buffer: Use "train": "done"
                elif data.get("train") == "done" and vid_no_ext in video_feature_map:
                    replay_videos.append(vid_no_ext)
                    print("Added to replay buffer: ",vid_no_ext, "len: ", len(replay_videos))


    # ✅ Ensure deterministic selection using trial_no as seed
    if len(replay_videos) > 0:
        np.random.seed(trial_no)
        replay_videos = np.random.choice(replay_videos, min(len(replay_videos), REPLAY_SIZE), replace=False).tolist()
    else:
        replay_videos = []

    print(f"📌 New Training Videos: {len(train_videos)}, Replay Buffer: {len(replay_videos)}")
    return train_videos + replay_videos


# ✅ Select Test Videos (Fixed Set)
def select_test_videos(metadata, video_feature_map):
    test_videos = []
    for category in ["facial", "non-facial"]:
        for label in ["real", "fake"]:
            for vid, data in metadata["videos"][category][label].items():
                vid_no_ext = vid.replace(".mp4", "")
                if data.get("test") == "yes" and vid_no_ext in video_feature_map:
                    print(f"📌 Added to test: {vid_no_ext}")
                    test_videos.append(vid_no_ext)
    print(f"📌 test Videos: {len(test_videos)}")
    return test_videos

# ✅ Updated Dataset Class for VeriVid with frequency features
class VeriVidDataset(Dataset):
    def __init__(self, video_feature_map, labels):
        self.video_feature_map = video_feature_map
        self.video_ids = list(video_feature_map.keys())
        self.labels = labels

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        feature_paths = self.video_feature_map[video_id]

        spatial_features = safe_load_features(feature_paths["spatial"])
        optical_features = safe_load_features(feature_paths["optical"])
        depth_features = safe_load_features(feature_paths["depth"])
        frequency_features = safe_load_features(feature_paths["frequency_dct"])


        # Concatenate all features
        video_feature_vector = np.concatenate([
            spatial_features.flatten(), 
            optical_features.flatten(), 
            depth_features.flatten(),
            frequency_features.flatten()  # Add frequency features
        ])
        
        return torch.tensor(video_feature_vector, dtype=torch.float32), torch.tensor(self.labels[video_id], dtype=torch.float32)


# ✅ VeriVid Model
class VeriVid(nn.Module):
    def __init__(self, input_dim):
        super(VeriVid, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)

# ✅ Update Training Status in dataset_metrics.json
def update_training_status(metadata, train_videos):
    """ Update training status: 'yes' → 'done' for next training cycle """
    for category in ["facial", "non-facial"]:
        for label in ["real", "fake"]:
            for vid in train_videos:
                vid_with_ext = vid + ".mp4"
                if vid_with_ext in metadata["videos"][category][label]:
                    metadata["videos"][category][label][vid_with_ext]["train"] = "done"

    # ✅ Save the updated metadata
    with open(DATASET_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

    print("✅ Training status updated (train → done).")

def save_results(trial_no):
    """ Save dataset, model, and plots for the current trial. """

    # ✅ Step 1: Update `trial_no` in `dataset_metrics.json`
    """ Update the trial number in dataset_metrics.json and ensure consistency. """
    if os.path.exists(DATASET_METADATA_FILE):
        with open(DATASET_METADATA_FILE, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # ✅ Step 1: Update `trial_num`
    metadata["trial_num"] = trial_no  # Ensure `trial_no` is updated

    # ✅ Step 2: Save updated `dataset_metrics.json` and flush immediately
    with open(DATASET_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)
        f.flush()  # ✅ Ensure data is immediately written to disk

    # ✅ Step 3: Copy `dataset_metrics.json` to the trial folder
    shutil.copy(DATASET_METADATA_FILE, os.path.join(current_trial_dir, "dataset_metrics.json"))

    print(f"✅ Updated `trial_num` to {trial_no} and saved in both base and trial folder.")

    # ✅ Step 4: Save Model Weights to BOTH Locations
    if os.path.exists(MODEL_PATH):
        shutil.copy(MODEL_PATH, os.path.join(current_trial_dir, MODEL_NAME))  # ✅ Save in trial folder
        print(f"✅ Model saved to {os.path.join(current_trial_dir, MODEL_NAME)}")
    else:
        print("⚠️ Warning: Model file not found. Skipping model save.")


    # ✅ Step 5: Save Plots to Trial Folder
    if os.path.exists(PLOT_DIR) and os.listdir(PLOT_DIR):
        for plot_file in os.listdir(PLOT_DIR):
            shutil.copy(os.path.join(PLOT_DIR, plot_file), current_trial_dir)
        print(f"📊 Saved plots to {current_trial_dir}")
    else:
        print("⚠️ Warning: No plots found. Skipping plot save.")

    print(f"✅ Saved all results for trial {trial_no} in {current_trial_dir}")


# ✅ Load Existing Model or Initialize New One
def load_or_initialize_model(input_dim):
    model = VeriVid(input_dim)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print("✅ Loaded existing model.")
    else:
        print("🆕 Training new model.")
    return model

# ✅ Compute Accuracy, Precision, Recall, F1-score Per Epoch
def compute_epoch_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return accuracy, precision, recall, f1  # ✅ Returns only computed metrics


# ✅ Plot Metrics Over Epochs
def plot_epoch_metrics(epoch_metrics):
    os.makedirs(PLOT_DIR, exist_ok=True)  # Ensure directory exists

    plt.figure(figsize=(8, 5))
    for metric, values in epoch_metrics.items():
        plt.plot(values, label=metric.capitalize())

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Model Performance Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, "epoch_metrics_plot.png"))
    plt.show()

# ✅ Plot Training vs Validation Loss
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color="blue", marker="o")
    plt.plot(val_losses, label="Validation Loss", color="red", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "loss_plot.png"))
    plt.show()


def plot_metrics(accuracy, precision, recall, f1, PLOT_DIR="./plots", dataset_name=""):
    print("accuracy:", accuracy)
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)

    plt.figure(figsize=(10, 6))
    epochs = list(range(1, len(accuracy) + 1))  # Ensure multiple epoch values

    plt.plot(epochs, accuracy, label="Accuracy", marker="o", color="blue")
    plt.plot(epochs, precision, label="Precision", marker="o", color="green")
    plt.plot(epochs, recall, label="Recall", marker="o", color="red")
    plt.plot(epochs, f1, label="F1-score", marker="o", color="purple")

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Model Performance Metrics Over Epochs")

    # ✅ Auto-adjust Y-axis scale based on min & max values
    min_y = min(min(accuracy), min(precision), min(recall), min(f1))
    max_y = max(max(accuracy), max(precision), max(recall), max(f1))
    plt.ylim(min_y - 0.05, max_y + 0.05)  # Add margin for better visualization

    plt.xticks(epochs)  # Ensure the x-axis shows all epochs
    plt.legend()
    plt.grid(True)

    # Ensure the directory exists
    os.makedirs(PLOT_DIR, exist_ok=True)  

    # Construct the final filename
    plot_filename = os.path.join(PLOT_DIR, f"Metrics_{dataset_name.replace(' ', '_')}.png")

    plt.savefig(plot_filename)
    plt.show()

    print(f"Plot saved at: {plot_filename}")


# ✅ Improved Confusion Matrix Plot
def plot_confusion_matrix(y_true, y_pred, dataset_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {dataset_name}")
    print(cm)
    plt.savefig(os.path.join(PLOT_DIR, f"confusion_matrix_{dataset_name.replace(' ', '_')}.png"))
    plt.show()

# ✅ Modify `train_and_evaluate()` to Track Metrics per Epoch
def train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, criterion):
    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    epoch_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for features, labels in tqdm(train_loader, desc=f"🚀 Epoch {epoch+1}/{EPOCHS}"):
            features, labels = features.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss, y_true_val, y_pred_val = evaluate(model, val_loader, criterion)
        train_losses.append(total_loss / len(train_loader))
        val_losses.append(val_loss)

        # ✅ Compute & Log Metrics
        accuracy, precision, recall, f1 = compute_epoch_metrics(y_true_val, y_pred_val)
        epoch_metrics["accuracy"].append(accuracy)
        epoch_metrics["precision"].append(precision)
        epoch_metrics["recall"].append(recall)
        epoch_metrics["f1"].append(f1)

        print(f"📊 Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_loss:.4f}")
        print(f"📌 Metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)

    # ✅ Plot Loss & Metrics
    plot_loss(train_losses, val_losses)
    plot_metrics(epoch_metrics["accuracy"], epoch_metrics["precision"], epoch_metrics["recall"], epoch_metrics["f1"], PLOT_DIR, "Validation_Set")

    # ✅ Plot Confusion Matrices for Validation & Test Sets
    plot_confusion_matrix(y_true_val, y_pred_val, "Validation Set")
    return model

def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss, y_true, y_pred = 0, [], []

    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device).unsqueeze(1)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = (outputs > 0.5).float().cpu().numpy()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions)

    return total_loss / len(data_loader), y_true, y_pred  # ✅ Return three values


# ✅ Evaluate on Test Set
def evaluate_test(model, test_loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device).unsqueeze(1)
            outputs = model(features)
            predictions = (outputs > 0.5).float().cpu().numpy()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"✅ Test metrics ")
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    print(f"✅ F1-score: {f1:.4f}")

    plot_confusion_matrix(y_true, y_pred, "Test Set")

# ✅ Load trial number from dataset_metrics.json
def load_and_increment_trial_no():
    """ Load the last trial number from dataset_metrics.json and increment it. """
    if os.path.exists(DATASET_METADATA_FILE):
        with open(DATASET_METADATA_FILE, "r") as f:
            metadata = json.load(f)
        trial_no = metadata.get("trial_num", 0) + 1  # Increment last trial_no
    else:
        trial_no = 1  # Default to trial 1 if no previous metadata exists

    print("current trial no:", trial_no)
    return trial_no



# ✅ Main Execution
def main():
    args = parse_args()
    update_global_paths(args.base_dir)  # Update global paths

    metadata = load_metadata()
    video_feature_map, labels = load_features(metadata)

    # ✅ Get Current and Previous Trial Paths
    trial_no = load_and_increment_trial_no()
    current_trial_dir = os.path.join(RESULTS_DIR, f"trial{trial_no}")
    previous_trial_dir = os.path.join(RESULTS_DIR, f"trial{trial_no - 1}")


    print(f"📂 Results will be saved in: {current_trial_dir}")

    # ✅ Ensure Directory Exists
    os.makedirs(current_trial_dir, exist_ok=True)
    


    # ✅ Load dataset_metrics.json from previous trial if it exists
    previous_metadata_file = os.path.join(previous_trial_dir, "dataset_metrics.json")
    if os.path.exists(previous_metadata_file):
        shutil.copy(previous_metadata_file, DATASET_METADATA_FILE)
        print(f"✅ Loaded dataset_metrics.json from {previous_trial_dir}")

    # ✅ Load Model Checkpoint from previous trial
    previous_model_path = os.path.join(previous_trial_dir, MODEL_NAME)

    if trial_no > 1 and os.path.exists(previous_model_path):
        shutil.copy(previous_model_path, MODEL_PATH)
        print(f"✅ Loaded VeriVid.pth from {previous_trial_dir}")
    else:
        print("⚠️ No previous model found. Starting fresh training.")

    # ✅ Select Train, Replay & Test Sets
    train_videos = select_train_videos(metadata, video_feature_map, 50, trial_no)
    train_videos, val_videos = train_test_split(train_videos, test_size=VAL_SPLIT, shuffle=True)
    test_videos = select_test_videos(metadata, video_feature_map)

    # ✅ Create Datasets & Dataloaders
    train_dataset = VeriVidDataset({vid: video_feature_map[vid] for vid in train_videos}, labels)
    val_dataset = VeriVidDataset({vid: video_feature_map[vid] for vid in val_videos}, labels)
    test_dataset = VeriVidDataset({vid: video_feature_map[vid] for vid in test_videos}, labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ✅ Initialize Model
    sample_features, _ = next(iter(train_loader))
    model = load_or_initialize_model(sample_features.shape[1]).to("cuda")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    # ✅ Train the Model
    model = train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, criterion)

    evaluate_test(model, test_loader)

    # ✅ Update Training Status ("yes" → "done")
    update_training_status(metadata, train_videos)
    update_training_status(metadata, val_videos)

    # ✅ Save Updated Model
    torch.save(model.state_dict(), MODEL_PATH)
    print("✅ Model saved successfully.")
    save_results(trial_no)

if __name__ == "__main__":
    main()