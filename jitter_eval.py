import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from dataset import get_val_transforms
from models.model import FireNetMicroV2


JITTER_DIR = "jitter_search_results"
VAL_DATA_DIR = "./test"
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_one_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def main():
    print(f"Checking models in: {JITTER_DIR}")

    val_dataset = datasets.ImageFolder(VAL_DATA_DIR, transform=get_val_transforms())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    if not os.path.exists(JITTER_DIR):
        print(f"Directory does not exist: {JITTER_DIR}")
        return

    model_files = [f for f in os.listdir(JITTER_DIR) if f.endswith(".pth")]
    if not model_files:
        print("No .pth model files were found.")
        return

    model = FireNetMicroV2(num_classes=3).to(DEVICE)
    results = []

    print(f"Starting evaluation for {len(model_files)} models...\n")
    for filename in tqdm(model_files, desc="Evaluating"):
        filepath = os.path.join(JITTER_DIR, filename)
        try:
            checkpoint = torch.load(filepath, map_location=DEVICE)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)

            acc = validate_one_model(model, val_loader)
            results.append({"Model": filename, "Val Accuracy": acc})
        except Exception as exc:
            print(f"Failed to evaluate {filename}: {exc}")

    if not results:
        print("No models were evaluated successfully.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values(by="Val Accuracy", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 40)
    print("Top Models")
    print("=" * 40)
    print(df.to_string(index=False))

    df.to_csv("jitter_search_rank.csv", index=False)
    best_acc = df.iloc[0]["Val Accuracy"]
    best_name = df.iloc[0]["Model"]

    print(f"\nResults saved to: jitter_search_rank.csv")
    print(f"Best model: {best_name} (Acc: {best_acc:.2f}%)")


if __name__ == "__main__":
    main()
