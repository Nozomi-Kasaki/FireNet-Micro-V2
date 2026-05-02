import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TestDataset, get_val_transforms
from models.model import FireNetMicroV2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="variant_27_std0.007.pth")
    parser.add_argument(
        "--test_dir",
        type=str,
        default="./TestData",
        help="Folder containing unlabeled test images.",
    )
    parser.add_argument("--output_csv", type=str, default="submission.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def predict_tta(model, inputs):
    pred_orig = model(inputs)
    pred_flip = model(torch.flip(inputs, dims=[3]))
    return (pred_orig + pred_flip) / 2.0


def main():
    args = parse_args()

    model = FireNetMicroV2(num_classes=3)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    test_dataset = TestDataset(args.test_dir, transform=get_val_transforms())
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    results = []

    print("Starting inference with TTA...")
    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader):
            inputs = inputs.to(device)
            outputs = predict_tta(model, inputs)
            _, predicted = torch.max(outputs, 1)

            for fname, pred in zip(filenames, predicted):
                results.append({"ID": fname, "Label": pred.item()})

    df = pd.DataFrame(results)[["ID", "Label"]]
    df.sort_values(by="ID", inplace=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Done! Result saved to {args.output_csv}")


if __name__ == "__main__":
    main()
