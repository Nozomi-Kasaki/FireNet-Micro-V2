import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataset import get_val_transforms
from models.model import FireNetMicroV2


CLASS_TO_IDX = {"fire": 0, "no_fire": 1, "start_fire": 2}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a 3-class FireNet model on the public D-Fire benchmark."
    )
    parser.add_argument(
        "--benchmark_root",
        type=str,
        default=None,
        help="Root directory of the downloaded D-Fire dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "valid", "validation", "test"],
        help="Dataset split to evaluate when benchmark_root is provided.",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Path to the image directory for one split. Overrides benchmark_root auto-detection.",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default=None,
        help="Path to the YOLO label directory for one split. Overrides benchmark_root auto-detection.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="variant_27_std0.007.pth",
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument(
        "--both_policy",
        type=str,
        default="fire",
        choices=["fire", "start_fire", "skip"],
        help="How to map D-Fire images that contain both fire and smoke.",
    )
    parser.add_argument(
        "--fire_class_id",
        type=int,
        default=0,
        help="Fire class id in D-Fire YOLO labels.",
    )
    parser.add_argument(
        "--smoke_class_id",
        type=int,
        default=1,
        help="Smoke class id in D-Fire YOLO labels.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Use horizontal-flip test-time augmentation.",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default=None,
        help="Optional path to save per-image predictions as CSV.",
    )
    return parser.parse_args()


def resolve_split_dirs(args):
    if args.images_dir and args.labels_dir:
        return Path(args.images_dir), Path(args.labels_dir)

    if not args.benchmark_root:
        raise ValueError("Provide either --benchmark_root or both --images_dir and --labels_dir.")

    root = Path(args.benchmark_root)
    split = args.split
    candidates = [
        (root / split / "images", root / split / "labels"),
        (root / "images" / split, root / "labels" / split),
        (root / split, root / "labels" / split),
    ]

    for images_dir, labels_dir in candidates:
        if images_dir.exists() and labels_dir.exists():
            return images_dir, labels_dir

    raise FileNotFoundError(
        "Could not auto-detect D-Fire split directories. "
        "Please pass --images_dir and --labels_dir explicitly."
    )


def map_dfire_label(label_path, fire_class_id, smoke_class_id, both_policy):
    if not label_path.exists():
        return CLASS_TO_IDX["no_fire"]

    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return CLASS_TO_IDX["no_fire"]

    class_ids = set()
    for line in lines:
        parts = line.split()
        try:
            class_ids.add(int(float(parts[0])))
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid YOLO label line in {label_path}: {line}") from exc

    has_fire = fire_class_id in class_ids
    has_smoke = smoke_class_id in class_ids

    if has_fire and has_smoke:
        if both_policy == "skip":
            return None
        if both_policy == "start_fire":
            return CLASS_TO_IDX["start_fire"]
        return CLASS_TO_IDX["fire"]
    if has_fire:
        return CLASS_TO_IDX["fire"]
    if has_smoke:
        return CLASS_TO_IDX["start_fire"]
    return CLASS_TO_IDX["no_fire"]


class DFireThreeClassDataset(Dataset):
    def __init__(
        self,
        images_dir,
        labels_dir,
        transform,
        fire_class_id=0,
        smoke_class_id=1,
        both_policy="fire",
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.samples = []

        for image_path in sorted(self.images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS or not image_path.is_file():
                continue
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            target = map_dfire_label(
                label_path,
                fire_class_id=fire_class_id,
                smoke_class_id=smoke_class_id,
                both_policy=both_policy,
            )
            if target is None:
                continue
            self.samples.append((image_path, target))

        if not self.samples:
            raise RuntimeError("No evaluable samples were found. Check the D-Fire paths and label format.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, target = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, target, image_path.name


def predict_with_optional_tta(model, inputs, use_tta=False):
    outputs = model(inputs)
    if isinstance(outputs, tuple):
        outputs = outputs[0]

    if not use_tta:
        return outputs

    flipped = torch.flip(inputs, dims=[3])
    outputs_flip = model(flipped)
    if isinstance(outputs_flip, tuple):
        outputs_flip = outputs_flip[0]
    return (outputs + outputs_flip) / 2.0


def compute_metrics(confusion):
    metrics = {}
    total = confusion.sum()
    correct = np.trace(confusion)
    accuracy = correct / total if total else 0.0

    precisions = []
    recalls = []
    f1s = []

    for class_idx, class_name in IDX_TO_CLASS.items():
        tp = confusion[class_idx, class_idx]
        fp = confusion[:, class_idx].sum() - tp
        fn = confusion[class_idx, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[class_idx, :].sum()),
        }

    metrics["overall"] = {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "samples": int(total),
    }
    return metrics


def print_report(metrics, confusion):
    overall = metrics["overall"]
    print("\n=== D-Fire 3-Class Evaluation ===")
    print(f"Samples:           {overall['samples']}")
    print(f"Accuracy:          {overall['accuracy'] * 100:.2f}%")
    print(f"Macro Precision:   {overall['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall:      {overall['macro_recall'] * 100:.2f}%")
    print(f"Macro F1:          {overall['macro_f1'] * 100:.2f}%")

    print("\nPer-class metrics:")
    for class_idx in range(len(IDX_TO_CLASS)):
        class_name = IDX_TO_CLASS[class_idx]
        item = metrics[class_name]
        print(
            f"- {class_name:10s} "
            f"precision={item['precision'] * 100:6.2f}% "
            f"recall={item['recall'] * 100:6.2f}% "
            f"f1={item['f1'] * 100:6.2f}% "
            f"support={item['support']}"
        )

    print("\nConfusion matrix (rows=true, cols=pred):")
    header = "            " + " ".join(f"{IDX_TO_CLASS[idx]:>10s}" for idx in range(len(IDX_TO_CLASS)))
    print(header)
    for row_idx in range(len(IDX_TO_CLASS)):
        row_name = IDX_TO_CLASS[row_idx]
        row_values = " ".join(f"{confusion[row_idx, col_idx]:10d}" for col_idx in range(len(IDX_TO_CLASS)))
        print(f"{row_name:>10s} {row_values}")


def save_predictions_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "target_idx", "target_name", "pred_idx", "pred_name"])
        writer.writerows(rows)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images_dir, labels_dir = resolve_split_dirs(args)
    print(f"Images dir: {images_dir}")
    print(f"Labels dir: {labels_dir}")
    print(f"Both-label policy: {args.both_policy}")

    dataset = DFireThreeClassDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        transform=get_val_transforms(args.img_size),
        fire_class_id=args.fire_class_id,
        smoke_class_id=args.smoke_class_id,
        both_policy=args.both_policy,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = FireNetMicroV2(num_classes=3)
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    confusion = np.zeros((3, 3), dtype=np.int64)
    prediction_rows = []

    with torch.no_grad():
        for inputs, targets, filenames in dataloader:
            inputs = inputs.to(device)
            outputs = predict_with_optional_tta(model, inputs, use_tta=args.tta)
            preds = outputs.argmax(dim=1).cpu().numpy()
            targets = targets.numpy()

            for filename, target, pred in zip(filenames, targets, preds):
                confusion[target, pred] += 1
                prediction_rows.append(
                    [filename, int(target), IDX_TO_CLASS[int(target)], int(pred), IDX_TO_CLASS[int(pred)]]
                )

    metrics = compute_metrics(confusion)
    print_report(metrics, confusion)

    if args.save_csv:
        save_predictions_csv(args.save_csv, prediction_rows)
        print(f"\nSaved per-image predictions to: {args.save_csv}")


if __name__ == "__main__":
    main()
