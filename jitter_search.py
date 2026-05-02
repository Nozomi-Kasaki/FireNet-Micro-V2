import argparse
import copy
import os
import sys

import torch

from models.model import FireNetMicroV2


parser = argparse.ArgumentParser()
parser.add_argument(
    "--base_model",
    type=str,
    default="best_c/best_fire_model.pth",
    help="Path to the base model checkpoint.",
)
parser.add_argument("--save_dir", type=str, default="jitter_search_results")
parser.add_argument("--num_variants", type=int, default=50, help="Number of variants to generate")
parser.add_argument("--noise_std", type=float, default=7e-3, help="Gaussian noise standard deviation")
args = parser.parse_args()


def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Loading base model: {args.base_model}")
    print(f"Generating {args.num_variants} variants with noise_std={args.noise_std}...")

    base_model = FireNetMicroV2(num_classes=3)
    checkpoint = torch.load(args.base_model, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        base_model.load_state_dict(checkpoint["state_dict"])
    else:
        base_model.load_state_dict(checkpoint)

    print("Base model loaded. Starting jitter generation...")

    for i in range(args.num_variants):
        variant_model = copy.deepcopy(base_model)

        with torch.no_grad():
            for param in variant_model.parameters():
                param.add_(torch.randn_like(param) * args.noise_std)

        save_name = f"variant_{i + 1}_std{args.noise_std}.pth"
        save_path = os.path.join(args.save_dir, save_name)
        torch.save(variant_model.state_dict(), save_path)

        sys.stdout.write(f"\rGenerated {i + 1}/{args.num_variants}: {save_path}")
        sys.stdout.flush()

    print(f"\n\nDone! {args.num_variants} models saved to {args.save_dir}.")


if __name__ == "__main__":
    main()
