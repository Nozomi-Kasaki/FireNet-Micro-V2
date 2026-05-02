# Fire Image Classification Project

This repository contains a lightweight PyTorch project for 3-class fire image classification.

Classes:

- `fire`
- `no_fire`
- `start_fire`

The project includes training, parameter counting, weight-jitter search, model evaluation, and inference scripts. It is suitable for course experiments and result reproduction.

## Highlights

- Lightweight custom model: `FireNetMicroV2`
- Standard PyTorch training and validation pipeline
- Weight-jitter search for checkpoint exploration
- Simple inference script for unlabeled image folders

## Project Structure

```text
code/
|-- database/                 # Training set in ImageFolder format
|   |-- fire/
|   |-- no_fire/
|   `-- start_fire/
|-- test/                     # Validation set in ImageFolder format
|   |-- fire/
|   |-- no_fire/
|   `-- start_fire/
|-- models/
|   |-- __init__.py
|   `-- model.py              # FireNetMicroV2 definition
|-- clear.py                  # Remove duplicate images by MD5
|-- count_params.py           # Count model parameters
|-- dataset.py                # Data transforms and inference dataset
|-- infer.py                  # Inference script
|-- jitter_eval.py            # Evaluate jittered checkpoints
|-- jitter_search.py          # Generate jittered checkpoints
|-- SGLD.py                   # Experimental SGLD-style fine-tuning script
|-- train.py                  # Main training script
|-- utils.py                  # Utility functions
`-- variant_27_std0.007.pth   # Provided reference checkpoint
```

## Requirements

Recommended Python version: `3.10+`

```bash
pip install -r requirements.txt
```

If you need CUDA support, it is safer to install the correct `torch` and `torchvision` build from the official PyTorch instructions first, then install the remaining dependencies from `requirements.txt`.

## Quick Start

Clone the repository:

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Check model size:

```bash
python count_params.py
```

Train the model:

```bash
python train.py
```

Run inference on an unlabeled folder:

```bash
python infer.py --model_path variant_27_std0.007.pth --test_dir ./TestData --output_csv submission.csv
```

## Dataset Layout

The training and validation scripts use `torchvision.datasets.ImageFolder`, so the dataset must be organized by class subfolders.

Expected structure:

```text
database/
|-- fire/
|-- no_fire/
`-- start_fire/

test/
|-- fire/
|-- no_fire/
`-- start_fire/
```

Notes:

- `database/` is used as the training set
- `test/` is used as the validation set
- `infer.py` expects a separate unlabeled folder such as `TestData/`

## Training

Run the main training script:

```bash
python train.py
```

Default behavior:

- training directory: `./database`
- validation directory: `./test`
- best checkpoint: `best_c/best_fire_model.pth`

Example with explicit arguments:

```bash
python train.py --data_dir ./database --val_dir ./test --epochs 300 --batch_size 32 --lr 0.002
```

## Weight Jitter Search

Generate model variants from the best training checkpoint:

```bash
python jitter_search.py --base_model best_c/best_fire_model.pth --noise_std 0.007 --num_variants 50
```

Generated checkpoints are saved to `jitter_search_results/`.

Then evaluate them on the validation set:

```bash
python jitter_eval.py
```

This script also writes:

- `jitter_search_rank.csv`

## Inference

Run prediction on an unlabeled image folder:

```bash
python infer.py --model_path variant_27_std0.007.pth --test_dir ./TestData --output_csv submission.csv
```

Output file:

- `submission.csv`

Columns:

- `ID`: image filename
- `Label`: predicted class index

## Provided Checkpoint

The repository currently includes one reference checkpoint:

- `variant_27_std0.007.pth`

It can be used directly for inference or as a reference result for reproduction.

## Notes

- `clear.py` is only a data-cleaning helper and is not part of the main training pipeline
- `SGLD.py` is an experimental script for additional fine-tuning attempts
- Large datasets are intentionally not recommended for direct storage in the GitHub repository
