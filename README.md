# Fire Image Classification Project

This repository contains a lightweight PyTorch project for 3-class fire image classification.

Classes:

- `fire`
- `no_fire`
- `start_fire`

The project includes training, parameter counting, weight-jitter search, model evaluation, and inference scripts. It is suitable for course experiments and result reproduction.

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

## Dataset Layout

The training and validation scripts use `torchvision.datasets.ImageFolder`, so the dataset must be organized by class subfolders.

- `database/`: training set
- `test/`: validation set

Notes:

- In this repository, `test/` is a validation set, not an unlabeled competition test set.
- If you want to run `infer.py`, prepare a separate unlabeled image folder such as `TestData/` and pass it with `--test_dir`.
- `clear.py` is only a data-cleaning helper and is not part of the main training pipeline.

## Main Scripts

### 1. Count Parameters

```bash
python count_params.py
```

### 2. Train

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

### 3. Weight Jitter Search

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

### 4. Inference

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

## GitHub Publishing Notes

It is not recommended to commit the full datasets or compressed dataset archives to GitHub:

- `database/`
- `test/`
- `database.zip`
- `test.zip`

Reasons:

- they make the repository unnecessarily large
- GitHub has file size limits
- dataset download links or separate instructions are usually better than storing raw data in the repo

This repository now includes a `.gitignore` file that ignores large datasets and generated training outputs by default.

## What Was Corrected

This README was updated to match the current codebase:

- the model definition file is `models/model.py`
- the main training script saves the best checkpoint to `best_c/best_fire_model.pth`
- `infer.py` now accepts command line arguments for checkpoint path, test directory, and output path
- `dataset.py` now supports optional image-size arguments for compatibility with `SGLD.py`

## Possible Future Improvements

- add dataset source information
- add experiment results and accuracy tables
- add sample prediction images
- add a short training log or ablation summary
