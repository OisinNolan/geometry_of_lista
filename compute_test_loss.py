import os
from pathlib import Path
import yaml
import numpy as np
import torch
from distutils.util import strtobool
from training import get_loss_on_dataset_over_folds
from ista import load_model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", type=str,
                        help="Path to root directory containing multiple experiment directories")
    parser.add_argument("--model_type", type=str, nargs='+', default=["LISTA"], choices=['LISTA', 'ISTA'],
                        help="Model type(s) to process (can be a list)")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    EXPERIMENT_ROOT = Path(args.experiment_root)

    # Iterate through all experiment directories
    for experiment_dir in EXPERIMENT_ROOT.iterdir():
        if not experiment_dir.is_dir():
            continue
        config_path = experiment_dir / "config.yaml"
        if not config_path.exists():
            print(f"Skipping {experiment_dir}: config.yaml not found.")
            continue

        with open(config_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        print(f"Running for experiment: {experiment_dir.name}")

        # Iterate through all run directories
        for run_dir in experiment_dir.iterdir():
            if not run_dir.is_dir():
                continue
            print(f"Running for {run_dir}")

            data_dir = run_dir / "data"
            if not (data_dir / "test_data.tar").exists() or not (data_dir / "train_data.tar").exists():
                print(f"Skipping {run_dir}: test/train data not found.")
                continue

            datasets = {
                'test': torch.load(data_dir / "test_data.tar"),
                'train': torch.load(data_dir / "train_data.tar"),
            }

            for model_type in args.model_type:
                model_path = run_dir / f"{model_type}/{model_type}_state_dict.tar"
                A_path = run_dir / "A.tar"

                model = load_model(config, model_type, model_path, A_path,
                                  train_dataset=datasets["train"], experiment_run_path=run_dir)

                test_loss_l1 = get_loss_on_dataset_over_folds(model, datasets["test"], l1_weight=1.0, l2_weight=0.0)
                test_loss_l2 = get_loss_on_dataset_over_folds(model, datasets["test"], l1_weight=0.0, l2_weight=1.0)

                output_dir = run_dir / model_type
                output_dir.mkdir(exist_ok=True)
                torch.save(test_loss_l1, output_dir / "test_loss_l1.tar")
                torch.save(test_loss_l2, output_dir / "test_loss_l2.tar")