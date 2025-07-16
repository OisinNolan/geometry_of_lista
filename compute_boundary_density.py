import os
from pathlib import Path
import yaml
import numpy as np
import torch
from distutils.util import strtobool
from knot_density_analysis import knot_density_analysis
from ista import load_model

def find_run_dirs(experiment_root):
    """Recursively find all run dirs (ending in /0/, /1/, etc.) under all experiments in experiment_root."""
    run_dirs = []
    for exp in Path(experiment_root).iterdir():
        if exp.is_dir():
            for run in exp.iterdir():
                if run.is_dir() and run.name.isdigit():
                    run_dirs.append(run)
    return run_dirs

def process_run(run_path, args, nr_paths, compute_for_all_epochs, boundary_type):
    run_path = Path(run_path)
    model_name = args.model_type
    experiment_root = run_path.parent
    with open(experiment_root / "config.yaml", 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    datasets = {
        'test': torch.load(run_path / "data/test_data.tar"),
        'train': torch.load(run_path / "data/train_data.tar"),
    }
    print(f"Running for {model_name} at {run_path}")
    if compute_for_all_epochs:
        epoch_dir = run_path / model_name / "epoch_boundary_densities"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        import glob
        pattern = str(run_path / f"{model_name}/{model_name}_state_dict_epoch_*.tar")
        epoch_files = sorted(glob.glob(pattern))
        print(f"Found {len(epoch_files)} epoch state dicts.")
        for epoch_file in epoch_files:
            epoch_num = epoch_file.split("_epoch_")[-1].split(".tar")[0]
            print(f"Processing epoch {epoch_num}...")
            model = load_model(
                config, model_name, epoch_file, run_path / "A.tar",
                train_dataset=datasets["train"], experiment_run_path=run_path
            )
            knot_density_array, anchor_points = knot_density_analysis(
                model, model.nr_folds, model.A, nr_paths=nr_paths,
                nr_points_along_path=config["Path"]["nr_points_along_path"],
                path_delta=config["Path"]["path_delta"],
                specified_anchors=datasets[args.anchor_on].y,
                save_folder=epoch_dir, save_name=f"knot_density_{model_name}_epoch_{epoch_num}",
                verbose=True, tqdm_position=1, boundary_type=boundary_type
            )
            torch.save(
                torch.mean(knot_density_array, dim=0),
                epoch_dir / f"{boundary_type}_boundary_density_epoch_{epoch_num}.tar"
            )
    else:
        model = load_model(
            config, model_name, run_path / f"{model_name}/{model_name}_state_dict.tar",
            run_path / "A.tar", train_dataset=datasets["train"], experiment_run_path=run_path
        )
        knot_density_array, anchor_points = knot_density_analysis(
            model, model.nr_folds, model.A, nr_paths=nr_paths,
            nr_points_along_path=config["Path"]["nr_points_along_path"],
            path_delta=config["Path"]["path_delta"],
            specified_anchors=datasets[args.anchor_on].y,
            save_folder=".", save_name=f"knot_density_{model_name}",
            verbose=True, tqdm_position=1, boundary_type=boundary_type
        )
        torch.save(
            torch.mean(knot_density_array, dim=0),
            run_path / model_name / f"{boundary_type}_boundary_density.tar"
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, default=None,
                      help="Path to experiment run directory")
    parser.add_argument("--experiment_root", type=str, default=None,
                      help="Path to experiment root directory containing multiple experiments/runs")
    parser.add_argument("--anchor_on", type=str, default="train", choices=['train', 'test'],
                      help="Whether to anchor on samples from the train or test set")
    parser.add_argument("--model_type", type=str, nargs='+', default=["LISTA"], choices=['LISTA', 'ISTA'],
                      help="Model type(s) to process (can be a list)")
    parser.add_argument("--boundary_type", type=str, nargs='+', default=["support", "identity"], choices=['support', 'identity'],
                      help="Boundary type(s) to process (can be a list). 'Support' will calculate the decision density," \
                      "and 'identity' will calculate the knot density. ")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    NR_PATHS = 10
    COMPUTE_FOR_ALL_EPOCHS = False

    if args.experiment_root and args.run_path:
        raise UserWarning("Please choose one or the other!")
    if args.experiment_root:
        run_dirs = find_run_dirs(args.experiment_root)
        print(f"Found {len(run_dirs)} run directories in {args.experiment_root}")
        for run_dir in run_dirs:
            for model_type in args.model_type:
                for boundary_type in args.boundary_type:
                    try:
                        # Create a copy of args with updated model_type and boundary_type
                        class ArgsCopy:
                            pass
                        args_copy = ArgsCopy()
                        for k, v in vars(args).items():
                            setattr(args_copy, k, v)
                        args_copy.model_type = model_type
                        args_copy.boundary_type = boundary_type
                        process_run(run_dir, args_copy, NR_PATHS, COMPUTE_FOR_ALL_EPOCHS, boundary_type)
                    except Exception as e:
                        print(f"Failed to process {run_dir} for model {model_type} and boundary {boundary_type}: {e}")
    elif args.run_path:
        for model_type in args.model_type:
            for boundary_type in args.boundary_type:
                # Create a copy of args with updated model_type and boundary_type
                class ArgsCopy:
                    pass
                args_copy = ArgsCopy()
                for k, v in vars(args).items():
                    setattr(args_copy, k, v)
                args_copy.model_type = model_type
                args_copy.boundary_type = boundary_type
                process_run(args.run_path, args_copy, NR_PATHS, COMPUTE_FOR_ALL_EPOCHS, boundary_type)
    else:
        raise ValueError("You must provide either --run_path or --experiment_root")