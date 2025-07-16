"""
This file creates a large experiment to test the knot density of (R)(L)ISTA in different conditions.
"""

# %% imports
# standard library imports
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
import yaml
import shutil
import time
import pandas as pd
import argparse
import uuid
from pathlib import Path
from distutils.util import strtobool

# local imports
import ista
from experiment_design import sample_experiment
from knot_density_analysis import knot_density_analysis
from hyper_plane_analysis  import visual_analysis_of_ista
from data import create_train_validation_test_datasets
from training import grid_search_ista, train_lista, get_loss_on_dataset_over_folds, get_support_accuracy_on_dataset_over_folds


def parse_args():
    parser = argparse.ArgumentParser(description="Main experiment arguments")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        # default="/app/configs/main_experiments/estimation_error/8_16_64.yaml",
        default=None,
        help="Path to the experiment config file.",
    )
    parser.add_argument(
        "-m",
        "--model_types",
        nargs='+',
        default=["ISTA", "LISTA"],
        help="Specify which set of algorithms to run.",
    )
    parser.add_argument(
        "--sweep_L2",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to run an L2 regularization weight sweep",
    )
    parser.add_argument(
        "--sweep_jacobian",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to run a jacobian regularization weight sweep",
    )
    parser.add_argument(
        "--sweep_num_folds",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to run a num folds sweep",
    )
    parser.add_argument(
        "--sweep_noise_stds",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to run a noise std sweep",
    )
    parser.add_argument(
        "--sweep_ranks",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to run a rank sweep",
    )
    parser.add_argument(
        "--sweep_M",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to run an M sweep",
    )
    parser.add_argument(
        "--loss_on_all_folds",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to compute a loss on all folds during training",
    )
    parser.add_argument(
        "--reconstruction_loss",
        type=str,
        default="l2",
        choices=["l1", "l2"]
    )
    parser.add_argument('--crop_train_samples', type=int, default=None,
                      help='If set, crop training dataset to first N samples')
    parser.add_argument('--results_dir', type=str, default="./output",
                      help='Override results directory specified in config')
    parser.add_argument(
        "--compute_knot_density",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to compute knot density (default: True)",
    )
    parser.add_argument(
        "--visualize_hyperplane",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to compute knot density (default: True)",
    )
    return parser.parse_args()
args = parse_args()

colors = {
    "ISTA": "tab:blue", 
    "FISTA": "tab:red",
    "LISTA": "tab:orange", 
    "ToeplitzLISTA": "tab:olive"
}

def load_experiment_data(experiment_dir, experiment_id):
    """Load data from a specific experiment run directory"""
    data_dir = os.path.join(experiment_dir, str(experiment_id), "data")
    train_data = torch.load(os.path.join(data_dir, "train_data.tar"))
    validation_data = torch.load(os.path.join(data_dir, "validation_data.tar"))
    test_data = torch.load(os.path.join(data_dir, "test_data.tar")) 
    return train_data, validation_data, test_data

def get_experiment_params(experiment_dir, experiment_id):
    """Load experiment parameters from a specific experiment run"""
    params_path = os.path.join(experiment_dir, str(experiment_id), "parameters.yaml")
    with open(params_path, 'r') as file:
        return yaml.safe_load(file)

def print_experiment_summary(config, args, experiment_id, M, N, K, using_existing=False):
    """Print formatted summary of experiment configuration and data settings"""
    border = "=" * 80
    print(f"\n{border}")
    print(f"Experiment {experiment_id+1}/{config['max_nr_of_experiments']}")
    print(f"{'Using existing data' if using_existing else 'Creating new data'}")
    print("-" * 40)
    print(f"Matrix dims (M×N): {M}×{N}")
    print(f"Sparsity (K): {K}")
    
    train_samples = config['data_that_stays_constant']['nr_training_samples']
    if args.crop_train_samples:
        train_samples = min(train_samples, args.crop_train_samples)
        print(f"Training samples: {train_samples} (cropped from {config['data_that_stays_constant']['nr_training_samples']})")
    else:
        print(f"Training samples: {train_samples}")
        
    print(f"Validation samples: {config['data_that_stays_constant']['nr_validation_samples']}")
    print(f"Test samples: {config['data_that_stays_constant']['nr_test_samples']}")
    print(f"Noise std: {config['data_that_stays_constant']['noise_std']}")
    print(f"Loss type:  L2: {config['LISTA']['l2_weight']},  L1: {config['LISTA']['l1_weight']}")
    print(f"{border}\n")

def run_experiment(config_path, config, model_types, plot=False, save=True, output_identifier=""):
    nr_of_model_types = len(model_types) # the number of models we are comparing, ISTA, LISTA
    config_file_name = os.path.splitext(os.path.basename(Path(config_path)))[0]
    # Override results_dir if specified in args
    if args.results_dir is not None:
        config["results_dir"] = args.results_dir
    # create the directory to save the results, check first if it already exists, if so stop, and query the user if it should be overwritten
    results_dir_with_parent = os.path.join(config["results_dir"], config_file_name+"_"+output_identifier+"_"+str(uuid.uuid4())[:4])
    if os.path.exists(results_dir_with_parent):
        print(f"\nThis results directory already exists: {config['results_dir']}")
        print("Do you want to overwrite it? (y/n)")	
        answer = input()
        if answer == "y":
            shutil.rmtree(results_dir_with_parent, ignore_errors=True) # remove the directory and its contents
            time.sleep(1) # wait for the directory to be deleted
            os.makedirs(results_dir_with_parent, exist_ok=True)
        else:
            raise FileExistsError(f"The results directory {config['results_dir']} already exists.")
    else:
        os.makedirs(results_dir_with_parent)

    print(f"ℹ️ Results will be saved in {results_dir_with_parent}")

    # save the configuration file to the results directory
    with open(os.path.join(results_dir_with_parent, "config.yaml"), 'w') as file:
        yaml.dump(config, file)

    # %% loop over the experiments to run
    print("\nStarting the experiments")

    # inialize lists to store the results, to show them together in the end
    knot_density_over_experiments  = [[] for _ in range(nr_of_model_types)]
    named_knot_density_over_experiments  = {model_type: [] for model_type in model_types}
    test_loss_over_experiments     = [[] for _ in range(nr_of_model_types)]
    named_test_loss_over_experiments     = {model_type: [] for model_type in model_types}
    named_train_loss_over_experiments     = {model_type: [] for model_type in model_types}
    test_accuracy_over_experiments = [[] for _ in range(nr_of_model_types)]
    train_accuracy_over_experiments = [[] for _ in range(nr_of_model_types)]
    train_loss_over_experiments     = [[] for _ in range(nr_of_model_types)]

    df = pd.DataFrame()

    # loop
    for experiment_id in tqdm(range(config["max_nr_of_experiments"]), position=0, desc="running experiments", leave=True):
        tqdm_leave = False if experiment_id < config["max_nr_of_experiments"]-1 else True # set tqdm_leave to False if this is not the last experiment
        
        # create the directory for the experiment
        results_dir_this_experiment = os.path.join(results_dir_with_parent, str(experiment_id))
        os.makedirs(results_dir_this_experiment, exist_ok=True)

        M, N, K, A = sample_experiment(config)

        print_experiment_summary(config, args, experiment_id, M, N, K)
        # save the parameters of the experiment in a .yaml file in the experiment folder/str(experiment_id)
        with open(os.path.join(results_dir_this_experiment, "parameters.yaml"), 'w') as file:
            yaml.dump({"M": M, "N": N, "K": K}, file)

        if save:
            # save the A matrix in a .tar file
            torch.save(A, os.path.join(results_dir_this_experiment, "A.tar"))

        train_data, validation_data, test_data = create_train_validation_test_datasets(A, maximum_sparsity = K, x_magnitude=config["data_that_stays_constant"]["x_magnitude"], 
                                                                            N=N, noise_std = config["data_that_stays_constant"]["noise_std"],
                                                                            nr_of_examples_train = config["data_that_stays_constant"]["nr_training_samples"],
                                                                            nr_of_examples_validation = config["data_that_stays_constant"]["nr_validation_samples"],
                                                                            nr_of_examples_test = config["data_that_stays_constant"]["nr_test_samples"],
                                                                            test_magnitude_shift_epsilon = config["data_that_stays_constant"]["test_distribution_shift_epsilon"])
        
        if args.crop_train_samples is not None:
            train_data.x = train_data.x[:args.crop_train_samples]
            train_data.y = train_data.y[:args.crop_train_samples]
            train_data.nr_of_examples = args.crop_train_samples
            config['data_that_stays_constant']['nr_training_samples'] = args.crop_train_samples
            # validation_data.x = validation_data.x[:args.crop_train_samples]
            # validation_data.y = validation_data.y[:args.crop_train_samples]

        if save:
            # save the data in .tar files
            results_dir_this_experiment_data = os.path.join(results_dir_with_parent, str(experiment_id), "data")
            os.makedirs(results_dir_this_experiment_data, exist_ok=True)
            torch.save(train_data,      os.path.join(results_dir_this_experiment_data, "train_data.tar"))
            torch.save(validation_data, os.path.join(results_dir_this_experiment_data, "validation_data.tar"))
            torch.save(test_data,       os.path.join(results_dir_this_experiment_data, "test_data.tar"))

        # %% loop over each model type
        for model_idx, model_type in enumerate(model_types):
            config["ToeplitzLISTA"] = config["LISTA"]
            # get the model config for this model type
            model_config = config[model_type]

            # create a directory for this model type
            model_folder = os.path.join(results_dir_this_experiment, model_type)
            os.makedirs(model_folder, exist_ok=True)
                
            # check if this is ISTA or FISTA, in which case the parameters are found by grid search
            if model_type == "ISTA" or model_type == "FISTA":
                model_class = ista.ISTA if model_type == "ISTA" else ista.FISTA
                # create the ISTA/FISTA model with mu=0 and lambda=0
                model = model_class(A, mu = 0, _lambda = 0, nr_folds = model_config["nr_folds"], device = config["device"])

                # perform grid search on ISTA for the best lambda and mu for these parameters
                scaling_factors = [1.0, 0.75, 0.5, 0.25, 0.1] # Try progressively smaller data portions

                for scale in scaling_factors:
                    try:
                        # Create scaled subsets of the data
                        train_subset = train_data[:int(len(train_data) * scale)]
                        val_subset = validation_data[:int(len(validation_data) * scale)]
                        
                        print(f"Attempting grid search with {scale*100}% of data...")
                        model, mu, _lambda, losses, tested_mus, tested_lambdas = grid_search_ista(model, train_subset, val_subset, model_config, tqdm_position=1, verbose=True, tqdm_leave=tqdm_leave)
                        # If we get here, it succeeded
                        print(f"Successfully completed with {scale*100}% of data")
                        break
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"Out of memory with {scale*100}% of data, trying smaller portion...")
                            continue
                        else:
                            # Re-raise if it's not an OOM error
                            raise e
                else:
                    # This runs if we exhaust all scaling factors without success
                    raise RuntimeError("Unable to run grid search - out of memory even with minimum data size")
                
                # save the results of the grid search in the results directroy in a .yaml file
                with open(os.path.join(model_folder, "best_mu_and_lambda.yaml"), 'a') as file:
                    yaml.dump({"mu": mu.cpu().item(), "lambda": _lambda.cpu().item()}, file)   

                # put the losses in a .csv file, with the tested mus and lambdas as the rows and columns
                loss_df = pd.DataFrame(losses, index=tested_mus, columns=tested_lambdas)
                loss_df.to_csv(os.path.join(model_folder, "losses.csv"))


            # otherwise, the model is LISTA, and needs to be trained
            else:
                lista_class = ista.ToeplitzLISTA if model_type == "ToeplitzLista" else ista.LISTA
                # create the model using the parameters in the config file
                model = lista_class(A, mu = model_config["initial_mu"], _lambda = model_config["initial_lambda"], nr_folds = model_config["nr_folds"], 
                                device = config["device"], initialize_randomly = False, share_weights=model_config["share_weights"], train_inputs = torch.stack([y for (y, _) in train_data]))
                regularize = "regularization" in fixed_config[model_type].keys()
                model, train_losses, val_losses  =  train_lista(model, train_data, validation_data, model_config,show_loss_plot = False,
                                                                loss_folder = model_folder, save_name = model_type, regularize = regularize,
                                                                tqdm_position=1, verbose=True, tqdm_leave=tqdm_leave, save=save)
                
                    
            # Additional knot density analyses with specified anchors
            knot_density_train = None
            knot_density_test = None
            
            if args.compute_knot_density:
                if "specified_anchors" in config["Path"]:
                    if "train" in config["Path"]["specified_anchors"]:
                        print("Computing knot density on train set")
                        knot_density_train_array, _ = knot_density_analysis(model, model_config["nr_folds"], A, nr_paths = config["Path"]["nr_paths"], anchor_point_std = config["Path"]["anchor_point_std"],
                                                    nr_points_along_path=config["Path"]["nr_points_along_path"], path_delta=config["Path"]["path_delta"], specified_anchors=train_data.y, save_folder = model_folder,
                                                    save_name = "knot_density_"+ model_type, verbose = True, color = colors[model_type], tqdm_position=1, tqdm_leave=tqdm_leave)
                
                        knot_density_train = torch.mean(knot_density_train_array, dim=0)
                        
                    if "test" in config["Path"]["specified_anchors"]:
                        print("Computing knot density on test set")
                        knot_density_test_array, _ = knot_density_analysis(model, model_config["nr_folds"], A, nr_paths = config["Path"]["nr_paths"], anchor_point_std = config["Path"]["anchor_point_std"],
                                                    nr_points_along_path=config["Path"]["nr_points_along_path"], path_delta=config["Path"]["path_delta"], specified_anchors=test_data.y, save_folder = model_folder,
                                                    save_name = "knot_density_"+ model_type, verbose = True, color = colors[model_type], tqdm_position=1, tqdm_leave=tqdm_leave)
                
                        knot_density_test = torch.mean(knot_density_test_array, dim=0)
                else: 
                    knot_density_array, _ = knot_density_analysis(model, model_config["nr_folds"], A, nr_paths = config["Path"]["nr_paths"], anchor_point_std = config["Path"]["anchor_point_std"],
                                                    nr_points_along_path=config["Path"]["nr_points_along_path"], path_delta=config["Path"]["path_delta"], save_folder = model_folder,
                                                    save_name = "knot_density_"+ model_type, verbose = True, color = colors[model_type], tqdm_position=1, tqdm_leave=tqdm_leave)
                
                    knot_density = torch.mean(knot_density_array, dim=0)


                
                # Save train & test knot densities
                pd.DataFrame([{
                    "knot_density_train": knot_density_train,
                    "knot_density_test": knot_density_test,
                }]).to_csv(os.path.join(model_folder, "knot_densities.csv"), index=False)
                # default to knot_density_train for rest of script
                knot_density = knot_density_train

                if save:
                    # save the knot densities in a .tar file and to the lists
                    torch.save(knot_density, os.path.join(model_folder, "knot_density.tar"))
                knot_density_over_experiments[model_idx].append(knot_density)
                named_knot_density_over_experiments[model_type].append(knot_density)


            # evaluate the model on the test set
            test_loss = get_loss_on_dataset_over_folds(model, test_data, l1_weight=0.0, l2_weight=1.0)
            test_accuracy = get_support_accuracy_on_dataset_over_folds(model, test_data)
            
            train_accuracy = get_support_accuracy_on_dataset_over_folds(model, train_data)
            train_accuracy_over_experiments[model_idx].append(train_accuracy)

            # save the test loss in a .tar file and to the lists
            test_loss_over_experiments[model_idx].append(test_loss)
            named_test_loss_over_experiments[model_type].append(test_loss)
            test_accuracy_over_experiments[model_idx].append(test_accuracy)
            
            train_loss = get_loss_on_dataset_over_folds(model, train_data, l1_weight=0.0, l2_weight=1.0)
            named_train_loss_over_experiments[model_type].append(train_loss)
            train_loss_over_experiments[model_idx].append(train_loss)
            if save:
                torch.save(test_loss, os.path.join(model_folder, "test_loss.tar"))
                torch.save(test_accuracy, os.path.join(model_folder, "test_accuracy.tar"))

            if args.visualize_hyperplane:
                # visualize the results in a 2D plane
                hyperplane_config = config["Hyperplane"]
                if (hyperplane_config["enabled"]):
                    hyperplane_folder_norm           = os.path.join(model_folder, "hyperplane","norm")  
                    hyperplane_folder_jacobian_label = os.path.join(model_folder, "hyperplane","jacobian_label")
                    hyperplane_folder_jacobian_pca = os.path.join(model_folder, "hyperplane","jacobian_pca")

                    visual_analysis_of_ista(model, model_config, hyperplane_config, A, save_folder = hyperplane_folder_norm,           tqdm_position=1, tqdm_leave= tqdm_leave, verbose = True, color_by="norm", folds_to_visualize=[0, 1, 5, 9])
                    visual_analysis_of_ista(model, model_config, hyperplane_config, A, save_folder = hyperplane_folder_jacobian_label, tqdm_position=1, tqdm_leave= tqdm_leave, verbose = True, color_by="jacobian_label", folds_to_visualize=[0, 1, 5, 9])
                    visual_analysis_of_ista(model, model_config, hyperplane_config, A, save_folder = hyperplane_folder_jacobian_pca   , tqdm_position=1, tqdm_leave= tqdm_leave, verbose = True, color_by="jacobian_pca", folds_to_visualize=[0, 1, 5, 9])

            # === Profiling ===
            train_time = None
            avg_train_time_per_epoch = None
            inference_time = None
            avg_inference_time_per_sample = None

            if model_type in ["LISTA", "ToeplitzLISTA"]:
                # --- Profile training time ---
                start_train = time.time()
                lista_class = ista.ToeplitzLISTA if model_type == "ToeplitzLista" else ista.LISTA
                model = lista_class(A, mu = model_config["initial_mu"], _lambda = model_config["initial_lambda"], nr_folds = model_config["nr_folds"], 
                                    device = config["device"], initialize_randomly = False, share_weights=model_config["share_weights"], train_inputs = torch.stack([y for (y, _) in train_data]))
                regularize = "regularization" in fixed_config[model_type].keys()
                model, train_losses, val_losses  =  train_lista(model, train_data, validation_data, model_config, show_loss_plot = False,
                                                                loss_folder = model_folder, save_name = model_type, regularize = regularize,
                                                                tqdm_position=1, verbose=True, tqdm_leave=tqdm_leave, save=save)
                end_train = time.time()
                train_time = end_train - start_train
                # If you have epochs info, you can average per epoch
                if hasattr(model, 'nr_epochs'):
                    avg_train_time_per_epoch = train_time / model.nr_epochs
                else:
                    avg_train_time_per_epoch = None

                # --- Profile inference time (test set) ---
                test_y = test_data.y
                start_inf = time.time()
                with torch.no_grad():
                    _ = model(test_y)
                end_inf = time.time()
                inference_time = end_inf - start_inf
                avg_inference_time_per_sample = inference_time / len(test_y)

            elif model_type in ["ISTA", "FISTA"]:
                # --- Profile inference time (test set) ---
                test_y = test_data.y
                start_inf = time.time()
                with torch.no_grad():
                    _ = model(test_y)
                end_inf = time.time()
                inference_time = end_inf - start_inf
                avg_inference_time_per_sample = inference_time / len(test_y)

            # Save or print profiling results
            if model_type in ["LISTA"]:
                print(f"[{model_type}] Training time: {train_time:.2f}s, Avg train time/epoch: {avg_train_time_per_epoch}")
            print(f"[{model_type}] Inference time (test set): {inference_time:.2f}s, Avg/sample: {avg_inference_time_per_sample:.6f}s")

            # Optionally, save to file
            with open(os.path.join(model_folder, "profiling.txt"), "w") as f:
                f.write(f"Training time: {train_time}\n")
                f.write(f"Avg train time/epoch: {avg_train_time_per_epoch}\n")
                f.write(f"Inference time (test set): {inference_time}\n")
                f.write(f"Avg inference time/sample: {avg_inference_time_per_sample}\n")

        if plot and args.compute_knot_density:            
            df = make_plots(results_dir_this_experiment, model_types, knot_density_over_experiments, test_accuracy_over_experiments, test_loss_over_experiments, train_accuracy_over_experiments, train_loss_over_experiments, results_dir_with_parent, config, M, N, K, df)
    
    return test_accuracy_over_experiments, test_loss_over_experiments, knot_density_over_experiments

def make_plots(results_dir_this_experiment, model_types, knot_density_over_experiments, test_accuracy_over_experiments, test_loss_over_experiments, train_accuracy_over_experiments, train_loss_over_experiments, results_dir_with_parent, config, M, N, K, df):    
    # %% after looping over each model type, make combined plots of all model types together
    # create a directory for the combined results
    combined_folder = os.path.join(results_dir_this_experiment, "combined")
    os.makedirs(combined_folder, exist_ok=True)

    # make a joint plot of the knot densities
    plt.figure()

    max_folds = 0
    for model_idx, model_type in enumerate(model_types):
        knot_density = knot_density_over_experiments[model_idx][-1]
        folds = np.arange(0, len(knot_density))
        plt.plot(folds, knot_density, label = model_type, c = colors[model_type])
        max_folds = max(max_folds, len(knot_density))

    plt.grid()
    plt.xlabel("fold")
    plt.ylabel("knot density")
    plt.legend(loc='best')
    plt.xlim([0,max_folds-1])
    plt.tight_layout()
    plt.savefig(os.path.join(combined_folder, "knot_density.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(combined_folder, "knot_density.svg"), bbox_inches='tight')
    plt.close()
    
    # make a joint plot of the test losses
    plt.figure()

    max_folds = 0
    for model_idx, model_type in enumerate(model_types):
        test_loss = test_loss_over_experiments[model_idx][-1]
        folds = np.arange(0, len(test_loss))
        plt.plot(folds, test_loss, label = model_type, c = colors[model_type])
        max_folds = max(max_folds, len(test_loss))

    plt.grid()
    plt.xlabel("fold")
    plt.ylabel("test loss")
    plt.legend(loc='best')
    plt.xlim([0,max_folds-1])
    plt.tight_layout()
    plt.savefig(os.path.join(combined_folder, "test_loss.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(combined_folder, "test_loss.svg"), bbox_inches='tight')
    plt.close()

    # make a joint plot of the test accuracies
    plt.figure()

    max_folds = 0
    for model_idx, model_type in enumerate(model_types):
        test_accuracy = test_accuracy_over_experiments[model_idx][-1]
        folds = np.arange(0, len(test_accuracy))
        plt.plot(folds, test_accuracy, label = f"{model_type}_test", c = colors[model_type])
        
        train_accuracy = train_accuracy_over_experiments[model_idx][-1]
        folds = np.arange(0, len(train_accuracy))
        plt.plot(folds, train_accuracy, label = f"{model_type}_train", c = colors[model_type], linestyle='--')
        max_folds = max(max_folds, len(test_accuracy))

    plt.grid()
    plt.xlabel("fold")
    plt.ylabel("test accuracy")
    plt.legend(loc='best')
    plt.xlim([0,max_folds-1])
    plt.tight_layout()
    plt.savefig(os.path.join(combined_folder, "test_accuracy.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(combined_folder, "test_accuracy.svg"), bbox_inches='tight')
    plt.close()


    # %% now plot the results across all experiments
    # mean and standard deviation of the knot densities
    plt.figure()

    max_folds = 0
    for model_idx, model_type in enumerate(model_types):
        knot_density_over_experiments_this_model = knot_density_over_experiments[model_idx]
        knot_density_over_experiments_this_model = torch.stack(knot_density_over_experiments_this_model)
        knot_density_mean = knot_density_over_experiments_this_model.mean(dim=0)
        knot_density_std  = knot_density_over_experiments_this_model.std(dim=0)
        folds = np.arange(0, len(knot_density_mean))
        plt.plot(folds, knot_density_mean, label = model_type, c = colors[model_type])
        plt.fill_between(folds, knot_density_mean - knot_density_std, knot_density_mean + knot_density_std, alpha=0.3, color=colors[model_type])
        max_folds = max(max_folds, len(knot_density_mean))

    plt.grid()
    plt.xlabel("fold")
    plt.ylabel("knot density")
    plt.legend(loc='best')
    plt.xlim([0,max_folds-1])
    plt.title("mean and std of the knot density per fold over the random experiments")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_with_parent, "knot_density.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir_with_parent, "knot_density.svg"), bbox_inches='tight')
    plt.close()

    # mean and standard deviation of the test losses
    plt.figure()

    max_folds = 0
    for model_idx, model_type in enumerate(model_types):
        test_loss_over_experiments_this_model = test_loss_over_experiments[model_idx]
        test_loss_over_experiments_this_model = torch.stack(test_loss_over_experiments_this_model)
        test_loss_mean = test_loss_over_experiments_this_model.mean(dim=0)
        test_loss_std  = test_loss_over_experiments_this_model.std(dim=0)
        folds = np.arange(0, len(test_loss_mean))
        plt.plot(folds, test_loss_mean, label = f"{model_type} test", c = colors[model_type])
        plt.fill_between(folds, test_loss_mean - test_loss_std, test_loss_mean + test_loss_std, alpha=0.3, color=colors[model_type])
        
        train_loss_over_experiments_this_model = train_loss_over_experiments[model_idx]
        train_loss_over_experiments_this_model = torch.stack(train_loss_over_experiments_this_model)
        train_loss_mean = train_loss_over_experiments_this_model.mean(dim=0)
        train_loss_std  = train_loss_over_experiments_this_model.std(dim=0)
        folds = np.arange(0, len(train_loss_mean))
        plt.plot(folds, train_loss_mean, label = f"{model_type} train", c = colors[model_type], linestyle="dashed")
        plt.fill_between(folds, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.3, color=colors[model_type])
        max_folds = max(max_folds, len(test_loss_mean))

    plt.grid()
    plt.xlabel("fold")
    plt.ylabel("L1 loss")
    plt.legend(loc='best')
    plt.xlim([0,max_folds-1])
    plt.title("mean and std of train and test loss per fold over the random experiments")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_with_parent, "train_test_loss.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir_with_parent, "train_test_loss.svg"), bbox_inches='tight')
    plt.close()

    # mean and standard deviation of the test accuracies
    plt.figure()

    max_folds = 0
    for model_idx, model_type in enumerate(model_types):
        test_accuracy_over_experiments_this_model = test_accuracy_over_experiments[model_idx]
        test_accuracy_over_experiments_this_model = torch.stack(test_accuracy_over_experiments_this_model)
        test_accuracy_mean = test_accuracy_over_experiments_this_model.mean(dim=0)
        test_accuracy_std  = test_accuracy_over_experiments_this_model.std(dim=0)
        folds = np.arange(0, len(test_accuracy_mean))
        plt.plot(folds, test_accuracy_mean, label = f"{model_type}_test", c = colors[model_type], linestyle="-")
        plt.fill_between(folds, test_accuracy_mean - test_accuracy_std, test_accuracy_mean + test_accuracy_std, alpha=0.3, color=colors[model_type])
        max_folds = max(max_folds, len(test_accuracy_mean))
        
        train_accuracy_over_experiments_this_model = train_accuracy_over_experiments[model_idx]
        train_accuracy_over_experiments_this_model = torch.stack(train_accuracy_over_experiments_this_model)
        train_accuracy_mean = train_accuracy_over_experiments_this_model.mean(dim=0)
        train_accuracy_std  = train_accuracy_over_experiments_this_model.std(dim=0)
        folds = np.arange(0, len(train_accuracy_mean))
        plt.plot(folds, train_accuracy_mean, label = f"{model_type}_train", c = colors[model_type], linestyle="--")
        plt.fill_between(folds, train_accuracy_mean - train_accuracy_std, train_accuracy_mean + train_accuracy_std, alpha=0.3, color=colors[model_type])

    plt.grid()
    plt.xlabel("fold")
    plt.ylabel("test accuracy")
    plt.legend(loc='best')
    plt.xlim([0,max_folds-1])
    plt.title("mean and std of the test accuracy per fold over the random experiments")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_with_parent, "test_accuracy.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir_with_parent, "test_accuracy.svg"), bbox_inches='tight')
    plt.close()


    new_row = pd.DataFrame({"M": M, "N": N, "K": K}, index=[0])

    for model_idx, model_type in enumerate(model_types):
        knot_density_last_experiment_this_model = knot_density_over_experiments[model_idx][-1]

        new_row["model_type"] = model_type
        new_row["knot_density_max"] = knot_density_last_experiment_this_model.max().item()
        new_row["knot_density_end"] = knot_density_last_experiment_this_model[-1].item()

        test_loss_last_experiment_this_model = test_loss_over_experiments[model_idx][-1]
        new_row["test_loss_end"] = test_loss_last_experiment_this_model[-1].item()
        
        train_loss_last_experiment_this_model = train_loss_over_experiments[model_idx][-1]
        new_row["train_loss_end"] = train_loss_last_experiment_this_model[-1].item()

        test_accuracy_last_experiment_this_model = test_accuracy_over_experiments[model_idx][-1]
        new_row["test_accuracy_end"] = test_accuracy_last_experiment_this_model[-1].item()
        new_row["noise_std"] = config["data_that_stays_constant"]["noise_std"]

        df = pd.concat([df, new_row], ignore_index=True)

    parameters_output_path = os.path.join(results_dir_with_parent, "parameters.csv")
    df.to_csv(parameters_output_path)
    print(f"Saved results to {parameters_output_path}")
    
    return df


if __name__ == "__main__": 


    config_path = args.config

    with open(config_path, 'r') as file:
        fixed_config = yaml.load(file, Loader=yaml.FullLoader)

    torch.manual_seed(fixed_config["seed"])
    np.random.seed(fixed_config["seed"])

    fixed_config['LISTA']['compute_loss_on_all_folds'] = args.loss_on_all_folds
    if args.reconstruction_loss == "l1":
        fixed_config['LISTA']["l1_weight"] = 1.0
        fixed_config['LISTA']["l2_weight"] = 0.0
    elif args.reconstruction_loss == "l2":
        fixed_config['LISTA']["l1_weight"] = 0.0
        fixed_config['LISTA']["l2_weight"] = 1.0
    else:
        raise UserWarning()


    if args.sweep_L2 == True:
        weights = [0.0, 0.000005, 0.00001, 0.000025, 0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025]
        # weights = [0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
        
        for weight in weights:
            fixed_config['LISTA']['regularization'] = {"type": "L2", "weight": weight}
            run_experiment(config_path, config=fixed_config, model_types=args.model_types, save=True, plot=True, output_identifier=f"L2={weight}")
    elif args.sweep_jacobian == True:
        weights = [0.0, 0.000005, 0.00001, 0.000025, 0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025]
        
        for weight in weights:
            fixed_config['LISTA']['regularization'] = {"type": "jacobian", "weight": weight}
            run_experiment(config_path, config=fixed_config, model_types=args.model_types, save=True, plot=True, output_identifier=f"jacob={weight}")
    elif args.sweep_noise_stds == True:
        noise_stds = [0, 0.0001, 0.001, 0.01, 0.1, 1]

        for noise_std in noise_stds:
            # oh the irony
            fixed_config['data_that_stays_constant']['noise_std'] = noise_std
            run_experiment(config_path, config=fixed_config, model_types=args.model_types, save=True, plot=False, output_identifier=f"noise_std={noise_std}")
    elif args.sweep_ranks == True:
        ranks = [16, 14, 12, 10, 8]

        for rank in ranks:
            fixed_config['A_low_rank'] = rank
            run_experiment(config_path, config=fixed_config, model_types=args.model_types, save=True, plot=False, output_identifier=f"rank={rank}")
    elif args.sweep_M == True:
        Ms = [16, 14, 12, 10, 8] 

        for m in Ms:
            assert m % 2 == 0
            fixed_config['data_that_varies']['M'] = {'min': m, 'max': m}
            # we want to ensure that support(x) <= 2*spark(A)
            fixed_config['data_that_varies']['K'] = {'min': m // 2, 'max': m // 2}
            run_experiment(config_path, config=fixed_config, model_types=args.model_types, save=True, plot=False, output_identifier=f"M={m}")
    elif args.sweep_num_folds == True:
        num_foldss = [5, 6, 7, 8, 9, 10]
        
        for num_folds in num_foldss:
            fixed_config['LISTA']['nr_folds'] = num_folds
            run_experiment(config_path, config=fixed_config, model_types=args.model_types, save=True, plot=True, output_identifier=f"num_folds={num_folds}")
    else:
        run_experiment(config_path, config=fixed_config, model_types=args.model_types, plot=True)