import os
from pathlib import Path
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2,
    "figure.dpi": 300,
    "font.family": "serif",
})

def load_tensor_from_tar(path):
    return torch.load(path, map_location=torch.device('cpu')).numpy()

def get_nested(config, key):
    """Get nested value from dict using dot notation."""
    keys = key.split('.')
    for k in keys:
        config = config[k]
    return config

def collect_experiment_data(experiment_root, config_key, model_type="LISTA"):
    """Collects (x_value, final_support_densities, final_identity_densities) for each experiment in root."""
    experiment_data = []
    for exp_dir in sorted(Path(experiment_root).iterdir()):
        if not exp_dir.is_dir():
            continue
        config_path = exp_dir / "config.yaml"
        if not config_path.exists():
            continue
        # Load config and get x_value
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        try:
            x_value = get_nested(config, config_key)
        except Exception:
            continue
        # Find all run dirs (ending in /0/, /1/, etc.)
        run_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        final_support_densities = []
        final_identity_densities = []
        for run_dir in run_dirs:
            model_dir = run_dir / model_type
            support_path = model_dir / "support_boundary_density.tar"
            identity_path = model_dir / "identity_boundary_density.tar"
            if support_path.exists():
                arr = load_tensor_from_tar(support_path)
                final_support_densities.append(arr[-1])
            if identity_path.exists():
                arr = load_tensor_from_tar(identity_path)
                final_identity_densities.append(arr[-1])
        if final_support_densities and final_identity_densities:
            experiment_data.append({
                "x_value": x_value,
                "support_densities": np.array(final_support_densities),
                "identity_densities": np.array(final_identity_densities),
                "exp_name": exp_dir.name
            })
    return experiment_data

def plot_density_vs_x(experiment_data_l1, experiment_data_l2, experiment_data_ista, config_key, save_path=None, x_label_map=None):
    # Sort by x_value
    experiment_data_l1 = sorted(experiment_data_l1, key=lambda d: d["x_value"])
    experiment_data_l2 = sorted(experiment_data_l2, key=lambda d: d["x_value"])
    experiment_data_ista = sorted(experiment_data_ista, key=lambda d: d["x_value"])
    x_vals_l1 = np.array([d["x_value"] for d in experiment_data_l1])
    x_vals_l2 = np.array([d["x_value"] for d in experiment_data_l2])
    x_vals_ista = np.array([d["x_value"] for d in experiment_data_ista])

    support_means_l1 = np.array([d["support_densities"].mean() for d in experiment_data_l1])
    support_sems_l1 = np.array([d["support_densities"].std() / np.sqrt(d["support_densities"].shape[0]) for d in experiment_data_l1])
    identity_means_l1 = np.array([d["identity_densities"].mean() for d in experiment_data_l1])
    identity_sems_l1 = np.array([d["identity_densities"].std() / np.sqrt(d["identity_densities"].shape[0]) for d in experiment_data_l1])

    support_means_l2 = np.array([d["support_densities"].mean() for d in experiment_data_l2])
    support_sems_l2 = np.array([d["support_densities"].std() / np.sqrt(d["support_densities"].shape[0]) for d in experiment_data_l2])
    identity_means_l2 = np.array([d["identity_densities"].mean() for d in experiment_data_l2])
    identity_sems_l2 = np.array([d["identity_densities"].std() / np.sqrt(d["identity_densities"].shape[0]) for d in experiment_data_l2])

    support_means_ista = np.array([d["support_densities"].mean() for d in experiment_data_ista])
    support_sems_ista = np.array([d["support_densities"].std() / np.sqrt(d["support_densities"].shape[0]) for d in experiment_data_ista])
    identity_means_ista = np.array([d["identity_densities"].mean() for d in experiment_data_ista])
    identity_sems_ista = np.array([d["identity_densities"].std() / np.sqrt(d["identity_densities"].shape[0]) for d in experiment_data_ista])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharex=True)
    colors = {"ISTA": "tab:blue", "l1": "tab:green", "l2": "tab:orange"}
    model_labels = {"ISTA": "ISTA", "l2": "LISTA L2", "l1": "LISTA L1",}
    density_labels = {
        "support": r"Final fold decision density $\rho_{d}$",
        "identity": r"Final fold knot density $\rho_{kt}$"
    }

    # Map x label if mapping provided
    if x_label_map is not None and config_key in x_label_map:
        x_label = x_label_map[config_key]
    else:
        x_label = config_key.replace('.', ' ')

    # Decision density subplot
    ax = axes[0]
    if config_key == "data_that_stays_constant.noise_std":
        ax.set_xscale("symlog", linthresh=1e-6)
    ax.plot(x_vals_ista, support_means_ista, '-', color=colors["ISTA"], label=model_labels["ISTA"])
    ax.fill_between(x_vals_ista, support_means_ista - support_sems_ista, support_means_ista + support_sems_ista,
                    color=colors["ISTA"], alpha=0.2)
    ax.plot(x_vals_l2, support_means_l2, '-', color=colors["l2"], label=model_labels["l2"])
    ax.fill_between(x_vals_l2, support_means_l2 - support_sems_l2, support_means_l2 + support_sems_l2,
                    color=colors["l2"], alpha=0.2)
    ax.plot(x_vals_l1, support_means_l1, '-', color=colors["l1"], label=model_labels["l1"])
    ax.fill_between(x_vals_l1, support_means_l1 - support_sems_l1, support_means_l1 + support_sems_l1,
                    color=colors["l1"], alpha=0.2)
    ax.set_ylabel(density_labels["support"])
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_xlabel(x_label)

    # Knot density subplot
    ax = axes[1]
    if config_key == "data_that_stays_constant.noise_std":
        ax.set_xscale("symlog", linthresh=1e-6)
    ax.plot(x_vals_ista, identity_means_ista, '-', color=colors["ISTA"], label=model_labels["ISTA"])
    ax.fill_between(x_vals_ista, identity_means_ista - identity_sems_ista, identity_means_ista + identity_sems_ista,
                    color=colors["ISTA"], alpha=0.2)
    ax.plot(x_vals_l2, identity_means_l2, '-', color=colors["l2"], label=model_labels["l2"])
    ax.fill_between(x_vals_l2, identity_means_l2 - identity_sems_l2, identity_means_l2 + identity_sems_l2,
                    color=colors["l2"], alpha=0.2)
    ax.plot(x_vals_l1, identity_means_l1, '-', color=colors["l1"], label=model_labels["l1"])
    ax.fill_between(x_vals_l1, identity_means_l1 - identity_sems_l1, identity_means_l1 + identity_sems_l1,
                    color=colors["l1"], alpha=0.2)
    ax.set_ylabel(density_labels["identity"])
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_xlabel(x_label)

    # Shared legend above both plots, centered
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_key", type=str, default="data_that_stays_constant.noise_std",
                        help="Config key to use as x value (dot notation, e.g. 'data_that_stays_constant.noise_std')")
    parser.add_argument("--save_path", type=str, default=None, help="Optional path to save plot")
    parser.add_argument("--l1_root", type=str, required=True, help="Path to L1 experiment root directory")
    parser.add_argument("--l2_root", type=str, required=True, help="Path to L2 experiment root directory")
    parser.add_argument("--ista_root", type=str, required=True, help="Path to ISTA experiment root directory")
    args = parser.parse_args()

    # Example label mapping
    x_label_map = {
        "data_that_stays_constant.noise_std": r"$\sigma_n^2$",
        "data_that_varies.M.min": r"$M$",
    }

    print("Collecting ISTA experiments...")
    experiment_data_ista = collect_experiment_data(args.ista_root, args.config_key, model_type="ISTA")
    print("Collecting L1 experiments...")
    experiment_data_l1 = collect_experiment_data(args.l1_root, args.config_key)
    print("Collecting L2 experiments...")
    experiment_data_l2 = collect_experiment_data(args.l2_root, args.config_key)

    if args.save_path is None:
        save_path = f"./{args.config_key}_plot.png"
    else:
        save_path = args.save_path

    plot_density_vs_x(experiment_data_l1, experiment_data_l2, experiment_data_ista, args.config_key, save_path=save_path, x_label_map=x_label_map)