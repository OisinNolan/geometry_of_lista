import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import argparse

# IEEE-style plot settings
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
    """Load a tensor directly from a .tar file."""
    return torch.load(path, map_location=torch.device('cpu'))

def plot_results(l2_path, l1_path, save_path=None):
    runs = ["0", "1", "2"]
    models = ["ISTA", "LISTA", "LISTA-L1"]
    model_display_names = {
        "ISTA": "ISTA",
        "LISTA": "LISTA L2",
        "LISTA-L1": "LISTA L1"
    }
    
    # Add both L1 and L2 loss metrics
    metrics = [
        "identity_boundary_density.tar", 
        # "knot_density.tar", 
        "support_boundary_density.tar", 
        "test_loss_l1.tar",
        "test_loss_l2.tar"
    ]
    metric_labels = {
        "identity_boundary_density.tar": r"Knot Density  $\rho_{kt}$",
        # "knot_density.tar": r"Knot Density  $\rho_{kt}$",
        "support_boundary_density.tar": r"Decision Density $\rho_{d}$",
        "test_loss_l1.tar": "Test Loss L1",
        "test_loss_l2.tar": "Test Loss L2",
    }
    
    results = {metric: {model: [] for model in models} for metric in metrics}
    
    # Load data
    for run in runs:
        for model in models[:-1]:  # Process ISTA and LISTA normally
            model_path = os.path.join(l2_path, run, model)
            for metric in metrics:
                metric_path = os.path.join(model_path, metric)
                tensor = load_tensor_from_tar(metric_path)
                results[metric][model].append(tensor.numpy())
        
        model_path = os.path.join(l1_path, run, "LISTA")
        for metric in metrics:
            metric_path = os.path.join(model_path, metric)
            tensor = load_tensor_from_tar(metric_path)
            results[metric]["LISTA-L1"].append(tensor.numpy())
    
    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 5), sharex=True)
    colors = {"ISTA": "tab:blue", "LISTA": "tab:orange", "LISTA-L1": "tab:green"}
    
    # Flatten axes for easier iteration
    axes_order = ["identity_boundary_density.tar", "test_loss_l1.tar", 
                 "support_boundary_density.tar", "test_loss_l2.tar"]
    
    for ax, metric in zip(axes.flatten(), axes_order):
        for model in models:
            data = np.stack(results[metric][model])
            if "test_loss" in metric:
                data = np.pad(data, ((0, 0), (1, 0)), mode='edge')
            mean = np.mean(data, axis=0)[1:]
            sem = (np.std(data, axis=0) / np.sqrt(len(runs)))[1:]
            
            ax.plot(range(1, len(mean) + 1), mean, 
                   label=model_display_names[model], color=colors[model])
            ax.set_xscale("log")
            ax.fill_between(range(1, len(mean) + 1), mean - sem, mean + sem, 
                          alpha=0.2, color=colors[model])
            ax.set_ylabel(metric_labels[metric])
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            
            if metric == "support_boundary_density.tar":
                hline = ax.axhline(y=0.5, color='gold', linestyle='--', label='lower bound')
                ax.legend(handles=[hline], labels=[r"Lower bound optimal $\rho_d$"], loc='best')
    
    # Set x-labels for bottom row only
    for ax in axes[-1]:
        ax.set_xlabel("Iteration")
    
    # Shared legend
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2_path", type=str, required=True, help="Path to base directory")
    parser.add_argument("--l1_path", type=str, required=True, help="Path to extra base directory")
    parser.add_argument("--save_path", type=str, default="knots_and_loss_test.png", help="Path to save the plot")
    args = parser.parse_args()

    plot_results(args.l2_path, args.l1_path, args.save_path)