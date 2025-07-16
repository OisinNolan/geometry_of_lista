#!/bin/bash

# Usage (from project root directory): bash reproduce_results.sh ./configs/base_config.yaml ./output

CONFIG_PATH=$1
OUT_DIR=$2

L1_DIR="${OUT_DIR}/l1"
L2_DIR="${OUT_DIR}/l2"

# Create output directories if they don't exist
mkdir -p "${L1_DIR}"
mkdir -p "${L2_DIR}"

# train models
python main.py \
    --config="${CONFIG_PATH}" \
    --results_dir="${L1_DIR}" \
    --compute_knot_density=False \
    --visualize_hyperplane=False \
    --reconstruction_loss=l1 \
    --model_types LISTA

python main.py \
    --config="${CONFIG_PATH}" \
    --results_dir="${L2_DIR}" \
    --compute_knot_density=False \
    --visualize_hyperplane=False \
    --reconstruction_loss=l2 \
    --model_types ISTA LISTA

# compute knot density and decision density
python compute_boundary_density.py \
    --experiment_root="${L1_DIR}" \
    --model_type LISTA

python compute_boundary_density.py \
    --experiment_root="${L2_DIR}" \
    --model_type LISTA ISTA

# compute test performance with both L1 and L2 metrics
python compute_test_loss.py \
    --experiment_root="${L1_DIR}" \
    --model_type LISTA

python compute_test_loss.py \
    --experiment_root="${L2_DIR}" \
    --model_type ISTA LISTA

# make loss and knot / decision density plots
L1_CHILD=$(find $L1_DIR -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)
L2_CHILD=$(find $L2_DIR -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)

python plot_knots_and_loss.py \
    --l1_path="${L1_CHILD}" \
    --l2_path="${L2_CHILD}"

# make hyperplane visualization plots
python hyper_plane_analysis.py \
    --experiment_root="${L1_CHILD}" \
    --model_type LISTA \
    --decision_density=True

python hyper_plane_analysis.py \
    --experiment_root="${L1_CHILD}" \
    --model_type LISTA \
    --decision_density=False

python hyper_plane_analysis.py \
    --experiment_root="${L2_CHILD}" \
    --model_type LISTA ISTA \
    --decision_density=True

python hyper_plane_analysis.py \
    --experiment_root="${L2_CHILD}" \
    --model_type LISTA ISTA \
    --decision_density=False