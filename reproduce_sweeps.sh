#!/bin/bash

# Usage (from project root directory): bash reproduce_results.sh ./configs/base_config.yaml ./output

CONFIG_PATH=$1
OUT_DIR=$2

# 1. Sweep noise stds

NOISE_DIR="${OUT_DIR}/sweep_noise"

mkdir -p "${NOISE_DIR}"

L1_DIR="${NOISE_DIR}/l1"
L2_DIR="${NOISE_DIR}/l2"

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
    --sweep_noise_stds=True \
    --model_types LISTA

python main.py \
    --config="${CONFIG_PATH}" \
    --results_dir="${L2_DIR}" \
    --compute_knot_density=False \
    --visualize_hyperplane=False \
    --reconstruction_loss=l2 \
    --sweep_noise_stds=True \
    --model_types ISTA LISTA

# compute knot density and decision density
python compute_boundary_density.py \
    --experiment_root="${L1_DIR}" \
    --model_type LISTA

python compute_boundary_density.py \
    --experiment_root="${L2_DIR}" \
    --model_type LISTA ISTA

# plot sweep results
python plot_sweep.py \
    --l1_root="${L1_DIR}" \
    --l2_root="${L2_DIR}" \
    --ista_root="${L2_DIR}" \
    --config_key="data_that_stays_constant.noise_std"

# 2. Sweep M

NOISE_DIR="${OUT_DIR}/sweep_M"

mkdir -p "${NOISE_DIR}"

L1_DIR="${NOISE_DIR}/l1"
L2_DIR="${NOISE_DIR}/l2"

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
    --sweep_M=True \
    --model_types LISTA

python main.py \
    --config="${CONFIG_PATH}" \
    --results_dir="${L2_DIR}" \
    --compute_knot_density=False \
    --visualize_hyperplane=False \
    --reconstruction_loss=l2 \
    --sweep_M=True \
    --model_types ISTA LISTA

# compute knot density and decision density
python compute_boundary_density.py \
    --experiment_root="${L1_DIR}" \
    --model_type LISTA

python compute_boundary_density.py \
    --experiment_root="${L2_DIR}" \
    --model_type LISTA ISTA

# plot sweep results
python plot_sweep.py \
    --l1_root="${L1_DIR}" \
    --l2_root="${L2_DIR}" \
    --ista_root="${L2_DIR}" \
    --config_key="data_that_varies.M.min"