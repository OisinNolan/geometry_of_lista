"""
This file creates some functions usefull for the design of experiments.
"""

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
from scipy.linalg import toeplitz
import torch.nn.functional as F

# %%
def create_random_matrix_with_good_singular_values(M: int, N: int):
    """
    This function creates a random matrix A with good singular values for ISTA.
    i.e. the largest singular value of A.T@A is 1.0, this makes the ISTA more stable.
    """

    # create a random matrix A
    A = torch.randn(M, N)
    
    # We are going to adapt measurement A, such that its largest eigenvalue is 1.0, this makes the ISTA more stable
    largest_eigenvalue = torch.svd(A.t() @ A).S[0]
    A = 1.0 * A / (largest_eigenvalue**0.5)

    return A

def create_random_matrix(M: int, N: int):
    """
    This function creates a random matrix A.
    """

    # create a random matrix A
    A = torch.randn(M, N)

    # each row of A, the expected std is N, since we are adding N gaussians
    # we want to normalize the rows of A, such that the expected std is 1
    # this is done by dividing by the square root of N, so that the expected std is 1
    A = A / N**0.5

    return A


def create_unpadded_conv_matrix(M: int, N: int, kernel_size: int = 21):
    """
    Creates a matrix for an unpadded 1D convolution with a random kernel.

    Parameters:
    - signal_length: Length of the input signal.
    - kernel: The convolution kernel (filter) as a 1D array.

    Returns:
    - conv_matrix: The Toeplitz-like matrix of size (signal_length - kernel_length + 1) x signal_length.
    """
    assert M == (N - (kernel_size - 1)), "Using unpadded conv imposes a size constraint on M"
    kernel = np.random.randn(kernel_size)
    conv_matrix_length = N - kernel_size + 1
    
    # Initialize the Toeplitz-like matrix with zeros
    conv_matrix = np.zeros((conv_matrix_length, N))

    # Fill the Toeplitz-like matrix with shifted versions of the kernel
    for i in range(conv_matrix_length):
        conv_matrix[i, i:i + kernel_size] = kernel

    # normalize to preserve data std
    conv_matrix = conv_matrix / N**0.5

    return torch.tensor(conv_matrix, dtype=torch.float32)


def create_convolution_matrix(M: int, N: int, kernel_size: int = 5):
    assert kernel_size <= N, "❗️ Your kernel should be smaller than the signal dimension N"
    # generate a random kernel sampled from standard normal
    kernel = np.random.randn(kernel_size)
    half_kernel_size = kernel_size//2 
    # create a convolution sliding-window matrix with this kernel
    r = *kernel[half_kernel_size:], *np.zeros(N - (half_kernel_size + 1))
    c = *np.flip(kernel[:half_kernel_size + 1]), *np.zeros(N - (half_kernel_size+1))
    t = toeplitz(c=c, r=r)
    conv_matrix = torch.tensor(t, dtype=torch.float32)

    if M != N:
        print("❗️ Note: M and N are not equal, so the convolution matrix will be zero-padded or cropped.")
        if M > N:
            return torch.hstack(conv_matrix, torch.zeros((M-N, N)))
        if M < N:
            return conv_matrix[:M, :N]
    
    return conv_matrix
    

def create_random_toeplitz_matrix(M: int, N: int):
    """
    Creates a random Toeplitz matrix A of size M x N.

    A Toeplitz matrix has constant diagonals. We generate a random first column and first row
    to define the matrix, then normalize as in the reference.
    """
    # First column and first row of the Toeplitz matrix
    c = torch.randn(M)
    r = torch.randn(N)
    r[0] = c[0]  # ensure top-left entry agrees

    # Use numpy/scipy to generate Toeplitz structure, then convert back to torch
    A_np = toeplitz(c.numpy(), r.numpy())
    A = torch.tensor(A_np, dtype=torch.float32)

    largest_eigenvalue = torch.svd(A.t() @ A).S[0]
    A = 1.0 * A / (largest_eigenvalue**0.5)

    return A



def sample_experiment(config: dict, max_tries: int = 1000):
    """
    This function will sample parameters that vary to create an experiment.
    """
    for _ in range(max_tries):
        # sample the parameters that vary
        M = torch.randint(config["data_that_varies"]["M"]["min"], config["data_that_varies"]["M"]["max"] + 1, (1,)).item()
        N = torch.randint(config["data_that_varies"]["N"]["min"], config["data_that_varies"]["N"]["max"] + 1, (1,)).item()
        K = torch.randint(config["data_that_varies"]["K"]["min"], config["data_that_varies"]["K"]["max"] + 1, (1,)).item()

        # check if the parameters are valid
        if M <= N and K <= M:
            break

    else:
        # for-else triggers if the for loop did not break (i.e. we did not find valid parameters after max_tries)
        raise ValueError("Could not find valid parameters after {} tries.".format(max_tries))

    assert np.sum([config["A_with_good_singular_values"], config["A_is_convolution"], config["A_is_identity"]]) < 2, "❗️ You may not choose more than one type of A matrix."

    # create the A matrix that belongs to these parameters
    if config["A_with_good_singular_values"]:
        A = create_random_matrix_with_good_singular_values(M, N)
    elif config["A_is_convolution"]:
        A = create_unpadded_conv_matrix(M, N)
    elif config["A_is_toeplitz"]:
        A = create_random_toeplitz_matrix(M, N)
    elif config["A_is_identity"]:
        assert M == N, "x and y must have the same dimensionality when A=I"
        A = torch.eye(M)
    else:
        A = create_random_matrix(M, N)
    
    return M, N, K, A