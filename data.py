"""
This file creates the functions that create the data and the dataloaders for the experiments
"""

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings

# %% data generation
def data_generator(A: torch.tensor, nr_of_examples: int = 4, maximum_sparsity: int = 4, 
                   x_magnitude: tuple[float,float] = (0.5, 1.5), N: int = 16, noise_std: float = 0.1, magnitude_shift_epsilon = None):
    """
    Create some data for training LISTA. This data is simply a random x-vector, and the y-vector is created by multiplying A with x.

    There is however a constraint on the sparsity of x, it should be at most maximum_sparsity. We do this by multiplying the x-vector with a mask.

    inputs:
    - A (torch.tensor): the matrix A in the equation y=Ax, of shape (M, N), with M<N, i.e. M is the measurement dimension and N is the signal dimension
    - nr_of_examples (int): the total number of examples to create
    - maximum_sparsity (int): the maximum sparsity of the x-vector
    - x_magnitude (tuple of floats): the magnitude of the x-vector, (min, max)
    - N (int): the signal dimension of x
    - noise_std (float): the standard deviation of the noise added to the y-vector

    outputs:
    - y (torch.tensor): the y-vector, of shape (nr_of_examples, M)
    - x (torch.tensor): the x-vector, of shape (nr_of_examples, N)
    """
    # create a random x-vector
    sign      = torch.randint(0, 2, (nr_of_examples, N)) * 2 - 1
    x_magnitude_low, x_magnitude_high = perturb_x_magnitudes(x_magnitude[0], x_magnitude[1], magnitude_shift_epsilon) if magnitude_shift_epsilon else (x_magnitude[0], x_magnitude[1])
    magnitude = torch.rand(nr_of_examples, N) * (x_magnitude_high - x_magnitude_low) + x_magnitude_low
    x         = sign * magnitude        

    # create a mask that makes sure the x-vector is at most maximum_sparsity, or even less than that
    mask = torch.zeros(nr_of_examples, N)
    for i in range(nr_of_examples):
        sparsity = torch.randint(1, maximum_sparsity+1, (1,))
        x_indices = torch.randperm(N)[:sparsity]
        mask[i, x_indices] = 1

    # apply the mask
    x = x * mask

    # create the y-vector
    y = torch.nn.functional.linear(x, A) + noise_std*torch.randn(nr_of_examples, A.shape[0])

    return y, x  

# %% the pytorch dataset class 
class ISTAData(torch.utils.data.Dataset):
    def __init__(self, A: torch.tensor, nr_of_examples: int = 4, maximum_sparsity: int = 4, 
                 x_magnitude: tuple[float,float] = (0.5, 1.5), N: int = 16, noise_std: float = 0.1, magnitude_shift_epsilon = None):
        """
        Create a pytorch dataset for training LISTA. This dataset creates random x-vectors and y-vectors by multiplying A with x.

        There is however a constraint on the sparsity of x, it should be at most maximum_sparsity. We do this by multiplying the x-vector with a mask.

        inputs:
        - A (torch.tensor): the matrix A in the equation y=Ax, of shape (M, N), with M<N, i.e. M is the measurement dimension and N is the signal dimension
        - nr_of_examples (int): the total number of examples to create
        - maximum_sparsity (int): the maximum sparsity of the x-vector
        - x_magnitude (tuple of floats): the magnitude of the x-vector, (min, max)
        - N (int): the signal dimension of x
        - device (str): the device to run the module on, default is "cpu"
        - noise_std (float): the standard deviation of the noise added to the y-vector
        """
        # save the parameters
        self.A = A
        self.nr_of_examples = nr_of_examples
        self.maximum_sparsity = maximum_sparsity
        self.x_magnitude = x_magnitude
        self.N = N
        self.noise_std = noise_std

        # create the data
        self.y, self.x = data_generator(A, nr_of_examples, maximum_sparsity, x_magnitude, N, noise_std, magnitude_shift_epsilon)

    def __len__(self):
        return self.nr_of_examples

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, end = key.start or 0, key.stop
            new_dataset = ISTAData(self.A, 
                nr_of_examples=end-start,
                maximum_sparsity=self.maximum_sparsity,
                x_magnitude=self.x_magnitude,
                N=self.N,
                noise_std=self.noise_std)

            # Override the generated data with sliced versions
            new_dataset.x = self.x[start:end]
            new_dataset.y = self.y[start:end]
            new_dataset.nr_of_examples = end-start
            return new_dataset
        else:
            return self.y[key], self.x[key]
   
def make_distribution_shift_transform(N, epsilon):
    # x_transformed = (I + \epsilon*a)x + \epsilon*b
    # A ~ N; b ~ N
    A = torch.randn(N, N)
    b = torch.randn(N)
    return lambda x: torch.matmul(x, (torch.eye(N) + A*epsilon)) + epsilon*b # add epsilon*a on the diag
    
def perturb_x_magnitudes(x_magnitude_low, x_magnitude_high, epsilon):
    a = torch.randn(1) * epsilon
    b = torch.randn(1) * epsilon
    return x_magnitude_low + a, x_magnitude_high + b
    
def create_train_validation_test_datasets(A: torch.tensor, maximum_sparsity: int = 4, x_magnitude: tuple[float,float] = (0.5, 1.5), N: int = 16, noise_std: float = 0.1,
                                          nr_of_examples_train: int = 4, nr_of_examples_validation: int = 4, nr_of_examples_test: int = 4, test_magnitude_shift_epsilon: float = 0.0):
    """
    Create three datasets with the same parameters, except for the number of examples.
    """
    train_data      = ISTAData(A, nr_of_examples_train, maximum_sparsity, x_magnitude, N, noise_std)
    validation_data = ISTAData(A, nr_of_examples_validation, maximum_sparsity, x_magnitude, N, noise_std)
    test_data       = ISTAData(A, nr_of_examples_test, maximum_sparsity, x_magnitude, N, noise_std, test_magnitude_shift_epsilon)

    return train_data, validation_data, test_data
    