"""
This script creates the functions used to analyze the knot density of the ISTA algorithm.
"""

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings

from ista import ISTA



def binarize_jacobian(jacobian):
    """
    Sets each row in the jacobian containing a non-zero value to a row of 1s

    The idea is to identify which elements of the output y_i are definitely sparsified by the model.
    
    Args:
        jacobian: torch tensor of shape (batch_size, rows, cols)
    
    Returns:
        Binary tensor of same shape as input with rows either all 0s or all 1s
    """
    import torch
    
    # Convert to torch if not already
    if not isinstance(jacobian, torch.Tensor):
        jacobian = torch.tensor(jacobian)
    
    # Find rows with any non-zero elements
    row_mask = torch.any(torch.abs(jacobian) > 0, dim=-1, keepdim=True)
    
    # Expand mask to match input dimensions
    return row_mask.expand_as(jacobian).to(jacobian.dtype)

# %% helper functions
def generate_path(M: int, nr_points_along_path: int, path_delta: float, anchor_point_std: float, ista: ISTA, specified_anchors = None, anchor_on_sphere: bool = False):
    """
    generate a random path, this is done by sampling two anchor points, and then creating a path between them with a step size of path_delta.
    If we have do not yet reach the nr_points_along_path, we sample a new anchor point, and continue our path towards that, etc. etc.
    This is done untill we hit nr_points_along_path. Note that we can thus quit early before reaching the final anchor point. As we always exactly use nr_points_along_path points
    with a fixed delta between the points. The length of the path is thus always the same.
    """
    device = ista.device
    
    def sample_new_anchor_point():
        if anchor_on_sphere:
            point = torch.randn(M, device=device)
            norm = torch.norm(point)
            point_on_sphere = anchor_point_std * point / norm
            return point_on_sphere
        elif specified_anchors is not None:
            sampled_index = torch.randint(low=0, high=specified_anchors.shape[0], size=(1,))
            # return ista.train_inputs[sampled_index].to(device)
            return specified_anchors[sampled_index].to(device)
        else:
            return torch.randn(M, device = device) * anchor_point_std
        
    # step one, sample anchor point 0
    anchor_point_zero = sample_new_anchor_point()
    
    # initialze the current length of the path
    current_length = 0
    
    # initialize the y data
    y = torch.zeros(nr_points_along_path, M, device = device)

    anchor_points = [anchor_point_zero.cpu().numpy().flatten()]

    # keep looping untill we hit current_length == nr_points_along_path
    while current_length < nr_points_along_path:
        # sample a new anchor point
        anchor_point_one = sample_new_anchor_point()
        anchor_points.append(anchor_point_one.cpu().numpy().flatten())

        # calculate the direction vector
        direction_vector = (anchor_point_one - anchor_point_zero) / torch.linalg.norm(anchor_point_one - anchor_point_zero, ord=2)

        # calculate the distance between the two anchor points
        distance = torch.linalg.norm(anchor_point_one - anchor_point_zero, ord=2)

        # calculate the number of points we can still take
        nr_points_to_take = min(nr_points_along_path - current_length, int(distance / path_delta))

        # put the points in the y data
        y[current_length:current_length+nr_points_to_take] = anchor_point_zero + direction_vector * torch.arange(nr_points_to_take, device = device).float().unsqueeze(1) * path_delta

        # update the current length
        current_length += nr_points_to_take

        # update the anchor point
        anchor_point_zero = anchor_point_one

    # assert current length is equal to nr_points_along_path
    assert current_length == nr_points_along_path, "current_length should be equal to nr_points_along_path"

    return y, torch.tensor(np.array(anchor_points))

def extract_knots_from_jacobian(jacobian: torch.tensor, tolerance: float = 0, boundary_type: str = "identity"):
    """
    Given a Jacobian matrix, extract the knots from it. This assumes the jacobian was created along a 1D-path.
    This makes it simpler that extracting the linear regions, as we can just use the differences between consecutive points, and count the nr of times it is non-zero.

    inputs:
    - jacobian (torch.tensor): the Jacobian of the forward function, of shape (nr_lines_in_batch, nr_points_in_line, N, M), where the nr_of_lines_in_batch is the number of lines we are looking at, and nr_points_in_line is the number of points along the line
                               the Jacobian can also be of shape (nr_points_in_line, N, M), in other words we only have one line, and we are looking at the points along the line
    outputs:
    - nr_of_knots (int): the number of knots we encounter along the path
    """     
    # assert the jacobian has 3 or 4 dimensions
    assert len(jacobian.shape) == 4 or len(jacobian.shape) == 3, "the jacobian should have 3 or 4 dimensions"

    if boundary_type == "cardinality":
        if len(jacobian.shape) == 4:
            # Batched case
            binary_jacobian = binarize_jacobian(jacobian)
            row_means = binary_jacobian.mean(dim=3) # mean over M dimension
            number_of_1_rows = row_means.sum(dim=2) # sum over N dimension 
            nr_of_knots = torch.sum(number_of_1_rows[:,1:] != number_of_1_rows[:,:-1], dim=1)
        else:
            # Single line case (original code)
            binary_jacobian = binarize_jacobian(jacobian)
            row_means = binary_jacobian.mean(dim=2)
            number_of_1_rows = row_means.sum(dim=1) 
            nr_of_knots = torch.sum(number_of_1_rows[1:] != number_of_1_rows[:-1])
    elif boundary_type == "support":
        if len(jacobian.shape) == 4:
            # Batched case
            nr_lines_in_batch, nr_points_in_line, N, Z  = jacobian.shape # let's just call the last dimension Z for now
        
            # reshape jacobian to (batch, N*Z)
            binarized_jacobian = binarize_jacobian(jacobian).view(nr_lines_in_batch, nr_points_in_line, N*Z)

            # calculate the differences between consecutive points
            nr_of_knots = torch.sum(torch.any(torch.abs((binarized_jacobian[:,1:,:] - binarized_jacobian[:,:-1,:]))>tolerance, dim=2), dim=1)
        else:
            # get the shape of the jacobian
            nr_points_in_line, N, Z  = jacobian.shape

            # reshape jacobian to (batch, N*Z)
            binarized_jacobian = binarize_jacobian(jacobian).view(nr_points_in_line, N*Z)

            # calculate the differences between consecutive points
            nr_of_knots = torch.sum(torch.any(torch.abs((binarized_jacobian[1:,:] - binarized_jacobian[:-1,:]))>tolerance, dim=1))
    else:
        # get the number of dimensions in the jacobian, because we have slightly different behavior for the two cases
        if len(jacobian.shape) == 4:
            # get the shape of the jacobian
            nr_lines_in_batch, nr_points_in_line, N, Z  = jacobian.shape # let's just call the last dimension Z for now
        
            # reshape jacobian to (batch, N*Z)
            jacobian = jacobian.view(nr_lines_in_batch, nr_points_in_line, N*Z)

            # calculate the differences between consecutive points
            nr_of_knots = torch.sum(torch.any(torch.abs((jacobian[:,1:,:] - jacobian[:,:-1,:]))>tolerance, dim=2), dim=1)
        else:
            # get the shape of the jacobian
            nr_points_in_line, N, Z  = jacobian.shape

            # reshape jacobian to (batch, N*Z)
            jacobian = jacobian.view(nr_points_in_line, N*Z)

            # calculate the differences between consecutive points
            nr_of_knots = torch.sum(torch.any(torch.abs((jacobian[1:,:] - jacobian[:-1,:]))>tolerance, dim=1))
    
    return nr_of_knots

# %% main knot density analysis function
def knot_density_analysis(ista: ISTA, nr_folds: int, A: torch.tensor, 
                            nr_paths: int = 4,  anchor_point_std: float = 1, nr_points_along_path: int = 4000, path_delta: float = 0.001, specified_anchors = None,
                            save_name: str = "test_figures", save_folder: str = "knot_density_figures", verbose: bool = False, color: str = "black",
                            tqdm_position: int = 0, tqdm_leave: bool = True, tolerance: float = 0, anchor_on_sphere: bool = False, boundary_type: str = "identity"):
    """
    We sample random lines in the y-space, and see how many knots we get over the iterations.
    We can then divide the knots by the lenght of the line, to get a knot-density.
    
    Given many different lines, we get an expectation of the knot-density over the y-space (as well as a variance).

    inputs:
    - ista: the ISTA module
    - nr_folds: the number of iterations
    - A: the matrix A in the equation y=Ax, of shape (M, N), with M<N, i.e. M is the measurement dimension and N is the signal dimension
    - nr_paths: the number of random paths to sample
    - anchor_point_std: the standard deviation of the anchor point, which is the point gnerated from a gaussian, which defines the middle of the line
    - nr_points_along_path: the number of points along the path
    - path_delta: the length of each step along the path, should be small for a good approximation
    - save_name: the name to save the figure as
    - save_folder: the folder to save the figure in
    - verbose: if True, print a progress bar
    - color: the color of the plot
    - tolerance: TODO
    """
    # empty the CUDA cache
    torch.cuda.empty_cache()
    
    # get M and N of the matrix A
    M = A.shape[0]
    N = A.shape[1] # NOSONAR

    # create the save folder if it does not exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # create a knot density array of size (nr_lines, K)
    knot_density_array = torch.zeros(nr_paths, nr_folds+1)

    # precalculate the lenght of the path
    length_of_path = path_delta * nr_points_along_path

    all_anchor_points = []
    # loop over the lines, with tqdm enabled if verbose is True
    for path_idx in tqdm(range(nr_paths), position=tqdm_position, leave=tqdm_leave, disable=not verbose, desc="knot density analysis, runnning over paths"):
        y, anchor_points = generate_path(M, nr_points_along_path, path_delta, anchor_point_std, ista, specified_anchors=specified_anchors, anchor_on_sphere=anchor_on_sphere)
        all_anchor_points.append(anchor_points)

        # run the initials function to get the initial x and jacobian
        x, jacobian = ista.get_initial_x_and_jacobian(nr_points_along_path, calculate_jacobian = True)

        # loop over the iterations
        for fold_idx in range(nr_folds):
            with torch.no_grad():
                x, jacobian = ista.forward_at_iteration(x, y, fold_idx, jacobian)

            # extract the number of knots in the jacobian, along the batch dimension
            nr_of_knots =  extract_knots_from_jacobian(jacobian, tolerance=tolerance, boundary_type=boundary_type)
            knot_density = nr_of_knots.cpu() / length_of_path
            knot_density_array[path_idx, fold_idx+1] = knot_density

    # take the mean along the paths
    knot_denity_mean = torch.mean(knot_density_array, dim=0)

    
    # plot the knot density over the iterations
    plt.figure()
    folds = np.arange(0,nr_folds+1)
    plt.plot(folds,knot_denity_mean,'-', label = "mean", color = color)
    plt.xlabel("fold")
    plt.ylabel("knot density")
    plt.grid()
    plt.title(save_name)
    plt.xlim([0,nr_folds])
    plt.tight_layout()
    plt.savefig(f"{save_folder}/{save_name}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_folder}/{save_name}.svg", bbox_inches='tight')
    plt.close()

    # give the knot density array back
    return knot_density_array, all_anchor_points