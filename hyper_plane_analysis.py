"""
This script creates the functions used to analyze the linear regions of (RL)ISTA along a hyperplane.
"""
from pathlib import Path
import yaml
import torch
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
import os
import numpy as np
import warnings
from scipy.linalg import null_space
from distutils.util import strtobool

from ista import ISTA, load_model
from data_on_plane import DataOnPlane
import ncolor
from knot_density_analysis import binarize_jacobian

# %% helper functions
class MapToColors:
    def __init__(self, nr_colors: int, max_value:int = 1000000, max_nr_tries:int = 1000):
        # create a mapping from 0 to max_value to 0 to nr_colors
        # make it so that 0 maps to 0
        # all other values from 1 to max_value map to 1 to nr_colors in random order

        self.nr_colors = nr_colors
        self.max_value = max_value
        self.max_nr_tries = max_nr_tries

        # random permutation of the values from 1 to max_value to create as many possible mappings
        self.mapping = torch.cat([torch.zeros(1).long(), torch.randperm(self.max_value-1) + 1]).long()

        # ensure that the mapping loops around to 0
        self.mapping = self.mapping % (self.nr_colors-1) + 1

        # make sure that 0 maps to 0
        self.mapping[0] = 0     
        

    def __call__(self, x: torch.tensor):
        # x runs from 0 to some unknown number, first all values from 1 to that unknown number,
        # those we want to reararange in occurrence order, so that 1 is the largest region, 2 is the second largest region, and so on
        # but we ignore the zero region, which is the zero norm region
        # we can do this by using the mapping, which maps from 0 to max_value to 0 to nr_colors

        # make sure that the x is a long and on the cpu
        x = x.long().cpu()

        # create an edge map of the input
        edge_map_in_horizontal = x[:,1:] != x[:,:-1]
        edge_map_in_vertical   = x[1:,:] != x[:-1,:]

        # loop over the tries
        for i in range(self.max_nr_tries):
            # map the x to the colors
            x_out = self.mapping[x]

            # check that edges are perserved in the mapping
            edge_map_out_horizontal = x_out[:,1:] != x_out[:,:-1]
            edge_map_out_vertical   = x_out[1:,:] != x_out[:-1,:]

            # get the difference between the edge maps
            diff_horizontal = edge_map_in_horizontal != edge_map_out_horizontal
            diff_vertical   = edge_map_in_vertical != edge_map_out_vertical

            if torch.all(diff_horizontal == 0) and torch.all(diff_vertical == 0):
                break

            # change the mapping so that on of the differences is zero
            if torch.all(diff_horizontal == 0):
                look_at = diff_vertical
            else:
                look_at = diff_horizontal

            idx_x, idx_y = torch.where(look_at == 1)
            first_idx_x = idx_x[0]
            first_idx_y = idx_y[0]

            # swap the values at that index to a random value between 1 and nr_colors, in the hope that the edge is preserved next time
            value_in = x[first_idx_x, first_idx_y]
            self.mapping[value_in] = torch.randint(1, self.nr_colors, (1,))

        return x_out


def extract_linear_regions_from_jacobian(jacobian: torch.tensor, tolerance: float = None):
        """
        Given a Jacobian matrix, extract the linear regions from it. We also consider things the same region if they are within a certain tolerance of each other.
        This tolerance is achieved by converting the float to a long. Before we do this, we multiply is by 1/tolerance, and then convert it to a long. This way, we can use the unique function to find the unique entries.

        inputs:
        - jacobian (torch.tensor): the Jacobian of the forward function, of shape (batch_size, N, M) or (batch_size, N, 2) if jacobian_projection was used, we will call the last dimension Z
        - tolerance (float): the minimum l2 distance between the jacobian of regions to consider. Distances below this are considered the same region

        outputs:
        - nr_of_regions (int): the number of linear regions 
        - norms (torch.tensor of floats): the norm of the jacobian matrices in the same order as the linear regions of shape (batch_size)
        - unique_entries (torch.tensor): the unique entries of the jacobian, of shape (nr_of_regions, Z)
        - jacobian_labels (torch.tensor of longs): of shape (batch_size) with each a unique label for each region, norm==0 always has label 0
        """
        if tolerance is not None:
            assert tolerance > 0, "tolerance should be non-negative and non-zero"

        # if tolerence is non-zero, we need to multiply the jacobian by 1/tolerance and convert it to an integer
        if tolerance is not None:
            jacobian = torch.round((jacobian * 1/float(tolerance)))

        # get the shape of the jacobian
        batch_size, N, Z  = jacobian.shape # let's just call the last dimension Z for now

        # reshape jacobian to (batch, N*Z)
        jacobian = jacobian.view(batch_size, N*Z) 

        # calculate the norm of the jacobians
        norms = torch.linalg.norm(jacobian, ord=2, dim=1)

        # if tolerance is not none, cast it to a long now to speed up the unique function
        if tolerance is not None:
            jacobian = jacobian.long()

        # find the unique rows using consecutive on the sorted jacobian
        unique_entries, reverse_idxs = torch.unique(jacobian, dim=0, return_inverse=True)
        nr_of_regions  = len(unique_entries)
        
        # create the labels, but figure out where the zero norm is
        jacobian_labels = reverse_idxs + 1 # we add 1 to make sure the zero norm has label 0
        zero_norm = (norms < 1e-16)
        jacobian_labels[zero_norm] = 0
        
        return nr_of_regions, norms, unique_entries, jacobian_labels

def perform_pca_on_jacobian(jacobian: torch.tensor):
    """ 
    Perform PCA on the jacobian, and return only the first three principal components per input example.

    inputs:
    - jacobian (torch.tensor): the Jacobian of the forward function, of shape (batch_size, N, M) or (batch_size, N, 2) if jacobian_projection was used, we will call the last dimension Z

    outputs:
    - rgb_values (torch.tensor): the rgb values of the jacobian, of shape (batch_size, 3)
    """
    # step 1, reshape the jacobian to (batch_size, N*Z)
    batch_size, N, Z  = jacobian.shape # let's just call the last dimension Z for now
    jacobian = jacobian.view(batch_size, N*Z)

    # step 2, perform PCA on the jacobian
    pca = torch.linalg.svd(jacobian, full_matrices=False)
    principal_components = pca.Vh[:,:3]

    # step 3, project the jacobian to the first three principal components
    rgb_values = jacobian @ principal_components

    # step 4, normalize the rgb values, so that each channel is between 0 and 1
    rgb_values = rgb_values - rgb_values.min(dim=1).values.unsqueeze(1).repeat(1,3)
    rgb_values = rgb_values / rgb_values.max(dim=1).values.unsqueeze(1).repeat(1,3)

    return rgb_values

def create_y_from_projection(anchors: torch.tensor, nr_points_along_axis: int, margin: float = 0.5, max_magnitude: float = 1.0, symmetric: bool = False):
    """
    Given three anchor points, create a meshgrid of y values that forms the plane of the three anchor points.
    The meshgrid is of size (nr_points_along_axis, nr_points_along_axis)

    inputs:
    - anchors (torch.tensor): the anchor points, of shape (3, M)
    - nr_points_along_axis (int): the number of points along the axis
    - margin, by how much to extend both positive and negative along the axis
    - max_magnitude: the maximum magnitude of the anchor points
    - symmetric: if True, the meshgrid is symmetric around the origin

    outputs:
    - y (torch tensor): the points in a batch of shape (nr_points_along_axis*nr_points_along_axis, M)
    - Z1 (torch tensor): the first axis of the meshgrid, of shape (nr_points_along_axis, nr_points_along_axis)
    - Z2 (torch tensor): the second axis of the meshgrid, of shape (nr_points_along_axis, nr_points_along_axis)
    """
    # create the meshgrid in Z-space, which is the 2D space given by the anchor points
    if symmetric:
        line = torch.linspace(- (max_magnitude + margin), max_magnitude + margin, nr_points_along_axis)
    else:
        line = torch.linspace(- margin                  , max_magnitude + margin, nr_points_along_axis)
    Z1, Z2 = torch.meshgrid(line, line, indexing = 'ij')
    
    # reshape the mesh into a batch and create the three z values for the three anchor points
    z1 = Z1.reshape(-1)
    z2 = Z2.reshape(-1)

    # create the y values from the anchor points
    y = anchors[0,:] + z1.unsqueeze(1)*(anchors[1,:] - anchors[0,:]) + z2.unsqueeze(1)*(anchors[2,:] - anchors[0,:])

    # return the results
    return y, Z1, Z2

def create_anchors_from_x_indices(indices: tuple[int,int,int], A:torch.tensor, anchor_on_y_instead: bool = False):
    """
    creates anchor points given two indices that should be non-zero in the x-vector.

    inputs:
    - indices: which indices should be non-zero for the x-vectors used to create the three anchor points, if an index is None, that x-vector will be all zeros
    - A (torch.tensor): the matrix A in the equation y=Ax, of shape (M, N), with M<N, i.e. M is the measurement dimension and N is the signal dimension
    - anchor_on_y_instead: if False, create the anchor points from the x-indices, if True, create the anchor points from the y-indices

    outputs:
    - y_anchors (torch.tensor): the y-anchors, of shape (3, M)
    - x_anchors (torch.tensor): the x-anchors, of shape (3, N)
    """
    # extract the N size
    M, N = A.shape

    # create the anchor points
    x_anchors = torch.zeros(3, N, device=A.device)
    y_anchors = torch.zeros(3, M, device=A.device)
    for i in range(3):
        if indices[i] is not None:
            x_anchors[i, indices[i]] = 1

        y_anchors[i] = A @ x_anchors[i]

    # check if we overwrite the anchor points
    if anchor_on_y_instead:
        # create the anchor points from the x-indices
        x_anchors = torch.zeros(3, N)
        y_anchors = torch.zeros(3, M)
        for i in range(3):
            if indices[i] is not None:
                y_anchors[i, indices[i]] = 1

    return y_anchors, x_anchors

def create_jacobian_projection_from_anchors(anchors: torch.tensor):
    """
    Given three points that a plane needs to pass through, create a projection matrix that projects the Jacobian to a 2D space.

    inputs:
    - anchors (torch.tensor): the anchor points, of shape (3, M)

    outputs:
    - jacobian_projection (torch.tensor): the projection matrix, of shape (M, 2)
    """
    # create the projection matrix
    jacobian_projection = torch.zeros(anchors.shape[1], 2)

    # create the first vector
    jacobian_projection[:,0] = anchors[1,:] - anchors[0,:]

    # create the second vector
    jacobian_projection[:,1] = anchors[2,:] - anchors[0,:]


    # ensure the projection matrix is normalized
    jacobian_projection = torch.nn.functional.normalize(jacobian_projection, dim=0)

    return jacobian_projection

def extract_sparsity_label_from_x(x):
        """
        Given a bunch of x's, extract the sparsity label. That is to say, is the value zero, or non-zero?
        If x is 8 long, each element is either 0 or 1, then we can assign to each x an index from 0 to 2^8-1, i.e. 0 to 255
        This gives me regions of support of x, and I can assign a color to each region of support.

        input:
            x: the x's of shape (batch, N, 1)

        output:
            sparsity_label: the sparsity index of the x's of shape (batch) as in integer
            labels: the unique labels found in the x's
        """
        # step one, binarize the x's into 0 and non-zero
        x = (x!=0) # make sure to use a small value to avoid numerical issues

        # now use binary to decimal conversion
        sparsity_label = torch.sum(x[:,:]*2**(torch.arange(x.shape[1]).to(x.device)), dim=1).long()

        # get the unqiue labels
        labels = torch.unique(sparsity_label)

        return sparsity_label, labels


# %% visual analysis of ISTA along a hyperplane
def visual_analysis_of_ista(ista: ISTA, model_config: dict, hyperplane_config:dict, A: torch.tensor, save_folder: str = "test_figures", #NOSONAR
                            tqdm_position: int = 0, tqdm_leave: bool = True, verbose: bool = False, color_by: str = " norm", folds_to_visualize=[0, 1, 15, 31], 
                            train_dataset=None, plot_data=False, y_anchors=None, draw_path = True):
    """
    Creates a visual analysis of the ISTA module. This is done by visualizing the linear regions of the Jacobian, and the sparsity of the x-vector.
    We only visualize in part of the space, namely a hyperplane that passes through three anchor points. This hyperplane is embedded in y-space,
    which is the actual input space of the ISTA module. We project the Jacobian to a 2D space, and visualize the linear regions in this 2D space.

    inputs:
    - ista: the ISTA module
    - model_config: the configuration of the model with:
        - nr_folds: the number of iterations
    
    - hyperplane_config: the configuration of the hyperplane with:
        - nr_points_along_axis:     defaults to 1024    the number of points along the axis of the hyperplane
        - margin:                   defaults to 0.5     the margin around the hyperplane to visualize
        - indices_of_projection:    defaults to [~,0,1] the indices of the anchor points, A none means the origin, a 0 means x=[1,0,0,0,..] and a 1 means x=[0,1,0,0,..], and so on.
        - magntiude:                defaults to 1       the magnitude of the anchor points, default is 1.0
        - tolerance:                defaults to None    the minimum difference in jacboian to consider at all. if tolerance is e.g. 0.01 differnces smaller than that are trunctated 
        - color_by:                 defaults to "norm"  what to color the regions by, either "norm" or "jacobian_label" or "jacobian_pca"
        - draw_decision_boundary:   defaults to False   if True, draw the decision boundary of the sparsity of the x-vector
        - plot_data_regions:        defaults to False   if True, plot the data regions in the 2D space


    - A: the matrix A in the equation y=Ax, of shape (M, N), with M<N, i.e. M is the measurement dimension and N is the signal dimension
    - save_folder: the folder to save the figures in
    - tqdm_position: the position of the tqdm bar
    - tqdm_leave: if True, leave the tqdm bar
    - verbose: if True, print a progress bar
    - folds_to_visualize: a list of fold_idxs to visualize. If empty, all folds will be visualized.
    """
    # get the model config
    nr_folds = model_config["nr_folds"]

    # get the hyperplane config, with hasattr to allow for defaults
    nr_points_along_axis    = hyperplane_config.get("nr_points_along_axis", 1024)
    margin                  = hyperplane_config.get("margin", 0.5)
    indices_of_projection   = hyperplane_config.get("indices_of_projection", [None,0,1])
    anchor_on_y_instead     = hyperplane_config.get("anchor_on_y_instead", False)
    magntiude               = hyperplane_config.get("magnitude", 1.0)
    tolerance               = hyperplane_config.get("tolerance", None)
    draw_decision_boundary  = hyperplane_config.get("draw_decision_boundary", False)
    plot_data_regions       = hyperplane_config.get("plot_data_regions", False)
    data_region_extend      = hyperplane_config.get("data_region_extend", [0.5, 1.5])
    K                       = hyperplane_config.get("K", 4)
    symmetric               = hyperplane_config.get("symmetric", False)
    only_positive           = hyperplane_config.get("only_positive", True)

    # create the save folder if it does not exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:
        # empty the folder of its contents
        for file in os.listdir(save_folder):
            os.remove(f"{save_folder}/{file}")

    if train_dataset is not None:
        train_y = train_dataset.y
        train_x = train_dataset.x

    # figure out if we need to up to magnitude
    max_magnitude = magntiude

    if y_anchors is None:
        # we create a projection matrix that projects the jacobian to a 2d space, for visualization, this is done by specifying three anchor points
        # anchor point 0 is where the x-vector is [0,0,0,...,0]
        y_anchors, _ = create_anchors_from_x_indices(indices_of_projection, A, anchor_on_y_instead= anchor_on_y_instead)
    
    # create the projection matrix
    jacobian_projection = create_jacobian_projection_from_anchors(y_anchors)

    # create y data from the projection
    y,Z1,Z2 = create_y_from_projection(y_anchors, nr_points_along_axis, margin = margin, max_magnitude = max_magnitude, symmetric = symmetric)

    # create the xmin, xmax, ymin, ymax
    if symmetric:
        xmin = -max_magnitude - margin
        ymin = -max_magnitude - margin
    else:
        xmin = -margin
        ymin = -margin

    xmax = max_magnitude + margin
    ymax = max_magnitude + margin

    # run the initials function to get the initial x and jacobian
    x, jacobian = ista.get_initial_x_and_jacobian(y.shape[0], calculate_jacobian = True, jacobian_projection = jacobian_projection)

    # create an array of nr regions over the iterations
    nr_regions_arrray = torch.zeros(nr_folds)

    if hyperplane_config.get('binarize', False) == True:
        color_by = "jacobian_binary"

    # if we are using jacboian labels, we need to create a map to colors object
    if color_by == "jacobian_label":
        map_to_colors = MapToColors(20, max_value=10000000)

    # if we want to plot the data regions, we need to create a function that does this  
    if plot_data_regions:
        # we transpose the y_anchors to fit the plane such that the path makes a right angle in the plot
        data_on_plane = DataOnPlane(A.cpu(), data_region_extend, y_anchors[[1, 0, 2]].cpu(), K=K, consider_partials=False, only_positive=only_positive)
        # data_on_plane = DataOnPlane(A.cpu(), data_region_extend, y_anchors.cpu(), K=K, consider_partials=False, only_positive=only_positive)

    add_colorbar = False
    vmax = None

    # loop over the iterations
    for fold_idx in tqdm(range(nr_folds), position=tqdm_position, leave=tqdm_leave, disable=not verbose, desc="visual analysis of ISTA, runnning over folds"):
        try:
            with torch.no_grad():
                x, jacobian = ista.forward_at_iteration(x, y, fold_idx, jacobian, jacobian_projection)

            if fold_idx in folds_to_visualize:
                if hyperplane_config.get('binarize', False) == True:
                    jacobian = binarize_jacobian(jacobian)
                # extract the linear regions from the jacobian
                nr_of_regions, norms, _, jacobian_labels = extract_linear_regions_from_jacobian(jacobian, tolerance = tolerance)       

                # extract the sparsity label from x
                sparsity_label, unique_labels = extract_sparsity_label_from_x(x)
                sparsity_label_reshaped = sparsity_label.reshape(nr_points_along_axis, nr_points_along_axis)

                # compress the sparsity label to the unique labels
                sparsity_label_reshaped = sparsity_label_reshaped.unique(return_inverse=True)[1].reshape(nr_points_along_axis, nr_points_along_axis)
                unique_labels = sparsity_label_reshaped.unique()

                # figure out what to color by
                if color_by == "norm":
                    norms_reshaped = norms.reshape(nr_points_along_axis, nr_points_along_axis)
                    norms_reshaped = torch.log(norms_reshaped + 1) 
                    color_data = norms_reshaped.cpu()
                    cmap = 'cividis'
                elif color_by == "jacobian_label":
                    jacobian_labels_reshaped = jacobian_labels.reshape(nr_points_along_axis, nr_points_along_axis)
                    color_data = map_to_colors(jacobian_labels_reshaped).cpu()
                    cmap = 'tab20'
                elif color_by == "jacobian_pca":
                    color_data = perform_pca_on_jacobian(jacobian)
                    color_data = color_data.cpu()
                    color_data = color_data.reshape(nr_points_along_axis, nr_points_along_axis, 3)
                    cmap = None
                elif color_by == "jacobian_binary":
                    binary_jacobian = binarize_jacobian(jacobian)
                    row_means = binary_jacobian.mean(dim=2)
                    fraction_of_1_rows = row_means.sum(dim=1)
                    color_data = fraction_of_1_rows.reshape(nr_points_along_axis, nr_points_along_axis).cpu()

                    # Create custom colormap
                    magma = plt.cm.get_cmap('magma_r')
                    # Get colors for values 1-8
                    colors = magma(np.linspace(0, 1, 10))  # 9 colors for values 1->8
                    # Add darkest color for values > 8
                    darkest_color = magma(1.0)  # Get the darkest color from magma
                    # Create the custom colormap
                    custom_cmap = ListedColormap(np.vstack((colors, darkest_color)))
                    # Clip values to range 1-9 (9 represents >8)
                    color_data = np.clip(color_data, 1, 9)

                    cmap = custom_cmap
                    vmax = 10 
                    add_colorbar = True
                else:
                    raise ValueError("color_by should be either 'norm' or 'jacobian_label'")

                # create the three names of the anchor points
                anchor_names = ["anchor x-index: "] * 3
                for i in range(3):
                    anchor_names[i] += str(indices_of_projection[i]) if indices_of_projection[i] is not None else "origin"

                # # plot the results
                dpi = matplotlib.rcParams['figure.dpi']
                nr_pixels_along_axis = nr_points_along_axis
                fig_size_along_axis = nr_pixels_along_axis / dpi

                fig = plt.figure(figsize=(fig_size_along_axis, fig_size_along_axis))
                # fig = plt.figure(figsize=(14, 10))
                ax = fig.add_axes([0, 0, 0.85, 1])
                ax.axis('off')
                if cmap is None:
                    im = ax.imshow(color_data, extent=[xmin, xmax, ymin, ymax], origin="lower", zorder = -10)
                else:
                    vmax = color_data.max() if vmax is None else vmax
                    im = ax.imshow(color_data, extent=[xmin, xmax, ymin, ymax], cmap = cmap, vmin = 0, vmax = vmax, origin="lower", zorder = -10)

                # Add colorbar if it was created
                if add_colorbar:
                    boundaries = np.arange(-0.5, vmax + 0.5, 1)  # boundaries between integers
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.2)
                    colorbar = plt.colorbar(
                        im, 
                        cax=cax,
                        ticks=np.arange(0, 10),  # include 8 in the ticks
                        boundaries=boundaries,
                        spacing='proportional',
                        format='%1i'  # format as integers
                    )
                    colorbar.ax.invert_yaxis()
                    colorbar.ax.tick_params(labelsize=18)

                    # Get the tick labels and modify the last one to show '>8'
                    ticks = colorbar.get_ticks()
                    tick_labels = [f'{int(tick)}' for tick in ticks[:-1]]  # Convert all but last tick to integers
                    tick_labels.append('>8')  # Add '>8' for the last tick
                    colorbar.ax.set_yticklabels(tick_labels)

                    colorbar.set_label('Support Cardinality', rotation=270, labelpad=25, fontsize=20)

                # if plot_data_regions:
                #     data_on_plane.plot_data_regions(show_legend=False, colors = ["white","white","white"], ax = ax)

                if plot_data:
                    # dim_x = indices_of_projection[1]
                    # dim_y = indices_of_projection[2]
                    
                    # we would like to plot each of the training samples on the hyperplane
                    # first, we need to change the basis of the training data to match the hyperplane
                    ys_on_hyperplane = torch.tensor([np.array(data) for y in train_y if (data := data_on_plane.y_to_hyperplane_coordinates(y, tolerance=0.01)) is not None])
                    ax.scatter(ys_on_hyperplane[:, 0].cpu(),  # x coordinates
                        ys_on_hyperplane[:, 1].cpu(),  # y coordinates
                        c='white',                # white markers
                        s=100,                     # marker size
                        edgecolor='red',          # red border
                        linewidth=0.5,
                        zorder=5,                 # layer above imshow
                        marker='.') 
                # else: 
                #     # scatter three points, at 0,0 and 1,0 and 0,1 and put a legen with the anchor points
                #     plt.scatter(0,         0,         c= 'white', label = anchor_names[0], zorder = 10, marker='x', s = 50)
                #     plt.scatter(magntiude, 0,         c= 'white', label = anchor_names[1], zorder = 10, marker='o', s = 50)
                #     plt.scatter(0,         magntiude, c= 'white', label = anchor_names[2], zorder = 10, marker='s', s = 50)
                #     plt.legend()

                # # put a contour plot around the sparsity labels
                if draw_decision_boundary:
                    ax.contour(Z2, Z1, sparsity_label_reshaped.cpu(), levels=unique_labels.cpu(), colors='white', linewidths=0.1, linestyles='solid', extent=[xmin, xmax, ymin, ymax], zorder = 1, origin="lower")

                if draw_path:
                    path = np.array([np.array(data_on_plane.y_to_hyperplane_coordinates(y_vector)) for y_vector in y_anchors])
                    # ax.plot(path[:,0], path[:,1], color = "white", zorder = 2, linewidth = 3, marker="x", markersize=8.5, markeredgewidth=3)
                    
                    # Double-plot to get outline effect
                    ax.plot(path[:, 0], path[:, 1], 
                            marker="o", markersize=15,  # Larger size for the "fill"
                            color="black", linewidth=0, 
                            markeredgewidth=8, zorder=2, linestyle="")

                    # Thin black marker (border)
                    ax.plot(path[:, 0], path[:, 1], 
                            marker="o", markersize=12,  # Slightly smaller size for the border
                            color="white", linewidth=0, 
                            markeredgewidth=3, zorder=3, linestyle="") 


                # x and y limits
                ax.set_xlim([xmin + 0.5, xmax])
                ax.set_ylim([ymin + 0.5, ymax])

                # for im in ax.get_images():  
                #     im.set_rasterized(True)

                # save the figure
                outpath = f"{save_folder}/iteration_{fold_idx}.png"
                plt.savefig(outpath, bbox_inches='tight', pad_inches=0.1, dpi=200)
                plt.close()
                print(f"✅ Saved to {outpath}\n")

                # save the number of regions
                nr_regions_arrray[fold_idx] = nr_of_regions
        
        except RuntimeError as e:
            nr_regions_arrray[fold_idx] = np.nan
            print(f"Skipped fold {fold_idx} due to runtime error")
            print(f"Error: {e}")
    
    # plot the number of regions over the iterations
    plt.figure()
    plt.plot(nr_regions_arrray,'-')
    plt.xlabel("iteration")
    plt.ylabel("number of linear regions")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{save_folder}/nr_regions_over_iterations.jpg", dpi=300, bbox_inches='tight') # save the figure as a jpg
    plt.savefig(f"{save_folder}/nr_regions_over_iterations.svg", bbox_inches='tight')          # save the figure as a svg
    plt.close()
    print(f"Hyperplane analysis saved to {save_folder}.")

    return nr_regions_arrray

def is_1sparse(anchor):
    """
    Determines if the given anchor is 1-sparse.

    An anchor is considered 1-sparse if it contains exactly one nonzero element.
    This implementation assumes that the anchor is a torch tensor representing a vector.

    Args:
        anchor (torch.Tensor): A 1D tensor.

    Returns:
        bool: True if the anchor is 1-sparse, False otherwise.
    """
    return (anchor != 0).sum().item() == 1

def get_first_k_1sparse_anchors(anchors, k):
    """
    Returns any set of k 1-sparse anchors from a torch tensor of vectors.

    The function assumes `anchors` is a 2D torch tensor of shape (num_anchors, vector_length),
    where each row is an anchor vector. It selects rows that are 1-sparse.

    Args:
        anchors (torch.Tensor): A 2D tensor with anchor vectors.
        k (int): Number of 1-sparse anchors to find.

    Returns:
        torch.Tensor: A tensor containing k selected 1-sparse anchors.

    Raises:
        ValueError: If fewer than k 1-sparse anchors are found.
    """
    # Create a boolean mask for anchors that are 1-sparse
    sparse_mask = (anchors.x != 0).sum(dim=1) == 1
    y_anchors = anchors.y[sparse_mask]
    return y_anchors[:k]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", type=str,
                      help="Path to experiment root directory")
    parser.add_argument("--run_id", type=str, default="0")
    parser.add_argument("--anchor_on", type=str, default="train", choices=['train', 'test'],
                      help="Whether to anchor on samples from the train or test set")
    parser.add_argument("--model_type", type=str, nargs='+', default=["LISTA"], choices=['LISTA', 'ISTA'],
                      help="Model type(s) to process (can be a list)")
    parser.add_argument("--decision_density", type=lambda x: bool(strtobool(x)), default=True,
                      help="If True, use decision density (sparsity map) instead of jacobian label")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    EXPERIMENT_ROOT = Path(args.experiment_root)
    RUN_ID = args.run_id
    DATASET_ROOT = EXPERIMENT_ROOT / RUN_ID
    DECISION_DENSITY = args.decision_density
    DECISION_BOUNDARIES = True
    FOLDS_TO_VISUALIZE = [0, 9, 99, 1023]
    NR_PATHS = 10

    experiment_run_path = EXPERIMENT_ROOT / RUN_ID        
    with open(EXPERIMENT_ROOT / "config.yaml", 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config['Hyperplane']['binarize'] = DECISION_DENSITY
    config['Hyperplane']['draw_decision_boundary'] = DECISION_DENSITY
    config['Hyperplane']['margin'] = 0
    config['Hyperplane']['symmetric'] = True

    datasets = {
        'test': torch.load(DATASET_ROOT / "data/test_data.tar"),
        'train': torch.load(DATASET_ROOT / "data/train_data.tar"),
    }

    for model_type in args.model_type:
        print(f"Running for {model_type}")

        OUTDIR = EXPERIMENT_ROOT / RUN_ID / model_type / ("hyperplane/sparsity" if DECISION_DENSITY else "hyperplane/jacobian_label")

        model = load_model(config, model_type, experiment_run_path / f"{model_type}/{model_type}_state_dict.tar", experiment_run_path / "A.tar", train_dataset = datasets["train"], experiment_run_path=experiment_run_path)
        if model_type == "ISTA":
            model.train_inputs = datasets['test'].y

        y_anchors = get_first_k_1sparse_anchors(datasets[args.anchor_on], 3)
        visual_analysis_of_ista(model, config[model_type], config["Hyperplane"], model.A.cpu(), save_folder = OUTDIR, tqdm_position=1, verbose = True, color_by="jacobian_label", folds_to_visualize=FOLDS_TO_VISUALIZE, draw_path=DECISION_BOUNDARIES, y_anchors=y_anchors)
