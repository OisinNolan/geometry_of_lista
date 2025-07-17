# The Geometry of LISTA
This codebases implements the metrics and experiments described in the paper "The Geometry of LISTA".

<img width="1402" height="222" alt="image" src="https://github.com/user-attachments/assets/b3a5ea77-9e2d-4407-8436-e12785a69c1f" />

### Visualizing and quantifying the geometry of LISTA and ISTA
Below we plot the _reconstruction sparsity_ of LISTA on a 2D plane anchored to training samples in the input space. That is, for each point $y$ on this plane, we plot $\lVert f(x) \rVert_0$, where $f$ is ISTA or LISTA. On the lower sub-plot, we plot the _decision density_ -- an empirical metric quantifying the complexity of ISTA / LISTA function in terms of how often it changes its decision about the support of the predicted output $\texttt{supp}(f(y))$. We observe that training LISTA with an L1 reconstruction loss leads to a simpler geometry, with similar reconstruction performance.
![all_iterations-ezgif com-optimize](https://github.com/user-attachments/assets/47f991b8-7e9b-408e-a36e-5669a2a5c2bf)

### Setup
First, clone the repository and navigate to the root directory. Then, build the docker image with the following command:
```bash
docker build -t ista_geom:latest .
```

### Reproduce the main results
You can reproduce the main results from the paper, including knot and decision density estimation, and 2D geometry visualization. The data, models, and plots will be saved in `./output` as per the command below.
Note that the script below selects the first GPU device with `--gpus device=0`, but you may want to change this.
```bash
docker run -it --name=ista_geom -v "$(pwd)":/app -w /app --rm --gpus device=0 ista_geom:latest bash reproduce_main_results.sh ./configs/base_config.yaml ./output
```

# Additional Results

### Varying the noise level $\sigma_n^2$
First, We re-run our experiments with varying noise standard deviation, including the noiseless case, with values $\sigma_n^2 \in \{0, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1\}$. Measuring the knot density and decision density at the final fold for each model, we find that training LISTA with the L1 objective results in a simpler decision density across noise levels.
<img width="2952" height="953" alt="data_that_stays_constant noise_std_plot" src="https://github.com/user-attachments/assets/a22aaec0-bf68-4107-8683-6b7709bd95bf" />



### Varying the size of $A$
Next, we re-run our experiment varying the size of the measurement vector $M$, with $M \in \{8, 10, 12, 14, 16\}$. Again, we find a consistently simpler decision geometry across $M$ for LISTA L1.
<img width="2952" height="953" alt="data_that_varies M min_plot" src="https://github.com/user-attachments/assets/34a094e4-4a00-4f7a-a87d-240e6526daa1" />



### Toeplitz sensing matrix
Finally, we consider a sensing matrix $A$ with Toeplitz structure. In particular, we sample an initial set of random Gaussian values for the first row and column of $A$, and repeat those diagonally to fill up the matrix. When examining tests losses and geometry, we find the results to be consistent with our main findings.
<img width="3552" height="1480" alt="knots_and_loss_toeplitz" src="https://github.com/user-attachments/assets/ed10c0ad-a73c-416d-bcae-92a28b14632d" />



### Reproducing additional results
**To reproduce the sweeps across $\sigma_n^2$ and $M$, run:**
```bash
docker run -it --name=ista_geom -v "$(pwd)":/app -w /app --rm --gpus device=0 ista_geom:latest bash reproduce_sweeps.sh ./configs/base_config.yaml ./output/sweeps
```

**To reproduce the results with the Toeplitz sensing matrix, run:**
```bash
docker run -it --name=ista_geom -v "$(pwd)":/app -w /app --rm --gpus device=0 ista_geom:latest bash reproduce_main_results.sh ./configs/toeplitz_config.yaml ./output
```
