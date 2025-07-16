# The Geometry of LISTA
This codebases implements the metrics and experiments described in the paper "The Geometry of LISTA".

<img width="1402" height="222" alt="image" src="https://github.com/user-attachments/assets/cdb698b0-ac55-4c83-9065-3e00bdc4b924" />

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
<img width="2952" height="953" alt="data_that_stays_constant noise_std_plot" src="https://github.com/user-attachments/assets/82d2c524-4d97-4919-9b8e-a0ba3a7a9d92" />


### Varying the size of $A$
Next, we re-run our experiment varying the size of the measurement vector $M$, with $M \in \{8, 10, 12, 14, 16\}$. Again, we find a consistently simpler decision geometry across $M$ for LISTA L1.
<img width="2952" height="953" alt="data_that_varies M min_plot" src="https://github.com/user-attachments/assets/61cbb6e7-cee5-4831-a7bc-0c94b2ce549d" />


### Toeplitz sensing matrix
Finally, we consider a sensing matrix $A$ with Toeplitz structure. In particular, we sample an initial set of random Gaussian values for the first row and column of $A$, and repeat those diagonally to fill up the matrix. When examining tests losses and geometry, we find the results to be consistent with our main findings.
<img width="3552" height="1480" alt="knots_and_loss_toeplitz" src="https://github.com/user-attachments/assets/3440ff4e-658a-47b6-b3a0-02604748b698" />


### Reproducing additional results
**To reproduce the sweeps across $\sigma_n^2$ and $M$, run:**
```bash
docker run -it --name=ista_geom -v "$(pwd)":/app -w /app --rm --gpus device=0 ista_geom:latest bash reproduce_sweeps.sh ./configs/base_config.yaml ./output/sweeps
```

**To reproduce the results with the Toeplitz sensing matrix, run:**
```bash
docker run -it --name=ista_geom -v "$(pwd)":/app -w /app --rm --gpus device=0 ista_geom:latest bash reproduce_main_results.sh ./configs/toeplitz_config.yaml ./output
```
