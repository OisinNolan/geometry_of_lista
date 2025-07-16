**Proposition 1.** Let $f: \mathbb{R}^M \rightarrow \mathbb{R}^N$ be a CPWL function representing a sparse recovery algorithm (e.g., ISTA or LISTA), and let the sparsity level of the solution $|\mathbf{x}|_0$ be bounded by $K$. Then, the optimal number of projection regions induced by the least-squares solution over all possible supports is at most:

$$
\text{optimal number of projection regions} \leq \sum_{k=0}^{K} \binom{N}{k}.
$$



_Proof._ Since $f$ is a CPWL (continuous piecewise-linear) function, it is continuous and piecewise affine. Each input $\mathbf{y} \in \mathbb{R}^M$ is mapped to a unique output $\hat{\mathbf{x}} \in \mathbb{R}^N$ via exactly one affine function. This implies that for every $\mathbf{y}$, a unique support $S \subseteq \\{1, \dots, N\\}$ must be selected, corresponding to the non-zero entries of $\hat{\mathbf{x}}$.

Suppose we are given access to an oracle function that, for each $\mathbf{y}$, returns the optimal support $S^*$. Since the support is known, the LASSO problem can be bypassed, and the optimal solution is given by the least-squares estimate:

$$
\hat{\mathbf{x}}_S = \mathbf{A}_S^+ \mathbf{y},
$$

where $\mathbf{A}_S$ is the submatrix of $\mathbf{A}$ corresponding to the support $S$, and $\mathbf{A}_S^+$ is its Moore-Penrose pseudoinverse.

Now consider the set of all possible supports of size at most $K$. The total number of such supports is:

$$
\sum_{k=0}^{K} \binom{N}{k}.
$$

This represents the maximum number of distinct affine mappings that the oracle could assign across the entire input space $\mathbb{R}^M$, since if it assigned more than that, it would no longer be optimal nor an oracle. Since the function $f$ is single-valued, the oracle must assign exactly one support to each $\mathbf{y}$, and cannot assign more than one. Therefore, the number of distinct projection regions (each corresponding to a unique support) is upper bounded by the number of possible supports. (Note that this is an upper bound and not an equality. This is because the Oracle only has a maximum number of supports to assign and could e.g. never assign one of the supports as other supports are always more likely).

In practice, the actual number of regions realized by ISTA or LISTA may be much larger due to their iterative or learned structure, which can introduce many more affine pieces. However, the number of projection regions that are *optimal* in the sense of least-squares recovery with known support cannot exceed the number of possible supports. Hence, the bound holds.
