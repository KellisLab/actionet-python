# Batch Orthogonalization Algorithm in ACTIONet

## Overview

The `correct_batch_effect()` function removes batch-associated variation from a low-rank representation of single-cell data. Unlike methods that modify expression values directly, this approach operates in the reduced latent space, making it both mathematically principled and computationally efficient.

The core insight is that batch effects can be modeled as low-rank perturbations to the data. By orthogonalizing the latent space with respect to batch covariates and applying a **perturbed SVD update**, the algorithm removes batch effects without recomputing the full SVD from scratch.

---

## Mathematical Formulation

### Notation

| Symbol | Dimension | Description |
|--------|-----------|-------------|
| $S$ | $g \times n$ | Expression matrix (genes × cells) |
| $D$ | $n \times b$ | Design matrix (cells × batch indicators) |
| $U, \Sigma, V$ | Various | SVD of $S$: $S \approx U\Sigma V^\top$ |
| $k$ | scalar | Number of latent components |
| $Z$ | $g \times b$ | Batch subspace basis |

### Step 1: Construct the Batch Subspace

The batch design matrix $D$ encodes batch membership. For categorical batch labels, this is typically a one-hot encoding:

$$
D_{ij} = \begin{cases} 1 & \text{if cell } i \text{ belongs to batch } j \\ 0 & \text{otherwise} \end{cases}
$$

We compute the batch-associated signal in feature space:

$$
Z = S \cdot D \quad \in \mathbb{R}^{g \times b}
$$

Each column of $Z$ represents the summed expression profile across cells in that batch.

### Step 2: Gram-Schmidt Orthonormalization

The columns of $Z$ are orthonormalized via classical Gram-Schmidt:

$$
Z \leftarrow \text{orth}(Z)
$$

This produces an orthonormal basis for the subspace spanned by batch effects. The implementation:

```cpp
void gram_schmidt(arma::mat &A) {
    for (uword i = 0; i < A.n_cols; ++i) {
        for (uword j = 0; j < i; ++j) {
            double r = dot(A.col(i), A.col(j));
            A.col(i) -= r * A.col(j);
        }
        double col_norm = norm(A.col(i), 2);
        if (col_norm < 1E-4) {
            // Near-zero columns are zeroed out
            for (uword k = i; k < A.n_cols; ++k)
                A.col(k).zeros();
            return;
        }
        A.col(i) /= col_norm;
    }
}
```

### Step 3: Define the Low-Rank Deflation

We define a rank-$b$ correction that will remove batch variation:

$$
A = Z \quad \in \mathbb{R}^{g \times b}
$$

$$
B = -Z^\top S \quad \in \mathbb{R}^{n \times b}
$$

The intuition: $AB^\top = -Z(Z^\top S)^\top = -ZS^\top Z$ represents the component of the data that lies in the batch subspace. By subtracting this from the decomposition, we orthogonalize out batch effects.

### Step 4: Deflation with Intercept Adjustment

Before applying the perturbed SVD, we augment the correction matrices to handle mean effects:

$$
\mu_A = \text{mean}(A, \text{axis}=1)
$$

$$
\mu = B \cdot \mu_A
$$

$$
A \leftarrow [1 \mid A] \quad \text{(prepend column of ones)}
$$

$$
B \leftarrow [-\mu \mid B] \quad \text{(prepend negative mean)}
$$

This converts a rank-$b$ update into a rank-$(b+1)$ update that also centers the correction.

### Step 5: Perturbed SVD (Rank-k Update)

Given the existing SVD $S \approx U\Sigma V^\top$ and the low-rank update defined by $A$ and $B$, we efficiently compute the updated decomposition.

#### 5.1 Project onto Orthogonal Complements

Compute the components of $A$ and $B$ that lie outside the current subspaces:

$$
M = U^\top A \quad \text{(projection of } A \text{ onto } U\text{)}
$$

$$
A_\perp = A - U M \quad \text{(orthogonal complement)}
$$

$$
N = V^\top B \quad \text{(projection of } B \text{ onto } V\text{)}
$$

$$
B_\perp = B - V N \quad \text{(orthogonal complement)}
$$

#### 5.2 Orthonormalize the Complements

$$
P = \text{orth}(A_\perp), \quad R_P = P^\top A_\perp
$$

$$
Q = \text{orth}(B_\perp), \quad R_Q = Q^\top B_\perp
$$

#### 5.3 Build the Core Matrix

Construct a small $(k + c) \times (k + c)$ matrix (where $c$ is the number of correction columns):

$$
K = \begin{bmatrix} \Sigma & 0 \\ 0 & 0 \end{bmatrix} + \begin{bmatrix} M \\ R_P \end{bmatrix} \begin{bmatrix} N \\ R_Q \end{bmatrix}^\top
$$

This captures how the update interacts with the existing decomposition.

#### 5.4 SVD of Core Matrix

$$
K = U_p \Sigma_p V_p^\top
$$

This is a small dense SVD of dimension $(k + c) \times (k + c)$.

#### 5.5 Expand to Full Factors

$$
U_{\text{new}} = [U \mid P] \cdot U_p
$$

$$
V_{\text{new}} = [V \mid Q] \cdot V_p
$$

Keep the leading $k$ columns to match the original reduction dimension.

---

## Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| $Z = S \cdot D$ | $O(nnz \cdot b)$ | Sparse-aware for sparse $S$ |
| Gram-Schmidt on $Z$ | $O(g \cdot b^2)$ | $b$ is typically small |
| $B = Z^\top S$ | $O(nnz \cdot b)$ | Sparse-aware |
| Projections $M, N$ | $O(gk \cdot c + nk \cdot c)$ | $c = b + 1$ |
| Core SVD | $O((k+c)^3)$ | Dense, but small |
| Expand factors | $O(g(k+c)k + n(k+c)k)$ | Matrix multiplications |

**Key advantage**: The algorithm avoids recomputing the full SVD of $S$. For large $g$ and $n$, this provides substantial savings when $k$ and $b$ are small.

---

## Algorithm Summary

```
Algorithm: orthogonalizeBatchEffect(S, SVD_results, D)
─────────────────────────────────────────────────────
Input:
    S           : Expression matrix (genes × cells)
    SVD_results : {U, σ, V, A_prev, B_prev} from prior reduction
    D           : Design matrix (cells × batches)

Output:
    Updated SVD with batch effects removed

1. Z ← S × D                          // Batch signals
2. Z ← gram_schmidt(Z)                // Orthonormalize
3. A ← Z                              // Feature-side correction
4. B ← −(Z^T × S)^T                   // Cell-side correction
5. μ_A ← mean(A, axis=1)              // Column means
6. μ ← B × μ_A                        // Intercept adjustment
7. A ← [ones | A]                     // Augment with intercept
8. B ← [−μ | B]                       // Augment correction
9. return perturbedSVD(SVD_results, A, B)
```

---

## Geometric Interpretation

The algorithm can be understood geometrically:

1. **Batch Subspace**: The design matrix $D$ defines which cells belong to which batch. Multiplying by $S$ maps this into feature space, identifying directions correlated with batch.

2. **Orthonormalization**: Gram-Schmidt produces an orthonormal basis for this batch subspace.

3. **Projection Removal**: The low-rank update $AB^\top$ represents the projection of data onto the batch subspace. Subtracting this from the SVD effectively "deflates" the batch component.

4. **Efficient Update**: Rather than recomputing everything, the perturbed SVD exploits the structure of the update to work in a small $(k+c)$-dimensional space.

---

## Implementation Notes

### Python Interface

```python
import actionet as an

# Batch correction using categorical labels
an.correct_batch_effect(adata, batch_key="batch")

# Or with a custom design matrix
design = my_design_matrix  # shape: (n_cells, n_covariates)
an.correct_batch_effect(adata, design=design)
```

### Storage Convention

The function stores results in:
- `adata.obsm[corrected_key]`: Corrected cell embeddings
- `adata.varm[f"{corrected_key}_V"]`: Feature loadings
- `adata.varm[f"{corrected_key}_A"]`: Accumulated A matrices
- `adata.obsm[f"{corrected_key}_B"]`: Accumulated B matrices
- `adata.uns[f"{corrected_key}_params"]`: Singular values and metadata

### Numerical Stability

- Gram-Schmidt includes a tolerance check ($\|v\| < 10^{-4}$) to handle near-collinear batch indicators.
- The perturbed SVD works with the small core matrix, avoiding ill-conditioning in the full space.

---

## References

1. Brand, M. (2006). "Fast low-rank modifications of the thin singular value decomposition." *Linear Algebra and its Applications*, 415(1), 20-30.

2. Bunch, J. R., & Nielsen, C. P. (1978). "Updating the singular value decomposition." *Numerische Mathematik*, 31(2), 111-129.

3. Mohammadi, S., et al. (2020). "A geometric approach to characterize the functional identity of single cells." *Nature Communications*, 11, 5890.
