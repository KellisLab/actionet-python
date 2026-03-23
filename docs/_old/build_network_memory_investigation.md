# `build_network()` Memory Investigation

## Scope

This note investigates why `build_network()` is the next major out-of-memory point after the backed `reduce_kernel()` refactor, and what changes are most likely to make the pipeline viable for datasets in the 1.7M to 10M observation range.

The investigation is based on the current Python wrapper and C++ implementation in:

- `src/actionet/core.py`
- `src/actionet/wp_network.cpp`
- `src/actionet/wp_utils.cpp`
- `src/libactionet/src/network/build_network.cpp`
- `src/libactionet/include/network/hnsw_imp.hpp`
- `src/libactionet/include/extern/hnswlib/hnswalg.h`


## Executive Summary

1. The current default path, `algorithm="k*nn"`, is not viable at 1.7M observations and is completely incompatible with the 10M target.
2. The immediate cause is not HNSW alone. The `k*nn` implementation materializes several dense `N x kNN` double matrices, where `kNN = 5 * round(sqrt(N))`.
3. At 1.7M observations, the current default implies `kNN = 6520`, so the dense neighbor matrices (`idx`, `dist`) are each allocated at shape `(N, kNN+1) = (1700000, 6521)`, about `88.7 GB`. The code builds at least five such matrices (`idx`, `dist`, `beta`, `lambda`, `Delta`), which is already about `443 GB` before accounting for the HNSW index, the input matrix copies, triplets, sparse graph copies, and Python conversion overhead.
4. Several exposed parameters do not help with this OOM:
   - `network_density` does not reduce the size of the large dense matrices.
   - `k` is ignored for `k*nn`.
   - `ef` and `ef_construction` are ignored for `k*nn`.
5. The fastest path to a usable large-scale pipeline is:
   - stop using `k*nn` for large `N`
   - use fixed-degree `knn`
   - build the network from a lower-dimensional representation (`H_merged` or `action`) when acceptable
   - remove wrapper-side dense copies and downcast to `float32`
   - rewrite graph assembly to stream directly into CSR/CSC instead of repeatedly duplicating edge storage
6. With those changes, 10M observations becomes plausible. Without them, it does not.


## Current Code Path

### Python wrapper path

`run_actionet()` calls `build_network()` on `adata.obsm["H_stacked"]`.

Current path:

1. `src/actionet/core.py`
   - reads `adata.obsm[obsm_key]`
   - transposes and copies it via `np.ascontiguousarray(H.T)`
2. `src/actionet/wp_network.cpp`
   - converts the NumPy array to `arma::mat`
3. `src/libactionet/src/network/build_network.cpp`
   - copies `arma::mat H` by value into the selected builder
   - converts `H` to `arma::fmat X` for HNSW
4. final sparse graph is converted back to SciPy CSR through another copy-heavy path in `src/actionet/wp_utils.cpp`

This means the stage currently pays for several dense copies before the nearest-neighbor work even starts.


## Important Findings From The Code

### 1. `k*nn` scales neighbor search as `sqrt(N)`

In `buildNetwork_KstarNN()`:

```cpp
double kappa = 5.0;
int kNN = std::min(sample_no - 1, (int)(kappa * round(std::sqrt(sample_no))));
```

So the default adaptive graph uses:

- `N = 1,700,000` -> `kNN = 6520` (exact: `5 * round(sqrt(1700000)) = 5 * 1304 = 6520`)
- `N = 10,000,000` -> `kNN = 15811` (exact: `5 * round(sqrt(10000000)) = 5 * 3162 = 15810`; confirm with implementation)

This is already too large for the current implementation.

### 2. `network_density` is not a memory lever in the current `k*nn` implementation

`density` only changes `LC = 1.0 / density` and therefore affects the adaptive thresholding math. It does **not** reduce:

- `kNN`
- `idx`
- `dist`
- `beta`
- `lambda`
- `Delta`

So lowering `network_density` does not fix the OOM.

### 3. `k`, `ef`, and `ef_construction` are effectively ignored for `k*nn`

The public Python interface exposes these parameters, but the `k*nn` implementation sets:

```cpp
ef_construction = ef = kNN;
```

and does not use the user-provided `k`.

So the usual HNSW tuning knobs are only meaningful for the `knn` path.

### 4. The wrapper currently makes unnecessary dense copies

These copies apply to **both** the `knn` and `k*nn` paths. The `knn` path is not exempt.

For the current default path:

- Python makes a transposed contiguous double array
- pybind copies that into an Armadillo double matrix via an element-by-element row-to-column-major loop (no BLAS or `memcpy`; a serial O(N × d) loop)
- the C++ builder copies that matrix by value
- the builder converts again to `arma::fmat`

At large `N`, this alone costs many GB.

### 5. Graph assembly duplicates edge storage several times

Both `knn` and `k*nn` currently:

- collect triplets in `std::vector`s
- convert those triplets to `arma::umat locations` and `arma::vec values`
- build `arma::sp_mat G`
- create `Gt = G.t()`
- create `G_sym`
- convert the result back to SciPy through a COO → CSR path in `arma_sparse_to_scipy` (three C++ `std::vector`s → three NumPy arrays → `coo_matrix` → `.tocsr()`; both COO and CSR are simultaneously live during the final conversion)

This is tolerable at 40k cells. It is wasteful at 1.7M, and becomes a real design limit at 10M.

Additionally, `knn` stores neighbor distances as `double` in its triplet vectors (`std::vector<double>`), even though HNSW returns `float`. The widening is unnecessary and doubles the per-edge storage in the triplet phase.

### 6. JSD normalisation creates transient dense copies of the full input matrix

Both `buildNetwork_KstarNN` and `buildNetwork_KNN` apply the JSD pre-processing path:

```cpp
H = arma::clamp(H, 0, 1);
H = arma::normalise(H, 1, 0);
```

Each of `arma::clamp` and `arma::normalise` returns a new `arma::mat` that is then move-assigned back into `H`. At the peak of each call, both the old and new matrices are simultaneously live. This creates a transient additional dense copy of `H` (double-precision, shape `d × N`) per operation.

At `N = 1.7M`, `d = 464`, each such transient copy is about `6.31 GB`. For `k*nn` this is dwarfed by the `idx`/`dist` allocation; for the `knn` path it is the dominant transient overhead beyond the four structural copies.

This is separate from — and in addition to — the four structural input copies described in Finding #4.


## Memory Model

The exact peak depends on allocator behavior and overlap between temporaries, but the code allows straightforward lower-bound estimates.

### Notation

- `N` = number of observations
- `d` = width of the network input representation
- `E` = number of directed edges retained before symmetrization

For default `run_actionet(k_min=2, k_max=30)`, an upper bound on `H_stacked` width is:

`sum_{k=2}^{30} k = 464`

In practice `H_stacked` may be smaller after pruning. `H_merged` and `action` are usually much smaller still. The dimensional estimates below use `d = 464` as a conservative worst-case for the current default representation.


## Why The 1.7M Run OOMs Today

### Current default: `k*nn` on `H_stacked`

For `N = 1,700,000`:

- `kNN = 6520`
- each dense `N x (kNN + 1)` double matrix (`idx` and `dist` are allocated at shape `(N, kNN+1)`) is about:
  - `1.7e6 * 6521 * 8 ~= 88.7 GB`

The `k*nn` code materializes at least these dense matrices:

- `idx`
- `dist`
- `beta`
- `lambda`
- `Delta`

That is already:

- `5 * 88.7 GB ~= 443.4 GB`

and this excludes:

- the HNSW index
- the dense `H` copies
- the `arma::fmat X` copy
- triplet vectors
- sparse graph copies
- Python conversion back to SciPy

So the failure on a 350 GB machine is expected from the code as written.


## 10M With Current Defaults

For `N = 10,000,000`:

- `kNN = 15810`
- each dense `N x (kNN + 1)` double matrix is about:
  - `10e6 * 15811 * 8 ~= 1265.0 GB`

Five such matrices imply:

- about `6.3 TB`

before counting any of the other copies.

This makes the current default `k*nn` path non-starter territory for 10M.


## Candidate Improvements

The list below is ordered from immediate operational mitigations to structural changes in the C++ core.

### A. Switch large runs from `k*nn` to `knn`

Engineering cost: trivial

Recommendation:

- Use `algorithm="knn"` for large datasets.
- For the current codebase, this should become the default once `N` crosses a threshold.

Memory impact:

- removes the `idx` / `dist` / `beta` / `lambda` / `Delta` dense-matrix explosion
- changes memory growth from approximately `O(N * sqrt(N))` neighbor state to `O(N * k)` edges

At `N = 1.7M`, `k = 10`:

- triplet vectors: about `0.41 GB`
- `locations + values`: about `0.41 GB`
- Armadillo sparse graph: about `0.29 GB`

At `N = 10M`, `k = 10`:

- triplet vectors: about `2.40 GB`
- `locations + values`: about `2.40 GB`
- Armadillo sparse graph: about `1.68 GB`

Performance impact:

- major improvement
- at 1.7M, the builder no longer asks HNSW for about 6520 neighbors per cell; it only asks for `k`
- for `k = 10`, neighbor result handling drops by roughly `652x` relative to current default `k*nn`

Risk:

- graph topology changes
- this needs biological validation, but it is the only practical path for current-scale runs

Verdict:

- required for 1.7M+
- absolutely required for 10M


### B. Lower-dimensional network input: prefer `H_merged` or `action` over `H_stacked` when possible

Engineering cost: trivial to low

Recommendation:

- expose a pipeline option that selects the representation used by `build_network()`
- default to a lower-dimensional representation at large `N`

Current code already accepts arbitrary `obsm_key` in `build_network()`. However, `run_actionet()` hardcodes `obsm_key="H_stacked"` and does not expose this parameter to the caller, so using a lower-dimensional representation requires a code change to `run_actionet()` in addition to `build_network()`. The engineering cost is still trivial, but the change is not testable end-to-end via pipeline arguments alone without that modification.

Memory impact:

memory for the dense representation and HNSW index scales linearly with `d`.

Using `d = 464` for worst-case `H_stacked` and `d = 30` for `action`:

At `N = 1.7M`:

- dense `float64 H`: `6.31 GB` -> `0.41 GB`
- dense `float32 H`: `3.16 GB` -> `0.20 GB`
- approximate HNSW level-0 storage: `3.41 GB` -> `0.46 GB`

At `N = 10M`:

- dense `float64 H`: `37.12 GB` -> `2.40 GB`
- dense `float32 H`: `18.56 GB` -> `1.20 GB`
- approximate HNSW level-0 storage: `20.04 GB` -> `2.68 GB`

Performance impact:

- roughly proportional reduction in distance-evaluation cost
- expect materially faster index build and query time

Risk:

- may change neighborhood quality versus the multi-level `H_stacked` representation

Verdict:

- very strong candidate
- likely necessary for 10M unless the graph builder is completely redesigned


### C. Keep `k` small and keep `mutual_edges_only=True`

Engineering cost: trivial

Recommendation:

- if using `knn`, start with `k = 10` or `k = 15`
- keep `mutual_edges_only=True`

Memory impact:

graph storage scales linearly with `k`.

At `N = 10M`:

- `k = 10`:
  - triplets: `2.40 GB`
  - Armadillo sparse graph: `1.68 GB`
  - ideal CSR with `float32` weights and `int32` indices: about `0.84 GB`
- `k = 50`:
  - triplets: `12.00 GB`
  - Armadillo sparse graph: `8.08 GB`
  - ideal CSR with `float32` weights and `int32` indices: about `4.04 GB`

Performance impact:

- nearly linear in `k` for result handling and downstream graph methods

Important caveat:

- `k` does not help if `algorithm="k*nn"` because that path ignores the user-specified `k`

Verdict:

- high-value lever once the pipeline is on `knn`


### D. Cap thread count for graph build

Engineering cost: trivial

Recommendation:

- do not use all cores by default for this stage
- start with something like `8` to `16` threads for large `N`

Memory impact:

HNSW allocates one visited-list array per concurrently active search thread. The visited-list pool starts with one pre-allocated list and grows dynamically on demand; peak pool size equals the actual peak concurrency.

Visited-list storage is about:

- `2 * N * threads` bytes (peak, assuming all threads are simultaneously in a graph traversal)

Examples:

- `N = 1.7M`, `88` threads -> about `0.30 GB`
- `N = 1.7M`, `8` threads -> about `0.03 GB`
- `N = 10M`, `88` threads -> about `1.76 GB`
- `N = 10M`, `8` threads -> about `0.16 GB`

`ef` and `ef_construction` do not contribute to static memory. They only control the maximum size of the per-query candidate priority queues (roughly `O(ef)` pairs of 8 bytes each per thread at peak), which is negligible compared to the visited lists. For the `k*nn` path where `ef = kNN = 5*sqrt(N)`, the per-thread candidate set can reach `~6520 * 8 = 52 KB` at N=1.7M — still negligible relative to the dense result matrices.

This does not fix the `k*nn` OOM, but it does improve stability for large `knn` runs.

Performance impact:

- could be slightly slower
- could also be neutral or faster if the current build is memory-bandwidth-bound

Verdict:

- useful secondary stabilization lever
- not a primary fix


### E. Remove wrapper-side dense copies and use `float32` end-to-end inside `build_network`

Engineering cost: low to medium

Recommendation:

- stop accepting only `py::array_t<double>` in the binding
- accept `float32`
- avoid the Python-side transpose-copy plus element-wise copy into `arma::mat`
- replace `numpy_to_arma_mat`'s element-by-element row-to-column-major loop with a vectorised transpose (e.g. a tiled `memcpy` or `mkl_domatcopy`) or eliminate the intermediate `arma::mat` entirely
- stop passing `H` by value
- feed HNSW from one `float32` buffer
- store neighbor distances as `float32` in triplet vectors (both `knn` and `k*nn` currently widen HNSW's `float` distances to `double` on insertion; this is unnecessary)

Potential implementation options:

- accept Fortran-contiguous `float32` input
- add a dedicated C++ path for row-major `float32` input
- replace the generic Armadillo conversion with a direct copy into `arma::fmat`

Memory impact:

Current stage-specific dense-copy overhead is approximately:

- transposed NumPy `float64` copy
- Armadillo `arma::mat` copy (serial element loop; also a latency bottleneck at large N)
- by-value `arma::mat` copy inside the builder
- `arma::fmat X` copy

These four copies apply to both `knn` and `k*nn`. Switching to `knn` does not eliminate them.

With `d = 464`:

At `N = 1.7M`, that is roughly:

- `6.31 + 6.31 + 6.31 + 3.16 ~= 22.09 GB`

At `N = 10M`, that is roughly:

- `37.12 + 37.12 + 37.12 + 18.56 ~= 129.92 GB`

The JSD normalisation step (Finding #6) adds up to an additional `~6.31 GB` transient copy per `clamp`/`normalise` call at N=1.7M, on top of these four structural copies. Eliminating the early double-precision copies reduces this transient overhead proportionally.

The exact savings depend on the final implementation, but removing the redundant `float64` copies should save on the order of:

- `~10 to ~20 GB` at 1.7M
- `~50 to ~110 GB` at 10M

Performance impact:

- faster stage startup
- much less memory-bandwidth spent on transposition and copy loops
- the serial element loop in `numpy_to_arma_mat` (O(N × d) individual assignments) is a latency bottleneck at scale; replacing it is a meaningful time saving independent of memory

Verdict:

- very worthwhile
- this should be done even if the algorithm stays `knn`


### F. Rewrite graph assembly to stream directly into CSR/CSC

Engineering cost: medium to high

Recommendation:

- replace the current triplets -> `locations` -> `arma::sp_mat` -> transpose -> symmetrize -> SciPy conversion pipeline
- build CSR/CSC incrementally
- symmetrize directly during or immediately after batched neighbor search

ABI note: this change requires the C++ builder to stop returning `arma::sp_mat` and to instead either write to a file path or return a CSR32 struct. The pybind binding currently returns a SciPy object materialized in memory; a streaming builder cannot do this while skipping the in-memory graph. Downstream code that reads `adata.obsp["actionet"]` (e.g. diffusion, layout) must be able to use a backed sparse reader or the graph must be re-loaded after the backed write. This is a contract change at the C++ level and needs to be coordinated with downstream consumers before implementation.

Memory impact:

This change does not alter HNSW memory, but it removes several full edge-set duplicates.

For `N = 10M`, `k = 10`, the current builder pays roughly:

- triplets: `2.40 GB`
- `locations + values`: `2.40 GB`
- Armadillo sparse graph: `1.68 GB`

and then adds more temporary duplication during transpose, symmetrization, and SciPy conversion.

A streamed CSR builder can get much closer to:

- final graph + small batch buffers

which should save several GB at 10M and materially reduce fragmentation risk.

Performance impact:

- likely better
- avoids multiple global copies and one large sparse transpose
- avoids `#pragma omp critical` insertion of big local vectors into global vectors

Verdict:

- strongest medium-term core refactor for the `knn` path
- likely necessary for robust 10M support


### G. Keep `k*nn`, but rewrite it to stream per-cell or per-batch

Engineering cost: high

Recommendation:

- if adaptive degree must be preserved, rewrite `k*nn` so it computes thresholds on one cell or one batch at a time
- never materialize global `idx`, `dist`, `beta`, `lambda`, or `Delta`

Memory impact:

- would remove the catastrophic `O(N * sqrt(N))` dense-state requirement
- peak memory would become approximately:
  - HNSW index
  - input representation
  - one query batch
  - final graph

Performance impact:

- still likely poor at very large `N`
- even without the dense matrices, the method still asks for:
  - about `11.1 billion` candidate neighbors at `N = 1.7M`
  - about `158 billion` candidate neighbors at `N = 10M`

So a memory-safe `k*nn` implementation may still be too slow to be the default at 10M.

Verdict:

- only pursue if adaptive `k*nn` behavior is biologically essential
- otherwise prefer `knn`


### H. Move the graph representation away from 64-bit Armadillo sparse matrices

Engineering cost: high

Recommendation:

- use CSR with `int32` indices and `float32` weights when `N < 2^31`
- either:
  - teach downstream C++ to operate on that representation directly, or
  - introduce a graph abstraction that can back onto CSR32

Memory impact:

At `N = 10M`, `k = 10` directed edges:

- current Armadillo-like sparse footprint: about `1.68 GB`
- `float32`/`int32` CSR: about `0.84 GB`

The larger benefit comes from shrinking all intermediate edge buffers as well.

Performance impact:

- usually positive due to lower bandwidth

Verdict:

- not the first change to make
- good long-term cleanup once the algorithmic issues are fixed


## Ranked Recommendation

### Immediate operational changes

Items 1, 2, and 4 below require no code changes — they are pipeline operator configuration choices today. Item 3 requires a one-line change to `run_actionet()` to expose `obsm_key` as a parameter.

1. For large runs, stop using `k*nn`.
2. Use `algorithm="knn"`, `k=10..15`, `mutual_edges_only=True`.
3. Prefer a lower-dimensional network representation:
   - first choice to test: `H_merged`
   - strongest memory saver: `action`
   - requires exposing `obsm_key` in `run_actionet()` (currently hardcoded to `"H_stacked"`)
4. Cap graph-build threads to something moderate, not all available cores.

### Near-term code changes

1. Add a guardrail:
   - auto-switch away from `k*nn` when `5 * sqrt(N)` exceeds a threshold, or error with a clear message
   - suggested threshold: switch when a single `idx` matrix would exceed a configurable memory limit; a practical hard-coded default would be `N > 50_000` (at which point `kNN ~= 1118` and each dense matrix is already ~400 MB), or more conservatively `N > 100_000` (kNN ~= 1581, each matrix ~10 GB at N=1M)
   - the simplest implementation is to error in Python before calling C++: `if algorithm == "k*nn" and kNN := int(5 * round(N**0.5)) > threshold: raise ValueError(...)`
2. Remove redundant dense copies and switch `build_network()` internals to `float32`, including neighbor distance storage in triplet vectors (currently widened from `float` to `double` unnecessarily).
3. Replace the element-by-element loop in `numpy_to_arma_mat` with a vectorised transpose or eliminate the `arma::mat` intermediate entirely.
4. Add a pipeline option for the network input representation instead of hard-coding `H_stacked` in `run_actionet()`.

### Core refactors for 10M

1. Rewrite the `knn` graph builder to stream directly into CSR/CSC (note: requires ABI change to stop returning `arma::sp_mat`; coordinate with downstream consumers).
2. If needed, redesign the graph container away from `arma::sp_mat`.
3. Only if scientifically required, build a memory-safe batched `k*nn`; do not keep the current global dense-matrix version.


## Practical Path To 10M

The current default implementation will not reach 10M.

The most realistic path is:

1. fixed-degree `knn`
2. low-dimensional network input (`action` or `H_merged`)
3. `float32` network build path
4. streamed CSR graph assembly
5. moderate thread count

Under that design, memory should be dominated by:

- the input representation
- the HNSW index
- the final sparse graph

Using `action`-like dimensionality (`d ~= 30`) and `k = 10`, rough 10M-scale memory becomes:

- input `float32`: about `1.2 GB`
- HNSW level-0 storage: about `2.7 GB`
- final CSR32 graph: about `0.8 GB` directed, higher after symmetrization
- plus batching overhead

Using worst-case `H_stacked` dimensionality (`d = 464`) and `k = 10`, it is still plausible if copy elimination is done:

- input `float32`: about `18.6 GB`
- HNSW level-0 storage: about `20.0 GB`
- final graph: low single-digit GB

So 10M is likely feasible only after the builder is moved to:

- `knn`
- low-copy `float32`
- streamed sparse assembly


## Implementing Disk-Backed Network Construction

This section narrows the question from "how do we reduce peak RAM?" to "how should a true backed `build_network()` be implemented in this repository?"

The short answer is:

- it should be implemented as a new backed `knn` path
- it should read the network input directly from `/obsm/<key>` in row batches
- it should write the graph directly to `/obsp/<key>` as CSR
- it should avoid materialising either the input representation or the output graph in Python memory

It should **not** start by trying to preserve the current `k*nn` implementation.


### What The Repo Already Has

The current codebase already contains most of the patterns needed for a backed network builder:

- chunked Python access to backed `X` / layers via `MatrixSource`
- direct HDF5-backed C++ reading for `reduce_kernel()` via `BackedSparseMatrixOperator`
- direct HDF5 persistence for `obsm`, `varm`, `obsp`, etc. via `_anndata_io.append_to_anndata`

So this is not a greenfield feature. The missing piece is a path that streams:

- dense reduced coordinates from `obsm`
- into HNSW / graph construction
- then back out to `obsp`

without routing through full in-memory NumPy / SciPy objects.


### Current Blockers In The Existing Implementation

#### 1. The backed reader abstraction stops at `X` / layers

`MatrixSource` is designed around `adata.X` and `adata.layers[...]`, not `adata.obsm[...]`.

That is fine for normalization, filtering, and feature-level streaming work, but `build_network()` uses:

- `obsm["H_stacked"]`
- `obsm["H_merged"]`
- or potentially `obsm["action"]`

So a backed network path needs a new accessor for backed `obsm` datasets, not just reuse of `MatrixSource`.

#### 2. `build_network()` only accepts an in-memory NumPy array

Today the binding entry point is:

```cpp
py::object build_network(py::array_t<double> H, ...)
```

That forces:

- full dense materialisation in Python
- a pybind copy into `arma::mat`
- then another copy to `arma::fmat`

So the current ABI is fundamentally in-memory.

#### 3. `persist_updates()` always assigns in-memory before persisting

For backed AnnData, `persist_updates()` first calls:

```python
apply_inmemory_updates(...)
```

and only then writes to disk.

That is fine for small `obsm` arrays and metadata, but it defeats the purpose for a multi-GB graph:

- a true backed graph builder cannot first build a giant SciPy matrix
- and then assign `adata.obsp[key] = G`

So the persistence layer needs either:

- a `persist_only` / `skip_inmemory` mode, or
- a dedicated backed graph writer that bypasses `persist_updates()`

for very large `obsp` results.

#### 4. The current H5AD writer assumes whole-matrix writes and defaults to gzip

`_anndata_io._write_matrix()` currently:

- expects the whole matrix object in memory
- converts sparse matrices to CSR
- writes `data`, `indices`, `indptr` in one shot
- writes dense matrices with gzip compression

This causes two separate problems for a backed network implementation:

1. the network output path cannot be streamed row-by-row into `obsp`
2. the network input persisted in `obsm` is likely gzip-compressed, which is a poor format for repeated row-batch reads during HNSW build/search

So the generic writer is useful as a reference, but not sufficient as the large-scale execution path.


### Recommended MVP: Backed `knn` Only

The first backed implementation should support:

- `algorithm="knn"`
- small fixed `k`
- `mutual_edges_only=True` by default
- dense backed input from `obsm`

It should explicitly reject or auto-switch away from:

- `algorithm="k*nn"`

for large backed runs.

This keeps the first implementation aligned with the only path that appears viable for 10M anyway.


### Recommended Architecture

#### A. Add a dedicated backed dense reader for `obsm`

Add a new HDF5-backed dense reader in `libactionet`, analogous to `BackedSparseMatrixOperator`, but designed for row access rather than matvecs.

Suggested shape:

```cpp
class BackedDenseMatrixReader {
public:
    arma::uword n_rows() const;
    arma::uword n_cols() const;
    void read_rows(arma::uword start, arma::uword end, std::vector<float>& out) const;
    void read_row(arma::uword i, std::vector<float>& out) const;
};
```

Important details:

- target HDF5 path should be `/obsm/<key>`
- storage layout is already natural for network construction because `obsm` is `(cells x dims)`
- reads should be row-batched with HDF5 hyperslabs
- output buffer should be `float32`
- for `jsd`, clamp/normalise each row on the fly rather than creating a full transformed copy

This is a better fit than `MatrixOperator`, because HNSW needs direct access to full point vectors, not repeated matrix-vector products.

#### B. Refactor the C++ builder around a row-access interface

The current in-memory builder assumes:

- full `arma::mat H`
- column-wise point access via `X.colptr(i)`

That is the wrong abstraction for backed `obsm`, which is naturally row-oriented.

The builder should be refactored so both in-memory and backed modes can share a common internal path, for example:

```cpp
template <class PointReader>
CSRGraph buildNetworkKNN_streaming(PointReader& reader, ...);
```

where `PointReader` supplies row batches or single rows in `float32`.

That removes the need for:

- Python transpose
- `arma::mat` copy
- by-value `arma::mat` copy
- full `arma::fmat X`

and also makes the backed and in-memory paths converge instead of diverge.

#### C. Build the HNSW index from row batches

For backed input, the C++ path should:

1. open `/obsm/<key>`
2. read `B` rows at a time
3. convert / normalise the batch to `float32`
4. call `addPoint()` for each row
5. discard the batch buffer

Because HNSW stores each point internally, the input batch does not need to stay resident after insertion.

This means backed input primarily removes:

- the full dense input copies
- not the HNSW index itself

That distinction matters. Backed construction is very useful, but it does not make HNSW memory-free.

Thread safety note: in the current hnswlib version used by this repository, parallel `addPoint` calls from multiple threads have known data races on `element_levels_` (read in `mutuallyConnectNewElement` while written by `addPoint`) and on the internal random-number generator (`level_generator_`). Both the existing in-memory builder and any backed builder that calls `addPoint` from an OpenMP parallel region inherit this issue. In practice the races are typically benign and the resulting graph structure is valid but non-reproducible across runs. If determinism is a hard requirement (e.g. for pipeline reproducibility or parity testing with the R front-end), `addPoint` must be serialized or the hnswlib version must be updated to one that provides explicit thread-safe insertion. Parallel `searchKnn` is safe — it is `const` and uses only the mutex-protected visited-list pool.

#### D. Store directed neighbor results in fixed-width scratch arrays

For fixed-degree `knn`, the cleanest large-scale implementation is:

- first pass: search neighbors for every cell
- write results into fixed-width directed arrays

Suggested scratch representation:

- `nbr_idx`: shape `(N, k)`, `int32`
- `nbr_dist`: shape `(N, k)`, `float32`
- `indptr`: shape `(N + 1)`, `int64` or implicit fixed stride

Memory for this scratch space is modest:

- `N = 1.7M`, `k = 10` -> about `0.14 GB`
- `N = 10M`, `k = 10` -> about `0.80 GB`
- `N = 10M`, `k = 15` -> about `1.20 GB`

This is much cheaper than building multiple sparse copies in memory, and it makes the second-stage mutualization logic straightforward.

These scratch arrays can live in:

- RAM
- a memory-mapped temp file
- or a temporary HDF5 group

For HPC-scale runs, a temp file on local scratch is a reasonable default.

#### E. Mutualize and symmetrize in a second pass

Once the directed neighbor arrays exist, symmetrization can happen without constructing `G`, `G.t()`, and `G_sym`.

For each row `i` and candidate neighbor `j`:

- check whether `i` is present in `j`'s directed neighbor list
- if yes, keep the edge for `mutual_edges_only=True`
- if `mutual_edges_only=False`, emit the union weight using the same averaging rule as the current implementation

Because `k` is small, checking reciprocity within a row can be done with:

- a linear scan of `j`'s `k` neighbors (O(k) per directed edge)
- or a tiny temporary hash map if needed

The full pass is O(N × k²): for each of the N × k directed edges, O(k) work to check reciprocity. For `N = 10M`, `k = 15` this is `10e6 × 225 = 2.25e9` comparisons. That sounds large but is very cache-friendly — each row's k neighbors fit in a small contiguous array — and should complete in seconds on a modern processor. It does not require parallelisation, but can be trivially parallelised per row.

This is exactly the kind of second pass that is practical once the graph is fixed-degree and the directed neighbor state is compact.

#### F. Write final CSR directly to `obsp`

The final graph should be written directly to HDF5, not round-tripped through SciPy.

Recommended output format:

- `obsp/<key>/data`
- `obsp/<key>/indices`
- `obsp/<key>/indptr`
- AnnData-compatible sparse attributes:
  - `shape`
  - `encoding-type = "csr_matrix"`
  - `encoding-version`

This implies a new streaming sparse writer, either:

- in Python under `experimental/_anndata_io.py`, or
- preferably in C++ next to the backed HDF5 reader

The writer should support:

- preallocation to an upper bound
- append-by-row or append-by-batch
- final resize to actual `nnz`
- configurable compression

For very large graphs, this path should default to:

- uncompressed or lightly compressed datasets
- chunking chosen for append performance

not unconditional gzip.


### Python-Side Dispatch Changes

The Python API can stay close to the current shape, but it needs a backed branch.

Recommended behavior in `core.build_network()`:

1. detect whether `adata` is backed
2. resolve whether `obsm_key` exists on disk at `/obsm/<key>`
3. if backed and large:
   - call `_core.build_network_backed_h5(...)`
   - persist output directly to file
   - avoid attaching the full graph in-memory by default
4. otherwise fall back to the existing in-memory path

Likely new arguments:

- `backed_chunk_size`
- `allow_compressed`
- `temp_dir`
- `attach_inmemory` or `persist_only`

`run_actionet()` should then forward `backed_chunk_size` into this stage as well, instead of only into the earlier backed-aware steps.

Downstream dependency note: if the backed path skips in-memory attachment (`attach_inmemory=False`), any pipeline step that runs after `build_network()` and reads `adata.obsp["actionet"]` will find `None` or receive a `KeyError` unless it also uses a backed sparse reader. In `run_actionet()`, the steps after `build_network()` (diffusion, layout, archetype specificity) must either be updated to accept a backed `obsp` reader, or the graph must be explicitly reloaded from disk before those steps run. This is a hard dependency that must be resolved before the backed path is used end-to-end in the pipeline, and it belongs in the Phase 2 scope at the latest.


### Compression Policy Matters More Than It Looks

The current writer stores:

- dense `obsm` arrays with gzip
- sparse `obsp` groups with gzip

For large-scale backed network construction, that is the wrong default for the network input.

If `/obsm/H_stacked` or `/obsm/action` is gzip-compressed, then HNSW build/search will repeatedly trigger decompression of row chunks. That can dominate runtime even when RAM usage is acceptable.

So the implementation should do one of the following:

1. write large network-input `obsm` keys uncompressed from the start
2. auto-decompress the target `obsm` dataset to a temporary file before graph construction
3. allow an explicit preprocessing step that rewrites the reduced representation uncompressed

This is exactly analogous to the compression warnings already added for operator-backed SVD.


### Expected Memory Impact Of A True Backed Builder

Backed network construction does **not** remove HNSW index memory, but it does remove large transient copies on both the input and output side.

#### Input-side savings

If the current path uses worst-case `H_stacked` width (`d = 464`), direct backed reading avoids approximately:

- `~22 GB` of dense input copies at `N = 1.7M`
- `~130 GB` of dense input copies at `N = 10M`

Those savings come from eliminating:

- transposed NumPy `float64`
- pybind `arma::mat`
- by-value `arma::mat`
- full `arma::fmat`

#### Output-side savings

A direct HDF5 CSR writer avoids:

- global triplet vectors
- `locations + values` duplication
- `arma::sp_mat` transpose duplication
- SciPy COO/CSR conversion buffers

At `N = 10M`, `k = 10`, this is worth several GB by itself.

#### What peak memory starts to look like

For `knn`, `k = 10`, direct HDF5 input/output, and `action`-like dimensionality (`d ~= 30`), peak memory becomes roughly:

- HNSW index: about `2.7 GB`
- directed neighbor scratch: about `0.8 GB`
- visited lists and thread-local scratch: sub-GB to low-GB depending on thread count
- plus batch buffers

That is now in the right regime for 10M.

For worst-case `H_stacked` dimensionality (`d = 464`), the HNSW index is still large:

- about `20 GB`

but that is still much better than the current path because the extra `~130 GB` of dense-copy overhead disappears.


### Expected Performance Impact

A backed implementation will add:

- one sequential read pass over the input file to build the HNSW index (row batches read from `/obsm/<key>` in order)
- one sequential read pass over the input file to query neighbors (rows read in order; HNSW internal traversal is random-access within the in-memory index, not on disk)
- one sequential write pass for the final graph

"Sequential" here refers to the file I/O access pattern for the input representation and the output CSR — not to HNSW's internal multi-layer graph traversals, which are random-access into in-memory data structures.

That does introduce more disk I/O than the in-memory path.

However:

- HNSW build/search is already the dominant compute
- row-batch HDF5 I/O is predictable and cache-friendly
- removing the huge memcpy/transposition phases is a real speedup
- avoiding gzip on the network input should keep the I/O overhead manageable

So the expected performance profile is:

- slightly slower than an ideal no-copy in-memory implementation on small data
- much faster and far more stable than the current materialize-everything path on large data


### Implementation Phasing

#### Phase 1: MVP

- backed `knn` only
- read dense `obsm` from HDF5
- build/query HNSW in batches
- write final CSR directly to `obsp`
- no `k*nn`
- no immediate in-memory `adata.obsp[key]` attachment for large graphs

#### Phase 2: Production hardening

- configurable compression policy for large `obsm` / `obsp`
- temp-file management and scratch-location controls
- parity and persistence tests for backed `build_network()`
- better user-facing warnings / auto-switch rules

#### Phase 3: Optional extensions

- backed support for `mutual_edges_only=False` with exact current semantics
- unified in-memory/backed row-reader abstraction
- optional temp graph reuse for downstream diffusion/layout
- only if scientifically required, a batched memory-safe `k*nn`


### Recommended Scope Decision

If the goal is to make 10M realistic, the recommended scope is:

1. implement backed `knn`, not backed `k*nn`
2. read network input from backed `obsm` directly
3. write network output to backed `obsp` directly
4. bypass in-memory graph attachment for large backed runs
5. make large reduced representations `float32` and uncompressed on disk

That is the minimum implementation that actually changes the scale envelope.


## Bottom Line

- The observed 1.7M OOM is expected from the current `k*nn` implementation.
- `network_density`, `k`, `ef`, and `ef_construction` are not enough to rescue the default large-scale path.
- The simplest workable fix is to stop using `k*nn` at large `N`.
- The best medium-term investment is a streamed `knn` builder with a low-copy `float32` input path.
- If 10M is the goal, the project should treat the current `k*nn` implementation as a small-data algorithm, not as the large-scale default.
