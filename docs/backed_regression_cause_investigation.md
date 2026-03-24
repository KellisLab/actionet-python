### Backed Regression Recovery Plan (Python Frontend + C++ ABI Boundary)

#### Summary
The regression pattern is concentrated in backed stages that persist large dense outputs (`correct_batch_effect`, `compute_network_diffusion`).  
Root-cause split:

- Python frontend: backed persistence writes dense arrays through a generic HDF5 path that does not control memory layout/chunked copy behavior, so large non-C-contiguous payloads incur costly relayout/copy during write ([`_anndata_io.py`](/Users/sebastian/Documents/git_projects/actionet-python/src/actionet/experimental/_anndata_io.py:882)).
- C++ ABI boundary: dense outputs are now emitted Fortran-order by design ([`wp_utils.cpp`](/Users/sebastian/Documents/git_projects/actionet-python/src/actionet/wp_utils.cpp:266)), which is good for in-memory memcpy but interacts poorly with backed persistence.
- Stage-specific confirmation:
  - `correct_batch_effect` persists multiple dense matrices in backed mode ([`batch_correction.py`](/Users/sebastian/Documents/git_projects/actionet-python/src/actionet/batch_correction.py:127)).
  - `compute_network_diffusion` always persists dense diffused output ([`core.py`](/Users/sebastian/Documents/git_projects/actionet-python/src/actionet/core.py:619)).
  - Operator math changes in orthogonalization are largely contract-equivalent (matmat/rmatmat swap) and are unlikely to be primary runtime drivers ([`orthogonalization.cpp`](/Users/sebastian/Documents/git_projects/actionet-python/src/libactionet/src/decomposition/orthogonalization.cpp:188)).
  - Diffusion kernel itself is unchanged algorithmically ([`network_diffusion.cpp`](/Users/sebastian/Documents/git_projects/actionet-python/src/libactionet/src/network/network_diffusion.cpp:86)).

#### Implementation Changes
1. **Immediate rollback of backed write overhead (no contract change):**
- Add a centralized backed-only dense payload normalization step before persistence: convert dense outputs to C-contiguous once (`np.ascontiguousarray`) for `obsm/varm/layers` payloads.
- Apply this in `persist_updates` path so all stages benefit, rather than patching each API function separately.

2. **Bound peak memory during backed dense writes:**
- Replace single-shot dense `create_dataset(..., data=matrix)` writes with chunked row-block writes when matrix size exceeds a threshold.
- Keep existing fast path for small dense arrays.

3. **ABI follow-up to prevent repeated layout churn:**
- Introduce dual dense return helpers in pybind (`arma -> numpy` C-order and F-order).
- Use C-order returns for high-volume stage outputs that are immediately persisted in backed workflows (`orthogonalize_*`, `run_action`, `compute_network_diffusion`, `reduce_kernel*` results).
- Keep F-order path for compute-only internals where it is beneficial.

4. **Stage-level instrumentation for regression-proofing:**
- Extend benchmark worker rows with:
  - dense payload sizes per persisted key,
  - payload contiguity flags (C/F),
  - split timing: compute vs persist.
- This makes future regressions attributable to compute or serialization immediately.

#### Public Interface / Contract Impact
- **No public API/AnnData contract changes.**
- Internal helper/interface additions only (persistence/layout controls and pybind return-path selection).

#### Test Plan
1. Functional/parity safety:
- Run existing parity/storage-parity suites (Python + cross-storage) unchanged.
- Confirm no numerical drift in `action_corrected`, `H_merged`, `archetype_footprint`.

2. Performance validation:
- Re-run backed branch-compare on at least `100k` and `200k` tiers.
- Acceptance gates:
  - `correct_batch_effect` backed wall delta returns near baseline (remove current +170% to +180% class regressions).
  - `compute_network_diffusion` backed wall/RSS regressions removed from flagged list.
  - No regression in in-memory benchmark set.

3. Memory behavior checks:
- Assert bounded peak memory during dense backed writes in synthetic stress tests (large `n_obs x k` matrices in `obsm`).

#### Assumptions and Defaults
- Assume no change to AnnData orientation contract (`cells x genes`) or algorithm defaults.
- Assume this environment keeps current backed write semantics (`append_to_anndata`) and uses compressed dataset writes.
- Prioritize backed performance even if it adds minor complexity to pybind return-path handling.
