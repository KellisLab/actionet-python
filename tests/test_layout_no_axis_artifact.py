"""Regression test for the canonical disconnected-vertex repair in
`layout_network`.

This guards against the axis-aligned-cells artifact that occurs at scale when
the post-pruned graph in `optimize_layout_uwot` leaves vertices with no
surviving edges. Without the repair such vertices receive zero updates from
the batch optimizer and remain frozen at their seed coordinate. With the
repair (canonical umap-learn `simplicial_set_embedding` behavior) those
vertices are translated by random per-component offsets and jittered, which
breaks any axis-aligned seed structure.
"""

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import actionet as act


def _build_disconnected_graph(n_main: int = 60, n_orphans: int = 20):
    """Construct a graph with one large connected component plus `n_orphans`
    fully isolated vertices.

    The main component is a simple ring of length `n_main` with uniform edge
    weights. Orphan vertices have no edges at all - they are the worst-case
    failure mode for the optimizer (batch mode applies zero attractive AND
    zero repulsive forces to them, so without repair they are frozen at
    their seed forever).
    """
    n_obs = n_main + n_orphans
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i in range(n_main):
        j = (i + 1) % n_main
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([1.0, 1.0])
    G = sp.csr_matrix((data, (rows, cols)), shape=(n_obs, n_obs))
    return G, n_main, n_orphans


def _axis_aligned_init(n_obs: int, n_components: int = 3) -> np.ndarray:
    """Create initial coordinates where every vertex sits on the first axis
    (X != 0, all other dims == 0). Forms a degenerate, axis-aligned seed."""
    init = np.zeros((n_obs, n_components), dtype=np.float64)
    init[:, 0] = np.linspace(-1.0, 1.0, n_obs)
    return init


def _make_adata_with_graph(G: sp.csr_matrix) -> ad.AnnData:
    n_obs = G.shape[0]
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_obs, 4)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.obsp["actionet"] = G
    return adata


def _frac_axis_aligned(coords: np.ndarray, eps: float = 1e-3) -> float:
    """Fraction of points lying within `eps` of either axis (X==0 or Y==0)."""
    on_x_axis = np.abs(coords[:, 1]) < eps
    on_y_axis = np.abs(coords[:, 0]) < eps
    return float(np.mean(on_x_axis | on_y_axis))


def test_layout_network_repair_disconnected_moves_orphans_off_axis():
    """With `repair_disconnected=True` (default), fully disconnected vertices
    must not remain on the seed axis."""
    G, n_main, n_orphans = _build_disconnected_graph(n_main=80, n_orphans=30)
    adata = _make_adata_with_graph(G)
    init = _axis_aligned_init(adata.n_obs, n_components=3)

    out = act.layout_network(
        adata,
        initial_coords=init,
        method="umap",
        n_components=2,
        n_epochs=50,
        seed=0,
        n_threads=1,
        verbose=False,
        key_added="X_repaired",
        inplace=False,
        repair_disconnected=True,
    )

    coords = out.obsm["X_repaired"]
    orphan_coords = coords[n_main:]

    # No orphan should still sit on either axis: the per-component offset and
    # jitter must have moved them off the line Y=0.
    on_axis = np.abs(orphan_coords[:, 1]) < 1e-3
    assert not on_axis.any(), (
        f"{int(on_axis.sum())}/{n_orphans} orphan vertices remained on the "
        f"X axis after repair_disconnected=True; max |Y| was "
        f"{np.abs(orphan_coords[:, 1]).max():.6f}"
    )

    # Overall axis-aligned fraction across all cells should be small.
    assert _frac_axis_aligned(coords) < 0.05


def test_layout_network_repair_disconnected_off_keeps_orphans_frozen():
    """With `repair_disconnected=False`, batch-mode UMAP cannot move
    fully disconnected vertices: they remain at their (axis-aligned) seed."""
    G, n_main, n_orphans = _build_disconnected_graph(n_main=80, n_orphans=30)
    adata = _make_adata_with_graph(G)
    init = _axis_aligned_init(adata.n_obs, n_components=3)

    out = act.layout_network(
        adata,
        initial_coords=init,
        method="umap",
        n_components=2,
        n_epochs=50,
        seed=0,
        n_threads=1,
        verbose=False,
        key_added="X_unrepaired",
        inplace=False,
        repair_disconnected=False,
    )

    coords = out.obsm["X_unrepaired"]
    orphan_coords = coords[n_main:]
    seed_orphans = init[n_main:, :2]

    # Orphans should be at their seed up to float32 precision (the layout
    # pipeline rounds initial coords through `fmat` for the optimizer).
    np.testing.assert_allclose(orphan_coords, seed_orphans, rtol=0.0, atol=1e-6)

    # And consequently, every orphan still lies on the X axis.
    assert np.all(np.abs(orphan_coords[:, 1]) < 1e-6)


def test_layout_network_repair_disconnected_jitters_single_component():
    """With a fully connected graph (single component) and a duplicate-row
    seed, `repair_disconnected=True` still applies the small per-coordinate
    jitter that breaks degenerate seeds."""
    n_obs = 40
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i in range(n_obs):
        j = (i + 1) % n_obs
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([1.0, 1.0])
    G = sp.csr_matrix((data, (rows, cols)), shape=(n_obs, n_obs))
    adata = _make_adata_with_graph(G)

    # Identical seed for every vertex.
    init = np.tile(np.array([[0.5, 0.0, 0.0]]), (n_obs, 1))

    out_repair = act.layout_network(
        adata,
        initial_coords=init.copy(),
        method="umap",
        n_components=2,
        n_epochs=50,
        seed=0,
        n_threads=1,
        verbose=False,
        key_added="X_jittered",
        inplace=False,
        repair_disconnected=True,
    )

    out_no_repair = act.layout_network(
        adata,
        initial_coords=init.copy(),
        method="umap",
        n_components=2,
        n_epochs=50,
        seed=0,
        n_threads=1,
        verbose=False,
        key_added="X_no_jitter",
        inplace=False,
        repair_disconnected=False,
    )

    # Jitter breaks the symmetry, allowing the optimizer to spread the points;
    # without jitter, all duplicate seeds get identical gradients and stay
    # collapsed (or close to it).
    coords_repair = out_repair.obsm["X_jittered"]
    coords_no_repair = out_no_repair.obsm["X_no_jitter"]

    spread_repair = float(np.linalg.norm(coords_repair - coords_repair.mean(0), axis=1).std())
    spread_no_repair = float(np.linalg.norm(coords_no_repair - coords_no_repair.mean(0), axis=1).std())

    assert spread_repair > spread_no_repair, (
        f"Expected `repair_disconnected=True` to break the duplicate-seed "
        f"symmetry, but got spread_repair={spread_repair:.6f} and "
        f"spread_no_repair={spread_no_repair:.6f}."
    )


def test_layout_network_repair_disconnected_is_seed_deterministic():
    """The repair logic uses `uwot_args.get_engine()`, which is seeded by
    the `seed` argument. Two runs with the same seed must produce identical
    embeddings."""
    G, _, _ = _build_disconnected_graph(n_main=40, n_orphans=10)
    adata = _make_adata_with_graph(G)
    init = _axis_aligned_init(adata.n_obs, n_components=3)

    out_a = act.layout_network(
        adata,
        initial_coords=init,
        method="umap",
        n_components=2,
        n_epochs=30,
        seed=42,
        n_threads=1,
        rng_type="deterministic",
        verbose=False,
        key_added="X_a",
        inplace=False,
        repair_disconnected=True,
    )
    out_b = act.layout_network(
        adata,
        initial_coords=init,
        method="umap",
        n_components=2,
        n_epochs=30,
        seed=42,
        n_threads=1,
        rng_type="deterministic",
        verbose=False,
        key_added="X_b",
        inplace=False,
        repair_disconnected=True,
    )

    np.testing.assert_allclose(out_a.obsm["X_a"], out_b.obsm["X_b"], rtol=0.0, atol=0.0)
