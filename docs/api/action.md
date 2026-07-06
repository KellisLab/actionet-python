# Action

ACTION decomposition and archetypal-analysis primitives.

`run_action` is the high-level entry point: it takes an `AnnData` with a
reduced representation and produces cell-to-archetype assignments. The
lower-level routines below let you compose or run individual stages of
archetypal analysis manually.

## Top-level

::: actionet.action
    options:
      members:
        - run_action
        - run_archetypal_analysis
        - decompose_action
        - collect_archetypes
        - merge_archetypes
        - run_simplex_regression
        - run_spa
        - run_label_propagation
        - compute_archetype_centrality
