"""Training-data generation for the learned mesh->build model.

This package is OFFLINE tooling: it never runs as part of the app pipeline. It exists to
manufacture (melted mesh, block grid) training pairs by running the EASY direction —
a downloaded .schem is already the label, so we render it to a mesh and degrade that mesh
to imitate a TRELLIS reconstruction, rather than trying to find meshes that match builds.
"""
