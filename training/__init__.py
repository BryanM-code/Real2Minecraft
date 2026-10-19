"""Phase 5: train and evaluate the learned model that turns a voxelised TRELLIS mesh into blocks.

Offline tooling: the app never imports it. The training pairs are made in BuildDataCollector
(`schemset.pairs`, docs/pairs-v1.md); this package reads them, trains the model, and scores
it and the rule-based baseline (D18) the same way. See training/README.md.
"""
