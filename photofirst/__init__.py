"""Photo-first pipeline (experimental, offline): a photo of a building -> the blocks it shows
(depth and segmentation of the photo) -> a completion model trained on the human builds ->
the whole build. Separate from the TRELLIS pipeline: no mesh generator. It reuses the
generic parts of `training` (the U-Net, tiling, scoring, window helpers). See README.md.
"""
