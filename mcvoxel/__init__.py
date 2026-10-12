"""mcvoxel: mesh -> Minecraft blocks, shared by Real2Minecraft, BuildDataCollector and the
Mesh-to-Schem model. See README.md for the API, the conventions and the regression bar.

Modules
    raster          conservative surface voxelisation, exterior flood fill, `clean_solid`
    scene           multi-material GLB loader (per-face material, glass / door kinds)
    voxel_sampling  per-voxel colour from surface samples (mode-seeking, side-aware)
    surface_color   per-voxel colour as seen from outside (rays through air-facing faces)
    grid            scene -> voxel model at any size and in any frame; visible shell,
                    enclosed rooms, ground plate and site, slab / stair shapes, surface
                    repeats (slivers), colours
    block_palette   block -> sRGB palettes and weighted-CIELAB matching
    color_quantize  LAB region quantisation, optional shadow repaint, material consolidation
    block_shapes    full / slab / stairs classification and block variants
    roofs           builder-style roofs (Approach B)
    assign          voxel model -> blocks and model-ready encodings
    vocab           the factored SHAPE x MATERIAL vocabulary
    blockstates     rotation / mirror with facing remap; stair corners
    tiling          windowed inference and reassembly for large buildings
    yaw_align       rotate a mesh so its facades sit on the grid axes

Real2Minecraft's own sculpture orchestrator, photo-guided TRELLIS colour repairs, `.schem`
export and TRELLIS mesh normaliser live in the app, not here.

Install into another project's venv (editable, so edits here are picked up):
    <venv>\\Scripts\\python -m pip install -e <path to Real2Minecraft>
"""

__version__ = "0.5.0"
