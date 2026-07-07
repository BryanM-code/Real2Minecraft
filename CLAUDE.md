

**Project: Real2Minecraft**

This project turns a single 2D photo of a real-world object into a buildable Minecraft blueprint.

**The user flow:** a user takes or uploads a photo of something they like — a tower, a house, a car, a tree, a statue — and the app produces a voxelized, textured 3D model made entirely of real Minecraft blocks, scaled to a size the user chooses, that they can then physically construct in their Minecraft world (or export as a `.schem` file and paste it in with a tool like WorldEdit/Litematica).

**Pipeline (photo → Minecraft build):**
1. **Image → 3D mesh** — the object is first cut out of the photo (client-side background removal, shown to the user so a bad segmentation is visible), then fed through TRELLIS (an image-to-3D diffusion model) to reconstruct a full colored 3D mesh of the object, including geometry and texture on faces the camera never saw. The TRELLIS server should run **full precision** (`C:\TRELLIS\run.bat`) — it fits in 16 GB VRAM, runs as fast as fp16, and produces markedly cleaner textures.
2. **Normalize** — the raw mesh is reoriented upright and centered so it sits cleanly at the origin, ready for voxelization.
3. **Voxelize** — the mesh is scaled so its longest axis matches a user-chosen size (in blocks), then converted into a solid, gap-free voxel grid (surface rasterization + exterior flood fill, so even non-watertight meshes come out solid rather than hollow/leaky).
4. **Color → block matching** — each voxel's sampled color is matched against a real Minecraft block palette (stone, wood, wool colors, etc.), so the output isn't just gray blocks — it approximates the real object's coloring using in-game materials.
5. **Photo-guided color repair** — surfaces the photo never saw are diffusion-generated and can carry wrong colors; a color-only repair pass compares mirror-pair voxels and repaints colors that don't exist in the input photo. It never changes geometry, and is default-on via the UI "Color mode" switch — "Raw (model colors as-is)" skips all corrections except the brightness lift, for sources with no baked shadows where corrections can hurt. (There is no semantic/vision pass — earlier LLaVA/Ollama integration was removed; TRELLIS's own reconstruction is the only judgement about the object.)
6. **Output** — the user gets a viewable 3D voxel preview (side-by-side with the normalized TRELLIS mesh in linked views, so texture artifacts can be traced back to the source mesh) plus an exportable `.schem` file sized and textured for direct use as a Minecraft build blueprint.

**Design principle:** the whole pipeline treats the photo as a *reference*, not a template to trace exactly — the goal is a recognizable, physically buildable Minecraft version of the object at whatever scale the user wants, using authentic block textures rather than arbitrary colored cubes.

---

