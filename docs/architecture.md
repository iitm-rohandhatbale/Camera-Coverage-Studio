# Architecture (Phase 1)

## Current modules

- `camera_viewer.app`
  - Camera model and optics math
  - Geometry loading (mesh/point cloud)
  - Coverage and detectability computation
  - Open3D GUI panels and interactions

## Next recommended split

- `camera_viewer.models.camera`
- `camera_viewer.services.visibility`
- `camera_viewer.services.detectability`
- `camera_viewer.ui.panels`
- `camera_viewer.io.project_state`

This phase creates project scaffolding first and keeps runtime behavior unchanged.
