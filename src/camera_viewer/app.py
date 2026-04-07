"""
Camera Coverage & Planning Tool
================================
Interactive 3D application for camera placement simulation and coverage analysis.
Uses Open3D GUI, raycasting, and real-world camera optics modeling.

Dependencies:
    pip install open3d numpy trimesh scipy

Usage:
    python tool.py
    python -m camera_viewer.app
    camera-viewer --mesh path/to/model.stl

Fixes applied (v3):
  - Removed all Unicode symbols (bullet, arrow, warning, cmd) that showed as ?????
    in Open3D's built-in font. Replaced with plain ASCII equivalents.
  - Panel now scrolls; all sections (OBJECT, CAMERA, ACTIONS, STATS, NAV) visible.
  - Default camera position pulled back to [3, 3, 2] so cuboid is clearly visible
    and not buried inside the frustum marker.
  - Stat value labels no longer clip: values placed on their own right-aligned row.
  - Separator uses plain dashes instead of Unicode box-drawing characters.
  - Navigation hints use plain ASCII only.
  - Window sized for MacBook M1 13" logical resolution (1240 x 778).
"""

import os
import argparse
import base64
import copy
import json
import re
import threading
import subprocess
import shutil
from datetime import datetime
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("[WARNING] trimesh not found. Using Open3D sampling fallback.")


# --------------------------------------------------------------
#  Camera Model
# --------------------------------------------------------------

class CameraModel:
    """Real-world camera optics: intrinsics, extrinsics, FOV, frustum."""

    def __init__(self):
        # Pulled back so the default cuboid is clearly visible in the frustum
        self.position     = np.array([3000.0, 3000.0, 2000.0], dtype=float)
        self.lookat       = np.array([0.0, 0.0, 0.0], dtype=float)
        self.focal_length = 35.0    # mm
        self.sensor_width = 36.0    # mm  (full-frame 35mm)
        self.image_width  = 1920    # px
        self.image_height = 1080    # px
        self.near         = 300.0   # mm
        self.far          = 12000.0 # mm
        self.up           = np.array([0.0, 0.0, 1.0], dtype=float)

    # -- Derived optics -----------------------------------------

    @property
    def sensor_height(self) -> float:
        return self.sensor_width * (self.image_height / self.image_width)

    @property
    def fov_h(self) -> float:
        return 2.0 * np.arctan(self.sensor_width / (2.0 * self.focal_length))

    @property
    def fov_v(self) -> float:
        return 2.0 * np.arctan(self.sensor_height / (2.0 * self.focal_length))

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        fx = self.focal_length * self.image_width  / self.sensor_width
        fy = self.focal_length * self.image_height / self.sensor_height
        cx, cy = self.image_width / 2.0, self.image_height / 2.0
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

    # -- Camera axes --------------------------------------------

    def _axes(self):
        fwd = self.lookat - self.position
        n = np.linalg.norm(fwd)
        fwd = fwd / n if n > 1e-9 else np.array([0.0, 0.0, -1.0])
        up_ref = self.up.copy()
        if abs(np.dot(fwd, up_ref)) > 0.99:
            up_ref = np.array([0., 1., 0.]) if abs(fwd[1]) < 0.99 else np.array([1., 0., 0.])
        right = np.cross(fwd, up_ref);  right /= np.linalg.norm(right)
        up    = np.cross(right, fwd);   up    /= np.linalg.norm(up)
        return fwd, right, up

    def extrinsic_matrix(self) -> np.ndarray:
        fwd, right, up = self._axes()
        R = np.array([right, -up, fwd])
        t = -R @ self.position
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
        return M

    # -- Frustum corners ----------------------------------------

    def frustum_corners(self):
        fwd, right, up = self._axes()
        hw_n = np.tan(self.fov_h / 2) * self.near
        hh_n = np.tan(self.fov_v / 2) * self.near
        hw_f = np.tan(self.fov_h / 2) * self.far
        hh_f = np.tan(self.fov_v / 2) * self.far
        o = self.position
        near = [o + self.near * fwd + s * hw_n * right + t * hh_n * up
                for s, t in [(-1,-1),(1,-1),(1,1),(-1,1)]]
        far  = [o + self.far  * fwd + s * hw_f * right + t * hh_f * up
                for s, t in [(-1,-1),(1,-1),(1,1),(-1,1)]]
        return near, far

    # -- Frustum planes (6 inward normals) ---------------------

    def frustum_planes(self):
        fwd, right, up = self._axes()
        o = self.position
        planes = []
        planes.append(( fwd, -np.dot(fwd,  o + self.near * fwd)))
        planes.append((-fwd,  np.dot(fwd,  o + self.far  * fwd)))
        for sign in (+1, -1):
            n = np.cos(self.fov_h / 2) * fwd + sign * np.sin(self.fov_h / 2) * right
            n /= np.linalg.norm(n)
            planes.append((n, -np.dot(n, o)))
        for sign in (+1, -1):
            n = np.cos(self.fov_v / 2) * fwd + sign * np.sin(self.fov_v / 2) * up
            n /= np.linalg.norm(n)
            planes.append((n, -np.dot(n, o)))
        return planes

    def points_in_frustum(self, pts: np.ndarray) -> np.ndarray:
        inside = np.ones(len(pts), dtype=bool)
        for normal, d in self.frustum_planes():
            inside &= (pts @ normal + d) >= 0
        return inside

    def points_in_image(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return np.zeros(0, dtype=bool)
        ext = self.extrinsic_matrix()
        pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=float)], axis=1)
        cam_pts = (ext @ pts_h.T).T[:, :3]
        z = cam_pts[:, 2]
        in_depth = (z >= self.near) & (z <= self.far)
        safe_z = np.where(np.abs(z) < 1e-9, 1e-9, z)
        K = self.intrinsic_matrix
        u = K[0, 0] * (cam_pts[:, 0] / safe_z) + K[0, 2]
        v = K[1, 1] * (cam_pts[:, 1] / safe_z) + K[1, 2]
        in_image = (
            (u >= 0.0) & (u < self.image_width) &
            (v >= 0.0) & (v < self.image_height)
        )
        return in_depth & in_image

    def resolution_mm_per_pixel(self, distance: float) -> float:
        return (self.sensor_width / self.image_width) * (distance / self.focal_length)


# --------------------------------------------------------------
#  Geometry helpers
# --------------------------------------------------------------

def build_frustum_lineset(camera: CameraModel,
                           color=(0.2, 0.8, 1.0)) -> o3d.geometry.LineSet:
    near, far = camera.frustum_corners()
    pts = near + far + [camera.position, camera.lookat]
    lines = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7],
        [8, 9],  # camera direction
    ]
    # also add near/far diagonals for better projection readout
    lines += [[0,6], [1,7], [2,4], [3,5]]
    ls = o3d.geometry.LineSet()
    ls.points  = o3d.utility.Vector3dVector(np.array(pts))
    ls.lines   = o3d.utility.Vector2iVector(lines)
    colors = [color] * 12 + [[0.8, 0.2, 0.2]] * 1 + [[0.4, 0.4, 0.4]] * 4
    ls.colors  = o3d.utility.Vector3dVector(colors)
    return ls


def build_camera_marker(position: np.ndarray, radius=50.0,
                         color=(1.0, 0.8, 0.0)) -> o3d.geometry.TriangleMesh:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
    s.translate(position)
    s.paint_uniform_color(color)
    s.compute_vertex_normals()
    return s


def create_cuboid_mesh(lx: float, ly: float, lz: float) -> o3d.geometry.TriangleMesh:
    box = o3d.geometry.TriangleMesh.create_box(width=lx, height=ly, depth=lz)
    box.translate([-lx/2, -ly/2, -lz/2])
    box.compute_vertex_normals()
    box.paint_uniform_color([0.55, 0.60, 0.65])
    return box


def sample_surface_points(mesh: o3d.geometry.TriangleMesh,
                           n_points: int = 20000) -> np.ndarray:
    if TRIMESH_AVAILABLE:
        tm = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                             faces=np.asarray(mesh.triangles), process=False)
        pts, _ = trimesh.sample.sample_surface(tm, n_points)
        return pts.astype(np.float32)
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    return np.asarray(pcd.points, dtype=np.float32)


def build_coverage_pointcloud(points: np.ndarray,
                               visible: np.ndarray,
                               visible_color=(0.1, 0.9, 0.2),
                               hidden_color=(0.9, 0.1, 0.1)) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.where(visible[:, None], visible_color, hidden_color)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def parse_colmap_points3d_txt(path: str) -> o3d.geometry.PointCloud:
    """Parse COLMAP points3D.txt and return an Open3D PointCloud."""
    pts = []
    cols = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            # Expected minimum: POINT3D_ID X Y Z R G B ERROR
            if len(parts) < 8:
                continue
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            except ValueError:
                continue
            pts.append([x, y, z])
            cols.append([r / 255.0, g / 255.0, b / 255.0])

    if len(pts) == 0:
        raise ValueError("No valid points found in COLMAP points3D.txt")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(cols, dtype=np.float64))
    return pcd


def sample_pointcloud_points(points: np.ndarray, n_points: int) -> np.ndarray:
    """Randomly sample points from an existing point cloud."""
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if len(points) <= n_points:
        return points.astype(np.float32)
    idx = np.random.choice(len(points), n_points, replace=False)
    return points[idx].astype(np.float32)


# --------------------------------------------------------------
#  Visibility / raycasting
# --------------------------------------------------------------

class VisibilityEngine:
    def __init__(self, mesh: o3d.geometry.TriangleMesh):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def compute_visibility(self, camera: CameraModel,
                           surface_points: np.ndarray,
                           occlusion_tol: float = 1.0) -> np.ndarray:
        visible = np.zeros(len(surface_points), dtype=bool)
        in_image = camera.points_in_image(surface_points)
        idx = np.where(in_image)[0]
        if len(idx) == 0:
            return visible
        pts   = surface_points[idx].astype(np.float32)
        cam   = camera.position.astype(np.float32)
        dirs  = pts - cam
        dists = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs  = dirs / np.maximum(dists, 1e-9)
        rays  = np.concatenate([np.tile(cam, (len(pts), 1)), dirs], axis=1)
        result   = self.scene.cast_rays(
            o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
        hit_dist = result['t_hit'].numpy()
        unoccluded = hit_dist >= (dists[:, 0] - occlusion_tol)
        visible[idx[unoccluded]] = True
        return visible


# --------------------------------------------------------------
#  Statistics
# --------------------------------------------------------------

def compute_statistics(camera: CameraModel, pts: np.ndarray,
                        visible: np.ndarray) -> dict:
    total = len(pts)
    n_vis = int(visible.sum())
    pct   = 100.0 * n_vis / total if total > 0 else 0.0
    if n_vis > 0:
        d     = np.linalg.norm(pts[visible] - camera.position, axis=1)
        avg_d = float(d.mean())
        mpp   = camera.resolution_mm_per_pixel(avg_d)
    else:
        avg_d = mpp = 0.0
    return dict(total=total, visible=n_vis, coverage=pct,
                avg_dist=avg_d, mm_per_px=mpp,
                fov_h_deg=float(np.degrees(camera.fov_h)),
                fov_v_deg=float(np.degrees(camera.fov_v)))


# --------------------------------------------------------------
#  Main application
# --------------------------------------------------------------

class CameraPlanner:
    """Single-window Open3D GUI: left panel controls, right SceneWidget."""

    _MESH     = "mesh"
    _FRUSTUM  = "frustum"
    _CAMDIR   = "cam_dir"
    _MARKER   = "cam_marker"
    _COVERAGE = "coverage"
    _CAMCOVERAGE = "coverage_cam"
    _GRID     = "grid"
    _AXES     = "axes"
    _POINTS   = "points"

    def __init__(self, initial_mesh_path: str = None):
        self.cameras      = [CameraModel()]
        self.camera_colors = [np.array([0.10, 0.75, 0.95], dtype=float)]
        self.camera_marker_radius = 50.0
        self.active_camera_idx = 0
        self.selected_type = "camera"
        self._dragging_camera = False
        self._dragging_object = False
        self._drag_start_xy = np.array([0.0, 0.0], dtype=float)
        self._drag_orig_pos = np.array([0.0, 0.0, 0.0], dtype=float)
        self._drag_orig_obj_trans = np.array([0.0, 0.0, 0.0], dtype=float)
        self.base_mesh = None
        self.base_points = None
        self.base_point_colors = None
        self.point_cloud = None
        self.mesh_transform = np.eye(4)
        self.object_translation = np.array([0.0, 0.0, 0.0], dtype=float)
        self.object_rotation = np.array([0.0, 0.0, 0.0], dtype=float)
        self.object_scale = 1.0
        self.mesh        = None
        self.surface_pts = None
        self.vis_engine  = None
        self._last_visible_all = None
        self._last_visible_cam = None
        self._last_camera_coverages = []
        self._camera_coord_labels = []
        self._last_stats = None
        self._defect_size_mm = 1.0
        self._defect_threshold_red_px = 3.0
        self._defect_threshold_yellow_px = 8.0
        self._last_surface_distances = None
        self._mesh_source = {"type": "cuboid", "dims": [1.0, 0.5, 0.5]}
        self._workspace_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
        self._projects_root = os.path.join(self._workspace_root, "projects")
        os.makedirs(self._projects_root, exist_ok=True)
        self._colmap_workflow_root = os.path.join(self._workspace_root, "colmap_workflow")
        self._colmap_script_path = os.path.join(
            self._colmap_workflow_root, "05 SCRIPTS", "batch_reconstruct.bat")
        self._colmap_scenes_dir = os.path.join(self._colmap_workflow_root, "04 SCENES")
        self._autosave_path = os.path.join(
            self._workspace_root,
            "camera_viewer_last_session.json")
        self._suspend_autosave = False
        self._current_project_name = None
        self._current_project_path = None
        self._pending_new_project = None
        self._recon_running = False
        self._recon_process = None
        self._recon_log_lines = []

        self.app = gui.Application.instance
        self.app.initialize()

        # Pass 1 UI tokens: consistent colors and visual hierarchy.
        self._ui = {
            "panel_bg": gui.Color(0.13, 0.14, 0.16),
            "section": gui.Color(0.57, 0.80, 0.98),
            "key": gui.Color(0.76, 0.78, 0.82),
            "value": gui.Color(0.98, 0.94, 0.64),
            "muted": gui.Color(0.55, 0.57, 0.62),
            "subtle": gui.Color(0.45, 0.47, 0.52),
            "accent": gui.Color(0.40, 0.84, 0.56),
        }

        # Use the default font - importantly, do NOT use any Unicode glyphs
        # anywhere in the UI because Open3D's bundled font does not cover them
        # and they render as ????? on macOS/Windows.
        # self.app.add_font(
        #     gui.FontDescription(gui.FontDescription.DEFAULT_FONT_FILENAME))
        self.em = 12   # compact spacing for MacBook M1 13"

        # MacBook M1 13": logical 1280x800.
        self.win = self.app.create_window(
            "Camera Coverage & Planning Tool", 1600, 920)
        self.win.set_on_layout(self._on_layout)
        self.win.set_on_close(self._on_close)

        # -- 3D scene -----------------------------------------
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.win.renderer)
        self.scene_widget.scene.set_background([0.94, 0.95, 0.97, 1.0])
        self.scene_widget.enable_scene_caching(False)
        self.scene_widget.set_view_controls(
            gui.SceneWidget.Controls.ROTATE_CAMERA)
        
        self.scene_widget.set_on_mouse(self._on_mouse)

        # -- Left scrollable panel ----------------------------
        m = int(0.7 * self.em)
        self.panel = gui.ScrollableVert(
            int(0.45 * self.em), gui.Margins(m, m, m, m))
        try:
            self.panel.background_color = self._ui["panel_bg"]
        except Exception:
            pass

        self._build_project_section()
        self.panel.add_child(self._hsep())
        self._build_mesh_section()
        self.panel.add_child(self._hsep())
        self._build_object_section()
        self.panel.add_child(self._hsep())
        self._build_camera_section()
        self.panel.add_child(self._hsep())
        self._build_action_section()
        self.panel.add_child(self._hsep())
        self._build_stats_section()
        self.panel.add_child(self._hsep())
        self._build_nav_section()

        self.win.add_child(self.panel)
        self.win.add_child(self.scene_widget)

        self._add_reference_grid(size=8000.0, n=16)
        self._add_reference_axes(size=300.0)

        if initial_mesh_path:
            self._load_mesh_from_path(initial_mesh_path)
        elif os.path.exists(self._autosave_path):
            self._load_project_from_path(self._autosave_path, show_status=False)
        else:
            self._create_default_cuboid()

        if self.mesh is None:
            self._create_default_cuboid()

        self._update_camera_visuals()
        self._update_camera_panel()

    # -- Layout -----------------------------------------------

    def _on_layout(self, _ctx):
        r  = self.win.content_rect
        pw = max(320, min(420, int(r.width * 0.28)))
        self.panel.frame        = gui.Rect(r.x, r.y, pw, r.height)
        self.scene_widget.frame = gui.Rect(r.x + pw, r.y, r.width - pw, r.height)

    def _on_close(self):
        self._autosave_now(show_status=False)
        self.app.quit()
        return True

    # -- UI primitives ----------------------------------------

    def _hsep(self):
        """Plain-ASCII separator line - no Unicode box-drawing characters."""
        lbl = gui.Label("----------------------------------------")
        lbl.text_color = self._ui["subtle"]
        return lbl

    def _section_lbl(self, text: str) -> gui.Label:
        """Section header - plain ASCII brackets, no Unicode bullets."""
        lbl = gui.Label(text)
        lbl.text_color = self._ui["section"]
        return lbl

    def _key_lbl(self, text: str) -> gui.Label:
        lbl = gui.Label(text)
        lbl.text_color = self._ui["key"]
        return lbl

    def _val_lbl(self, text: str = "--") -> gui.Label:
        lbl = gui.Label(text)
        lbl.text_color = self._ui["value"]
        return lbl

    def _float_ne(self, value: float, w: int = 80) -> gui.NumberEdit:
        ne = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        ne.double_value = value
        ne.decimal_precision = 3
        ne.set_preferred_width(w)
        return ne

    def _int_ne(self, value: int, w: int = 70) -> gui.NumberEdit:
        ne = gui.NumberEdit(gui.NumberEdit.INT)
        ne.int_value = value
        ne.set_preferred_width(w)
        return ne

    def _kv_row(self, key: str, val_widget) -> gui.Horiz:
        """Key label left, widget right on a single row."""
        row = gui.Horiz(4)
        row.add_child(self._key_lbl(key))
        row.add_stretch()
        row.add_child(val_widget)
        return row

    def _xyz_block(self, label: str, vals) -> tuple:
        """
        Vertical block: label row then [X ne][Y ne][Z ne].
        Each NumberEdit 62 px wide - fits 3 across in 275 px panel.
        Returns (container, (ne_x, ne_y, ne_z)).
        """
        vert = gui.Vert(2)
        vert.add_child(self._key_lbl(label))
        row = gui.Horiz(3)
        widgets = []
        for axis, val in zip(("X", "Y", "Z"), vals):
            sub = gui.Horiz(2)
            al = gui.Label(axis)
            al.text_color = self._ui["accent"]
            ne = gui.NumberEdit(gui.NumberEdit.DOUBLE)
            ne.double_value = float(val)
            ne.decimal_precision = 3
            ne.set_preferred_width(62)
            sub.add_child(al)
            sub.add_child(ne)
            row.add_child(sub)
            widgets.append(ne)
        vert.add_child(row)
        return vert, tuple(widgets)
    
    def _cam(self):
        return self.cameras[self.active_camera_idx]

    def _ensure_camera_color_count(self):
        while len(self.camera_colors) < len(self.cameras):
            self.camera_colors.append(np.array([0.80, 0.55, 0.15], dtype=float))

    def _camera_color(self, idx: int) -> np.ndarray:
        self._ensure_camera_color_count()
        c = np.asarray(self.camera_colors[idx], dtype=float)
        return np.clip(c, 0.0, 1.0)

    def _color_to_text(self, c: np.ndarray) -> str:
        r = int(np.clip(c[0], 0.0, 1.0) * 255)
        g = int(np.clip(c[1], 0.0, 1.0) * 255)
        b = int(np.clip(c[2], 0.0, 1.0) * 255)
        return f"RGB({r},{g},{b})"

    def _camera_color_presets(self):
        return [
            ("Red", np.array([0.95, 0.20, 0.20], dtype=float)),
            ("Green", np.array([0.20, 0.80, 0.30], dtype=float)),
            ("Blue", np.array([0.20, 0.40, 0.95], dtype=float)),
            ("Cyan", np.array([0.10, 0.75, 0.95], dtype=float)),
            ("Orange", np.array([0.95, 0.55, 0.15], dtype=float)),
            ("Yellow", np.array([0.95, 0.85, 0.20], dtype=float)),
            ("Magenta", np.array([0.85, 0.30, 0.85], dtype=float)),
            ("White", np.array([0.90, 0.90, 0.90], dtype=float)),
        ]

    def _set_dialog_color(self, color: np.ndarray, *args):
        if not hasattr(self, "_dlg_color_picker"):
            return
        self._dlg_color_picker.color_value = gui.Color(
            float(color[0]), float(color[1]), float(color[2]))

    def _dialog_color_value(self) -> np.ndarray:
        if not hasattr(self, "_dlg_color_picker"):
            return np.array([0.8, 0.55, 0.15], dtype=float)
        c = self._dlg_color_picker.color_value
        return np.array([
            np.clip(float(c.red), 0.0, 1.0),
            np.clip(float(c.green), 0.0, 1.0),
            np.clip(float(c.blue), 0.0, 1.0),
        ], dtype=float)
    
    def _on_mouse(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            if event.buttons == gui.MouseButton.LEFT:
                world_xy = self._screen_to_xy(event.x, event.y)
                cam_idx = self._camera_at_xy(world_xy)
                if cam_idx is not None:
                    self._select_camera(cam_idx)
                    self._dragging_camera = True
                    self._drag_start_xy = world_xy
                    self._drag_orig_pos = self.cameras[cam_idx].position.copy()
                    return gui.Widget.EventCallbackResult.HANDLED

                if self.mesh is not None:
                    self._select_object()
                    return gui.Widget.EventCallbackResult.IGNORED

        if event.type == gui.MouseEvent.Type.MOVE:
            if event.is_button_down(gui.MouseButton.LEFT):
                if self._dragging_camera:
                    world_xy = self._screen_to_xy(event.x, event.y)
                    delta = world_xy - self._drag_start_xy
                    cam = self._cam()
                    cam.position = np.array([
                        self._drag_orig_pos[0] + delta[0],
                        self._drag_orig_pos[1] + delta[1],
                        self._drag_orig_pos[2]
                    ])
                    self._update_camera_visuals()
                    self._update_camera_panel()
                    self._set_status("Dragging camera...")
                    return gui.Widget.EventCallbackResult.HANDLED

        if event.type == gui.MouseEvent.Type.BUTTON_UP:
            if event.buttons == gui.MouseButton.LEFT:
                self._dragging_camera = False
                self._set_status("Ready.")

        return gui.Widget.EventCallbackResult.IGNORED

    def _screen_to_xy(self, x: float, y: float) -> np.ndarray:
        if self.scene_widget.frame.width == 0 or self.scene_widget.frame.height == 0:
            return np.array([0.0, 0.0], dtype=float)
        return np.array([
            (x / self.scene_widget.frame.width - 0.5) * 4000.0,
            (y / self.scene_widget.frame.height - 0.5) * 4000.0,
        ], dtype=float)

    def _camera_at_xy(self, xy: np.ndarray) -> int | None:
        best_idx = None
        best_dist = 350.0
        for i, cam in enumerate(self.cameras):
            dist = np.linalg.norm(cam.position[:2] - xy)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def _set_selection(self, selection_type: str, camera_idx: int = None):
        self.selected_type = selection_type
        if selection_type == "camera" and camera_idx is not None:
            self.active_camera_idx = camera_idx
        self._refresh_selection_label()

    def _select_camera(self, idx: int):
        self._set_selection("camera", idx)
        self._update_camera_panel()
        self._set_status(f"Camera {idx + 1} selected.")

    def _select_object(self):
        self._set_selection("object")
        self._update_object_panel()
        self._set_status("Object selected.")

    def _refresh_selection_label(self):
        if self.selected_type == "camera":
            c = self._camera_color(self.active_camera_idx)
            self._selection_label.text = f"Selected: Camera {self.active_camera_idx + 1}"
            self._selection_label.text_color = gui.Color(0.20, 0.20, 0.20)
            self._selection_dot.text = "###"
            self._selection_dot.text_color = gui.Color(float(c[0]), float(c[1]), float(c[2]))
        else:
            self._selection_label.text = "Selected: Scene object"
            self._selection_label.text_color = gui.Color(0.86, 0.86, 0.60)
            self._selection_dot.text = "---"
            self._selection_dot.text_color = gui.Color(0.60, 0.60, 0.60)
        self._refresh_camera_legend()

    def _build_project_section(self):
        self.panel.add_child(self._section_lbl("[ PROJECT ]"))
        self.panel.add_child(self._key_lbl("Manage session/project state."))

        btn_row = gui.Horiz(4)
        btn_new_proj = gui.Button("New")
        btn_new_proj.set_on_clicked(self._on_new_project)
        btn_save_proj = gui.Button("Save")
        btn_save_proj.set_on_clicked(self._on_save_project)
        btn_load_proj = gui.Button("Load")
        btn_load_proj.set_on_clicked(self._on_load_project)
        btn_row.add_child(btn_new_proj)
        btn_row.add_child(btn_save_proj)
        btn_row.add_child(btn_load_proj)
        self.panel.add_child(btn_row)

    def _build_object_section(self):
        self.panel.add_child(self._section_lbl("[ OBJECT TRANSFORM ]"))

        pos_block, (self._ox, self._oy, self._oz) = self._xyz_block(
            "Position (mm)", self.object_translation)
        self.panel.add_child(pos_block)

        rot_block, (self._orx, self._ory, self._orz) = self._xyz_block(
            "Rotation (deg)", self.object_rotation)
        self.panel.add_child(rot_block)

        step_row = gui.Horiz(4)
        step_row.add_child(self._key_lbl("Rotate step (deg)"))
        step_row.add_stretch()
        self._rot_step = self._float_ne(15.0, 64)
        step_row.add_child(self._rot_step)
        self.panel.add_child(step_row)

        # Quick rotate buttons for common object orientation tweaks.
        for axis in ("X", "Y", "Z"):
            row = gui.Horiz(4)
            row.add_child(self._key_lbl(f"Rotate {axis}"))
            row.add_stretch()
            btn_minus = gui.Button(f"{axis}-")
            btn_plus = gui.Button(f"{axis}+")
            btn_minus.set_on_clicked(lambda a=axis: self._on_quick_rotate(a, -1.0))
            btn_plus.set_on_clicked(lambda a=axis: self._on_quick_rotate(a, +1.0))
            row.add_child(btn_minus)
            row.add_child(btn_plus)
            self.panel.add_child(row)

        self._oscale = self._float_ne(self.object_scale, 78)
        self.panel.add_child(self._kv_row("Scale", self._oscale))

        # Quick scale buttons for common adjustments
        scale_btn_row = gui.Horiz(4)
        scale_btn_row.add_child(self._key_lbl("Quick scale:"))
        scale_btn_row.add_stretch()
        for factor_label, factor in [("0.5x", 0.5), ("1x", 1.0), ("2x", 2.0)]:
            btn = gui.Button(factor_label)
            btn.set_on_clicked(lambda f=factor: self._on_quick_scale(f))
            scale_btn_row.add_child(btn)
        self.panel.add_child(scale_btn_row)

        btn_obj = gui.Button("Update Object")
        btn_obj.set_on_clicked(self._on_update_object)
        self.panel.add_child(btn_obj)

    def _update_camera_panel(self):
        cam = self._cam()
        self._cx.double_value = float(cam.position[0])
        self._cy.double_value = float(cam.position[1])
        self._cz.double_value = float(cam.position[2])
        self._lx.double_value = float(cam.lookat[0])
        self._ly.double_value = float(cam.lookat[1])
        self._lz.double_value = float(cam.lookat[2])
        self._fl.double_value = float(cam.focal_length)
        self._sw.double_value = float(cam.sensor_width)
        self._rw.int_value = int(cam.image_width)
        self._rh.int_value = int(cam.image_height)
        self._near.double_value = float(cam.near)
        self._far.double_value = float(cam.far)
        self._marker_radius.double_value = float(self.camera_marker_radius)
        self._refresh_fov_labels()
        self._refresh_selection_label()
        self._update_camera_info_display()

    def _update_object_panel(self):
        self._ox.double_value = float(self.object_translation[0])
        self._oy.double_value = float(self.object_translation[1])
        self._oz.double_value = float(self.object_translation[2])
        self._orx.double_value = float(self.object_rotation[0])
        self._ory.double_value = float(self.object_rotation[1])
        self._orz.double_value = float(self.object_rotation[2])
        self._oscale.double_value = float(self.object_scale)
        self._refresh_selection_label()

    def _read_object_ui(self):
        self.object_translation = np.array([
            self._ox.double_value,
            self._oy.double_value,
            self._oz.double_value,
        ], dtype=float)
        self.object_rotation = np.array([
            self._orx.double_value,
            self._ory.double_value,
            self._orz.double_value,
        ], dtype=float)
        self.object_scale = max(0.01, self._oscale.double_value)

    def _compute_object_transform(self) -> np.ndarray:
        rx, ry, rz = np.radians(self.object_rotation)
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
        R = Rz @ Ry @ Rx
        R *= self.object_scale
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = self.object_translation
        return M

    def _apply_mesh_transform(self):
        if self.base_mesh is None:
            return
        mesh = copy.deepcopy(self.base_mesh)
        mesh.transform(self._compute_object_transform())
        mesh.compute_vertex_normals()
        self.mesh = mesh
        self.surface_pts = None
        self.vis_engine = None
        self._remove(self._MESH)
        self.scene_widget.scene.add_geometry(
            self._MESH, mesh, self._mat_lit([0.55, 0.60, 0.65]))

    def _show_camera_dialog(self):
        cam = self._cam()
        cam_color = self._camera_color(self.active_camera_idx)
        dlg = gui.Dialog("Add Camera")
        layout = gui.Vert(4)

        # -- Sensor presets section (from Basler ACE2 cameras) --
        layout.add_child(self._section_lbl("[ SENSOR PRESETS ]"))
        
        sensor_presets = [
            ("Full-frame 35mm (36x24mm)", 36.0, "3:2"),
            ("APS-C (23.5x15.6mm)", 23.5, "3:2"),
            ("1\" (25.4x19.1mm)", 25.4, "3:2"),
            ("1.1\" (27.9x20.9mm)", 27.9, "3:2"),
            ("1.2\" (30.5x22.9mm)", 30.5, "3:2"),
            ("2/3\" (8.8x6.6mm)", 8.8, "4:3"),
            ("1/1.1\" (13.2x8.8mm) [Basler]", 13.2, "3:2"),
            ("1/1.8\" (7.6x5.7mm) [Basler]", 7.6, "3:2"),
            ("1/2.3\" (5.76x4.29mm) [Basler]", 5.76, "3:2"),
            ("1/2.8\" (4.8x3.6mm) [Basler]", 4.8, "4:3"),
        ]
        
        sensor_row = gui.Horiz(4)
        sensor_row.add_child(self._key_lbl("Select sensor:"))
        sensor_row.add_stretch()
        self._dlg_sensor_combo = gui.Combobox()
        for name, _, _ in sensor_presets:
            self._dlg_sensor_combo.add_item(name)
        self._dlg_sensor_combo.set_on_selection_changed(
            lambda a, b: self._on_sensor_preset_changed(a, b, sensor_presets))
        self._dlg_sensor_combo.selected_index = 6  # Default to 1/1.1" Basler
        sensor_row.add_child(self._dlg_sensor_combo)
        layout.add_child(sensor_row)

        # -- Resolution presets section (Basler ACE2 cameras + standards) --
        layout.add_child(self._section_lbl("[ RESOLUTION PRESETS ]"))
        
        resolution_presets = [
            ("2.3 MP - Basler ACE2 (1920x1200)", 1920, 1200),
            ("3.2 MP - Basler ACE2 (2048x1536)", 2048, 1536),
            ("5 MP - Basler ACE2 (2448x2048)", 2448, 2048),
            ("8 MP - Basler ACE2 (3840x2160)", 3840, 2160),
            ("8.3 MP - Basler ACE2 (3264x2448)", 3264, 2448),
            ("12 MP - Standard (4000x3000)", 4000, 3000),
            ("12.3 MP - Basler ACE2 (4096x3072)", 4096, 3072),
            ("16 MP - Standard (4688x3516)", 4688, 3516),
            ("16.1 MP - Basler ACE2 (5320x3020)", 5320, 3020),
            ("18.4 MP - Basler ACE2 (4880x3760)", 4880, 3760),
            ("20 MP - Standard (5184x3888)", 5184, 3888),
            ("20.2 MP - Basler ACE2 (4504x4480)", 4504, 4480),
            ("24 MP - Standard (6000x4000)", 6000, 4000),
            ("24.4 MP - Basler ACE2 (5328x4584)", 5328, 4584),
            ("25 MP - Basler ACE2 (5060x4948)", 5060, 4948),
            ("1080p HD (1920x1080)", 1920, 1080),
            ("2K DCI (2560x1440)", 2560, 1440),
            ("4K UHD (3840x2160)", 3840, 2160),
        ]
        
        res_row = gui.Horiz(4)
        res_row.add_child(self._key_lbl("Select resolution:"))
        res_row.add_stretch()
        self._dlg_res_combo = gui.Combobox()
        for name, _, _ in resolution_presets:
            self._dlg_res_combo.add_item(name)
        self._dlg_res_combo.set_on_selection_changed(
            lambda a, b: self._on_resolution_preset_changed(a, b, resolution_presets))
        self._dlg_res_combo.selected_index = 7  # Default to 20MP
        res_row.add_child(self._dlg_res_combo)
        layout.add_child(res_row)
        
        layout.add_child(self._key_lbl("Select from real Basler ACE2 cameras or standard formats."))

        pos_block, (self._dlg_cx, self._dlg_cy, self._dlg_cz) = self._xyz_block(
            "Position (mm)", cam.position)
        layout.add_child(pos_block)

        look_block, (self._dlg_lx, self._dlg_ly, self._dlg_lz) = self._xyz_block(
            "Look-At  (mm)", cam.lookat)
        layout.add_child(look_block)

        self._dlg_fl = self._float_ne(cam.focal_length, 78)
        layout.add_child(self._kv_row("Focal length (mm)", self._dlg_fl))

        self._dlg_sw = self._float_ne(cam.sensor_width, 78)
        layout.add_child(self._kv_row("Sensor width  (mm)", self._dlg_sw))

        self._dlg_rw = self._int_ne(cam.image_width,  150)
        self._dlg_rh = self._int_ne(cam.image_height, 150)
        res_row = gui.Horiz(4)
        res_row.add_child(self._key_lbl("Resolution (px)"))
        res_row.add_stretch()
        res_row.add_child(self._dlg_rw)
        res_row.add_child(gui.Label("x"))
        res_row.add_child(self._dlg_rh)
        layout.add_child(res_row)

        note = gui.Label("Use width x height values to set the camera resolution.")
        note.text_color = gui.Color(0.36, 0.36, 0.36)
        layout.add_child(note)

        self._dlg_near = self._float_ne(cam.near, 78)
        layout.add_child(self._kv_row("Near plane   (mm)", self._dlg_near))

        self._dlg_far = self._float_ne(cam.far, 78)
        layout.add_child(self._kv_row("Far plane    (mm)", self._dlg_far))

        color_row = gui.Horiz(4)
        color_row.add_child(self._key_lbl("Camera color"))
        color_row.add_stretch()
        self._dlg_color_picker = gui.ColorEdit()
        self._dlg_color_picker.color_value = gui.Color(
            float(cam_color[0]), float(cam_color[1]), float(cam_color[2]))
        color_row.add_child(self._dlg_color_picker)
        layout.add_child(color_row)

        btn_row = gui.Horiz(4)
        btn_ok = gui.Button("Add Camera")
        btn_ok.set_on_clicked(self._on_create_camera_from_dialog)
        btn_cancel = gui.Button("Cancel")
        btn_cancel.set_on_clicked(lambda: self.win.close_dialog())
        btn_row.add_child(btn_ok)
        btn_row.add_child(btn_cancel)
        layout.add_child(btn_row)

        layout.preferred_width = 480
        dlg.add_child(layout)
        self.win.show_dialog(dlg)
        self._camera_dialog = dlg
        
        # Apply initial preset
        self._on_sensor_preset_changed(6, None, sensor_presets)
        self._on_resolution_preset_changed(7, None, resolution_presets)

    def _show_camera_dialog_for_edit(self):
        """Open edit dialog for the selected camera."""
        cam = self._cam()
        cam_color = self._camera_color(self.active_camera_idx)
        dlg = gui.Dialog("Edit Camera")
        layout = gui.Vert(4)

        # -- Sensor presets section (from Basler ACE2 cameras) --
        layout.add_child(self._section_lbl("[ SENSOR PRESETS ]"))
        
        sensor_presets = [
            ("Full-frame 35mm (36x24mm)", 36.0, "3:2"),
            ("APS-C (23.5x15.6mm)", 23.5, "3:2"),
            ("1\" (25.4x19.1mm)", 25.4, "3:2"),
            ("1.1\" (27.9x20.9mm)", 27.9, "3:2"),
            ("1.2\" (30.5x22.9mm)", 30.5, "3:2"),
            ("2/3\" (8.8x6.6mm)", 8.8, "4:3"),
            ("1/1.1\" (13.2x8.8mm) [Basler]", 13.2, "3:2"),
            ("1/1.8\" (7.6x5.7mm) [Basler]", 7.6, "3:2"),
            ("1/2.3\" (5.76x4.29mm) [Basler]", 5.76, "3:2"),
            ("1/2.8\" (4.8x3.6mm) [Basler]", 4.8, "4:3"),
        ]
        
        sensor_row = gui.Horiz(4)
        sensor_row.add_child(self._key_lbl("Select sensor:"))
        sensor_row.add_stretch()
        self._dlg_sensor_combo = gui.Combobox()
        for name, _, _ in sensor_presets:
            self._dlg_sensor_combo.add_item(name)
        self._dlg_sensor_combo.set_on_selection_changed(
            lambda a, b: self._on_sensor_preset_changed(a, b, sensor_presets))
        sensor_row.add_child(self._dlg_sensor_combo)
        layout.add_child(sensor_row)

        # -- Resolution presets section (Basler ACE2 cameras + standards) --
        layout.add_child(self._section_lbl("[ RESOLUTION PRESETS ]"))
        
        resolution_presets = [
            ("2.3 MP - Basler ACE2 (1920x1200)", 1920, 1200),
            ("3.2 MP - Basler ACE2 (2048x1536)", 2048, 1536),
            ("5 MP - Basler ACE2 (2448x2048)", 2448, 2048),
            ("8 MP - Basler ACE2 (3840x2160)", 3840, 2160),
            ("8.3 MP - Basler ACE2 (3264x2448)", 3264, 2448),
            ("12 MP - Standard (4000x3000)", 4000, 3000),
            ("12.3 MP - Basler ACE2 (4096x3072)", 4096, 3072),
            ("16 MP - Standard (4688x3516)", 4688, 3516),
            ("16.1 MP - Basler ACE2 (5320x3020)", 5320, 3020),
            ("18.4 MP - Basler ACE2 (4880x3760)", 4880, 3760),
            ("20 MP - Standard (5184x3888)", 5184, 3888),
            ("20.2 MP - Basler ACE2 (4504x4480)", 4504, 4480),
            ("24 MP - Standard (6000x4000)", 6000, 4000),
            ("24.4 MP - Basler ACE2 (5328x4584)", 5328, 4584),
            ("25 MP - Basler ACE2 (5060x4948)", 5060, 4948),
            ("1080p HD (1920x1080)", 1920, 1080),
            ("2K DCI (2560x1440)", 2560, 1440),
            ("4K UHD (3840x2160)", 3840, 2160),
        ]
        
        res_row = gui.Horiz(4)
        res_row.add_child(self._key_lbl("Select resolution:"))
        res_row.add_stretch()
        self._dlg_res_combo = gui.Combobox()
        for name, _, _ in resolution_presets:
            self._dlg_res_combo.add_item(name)
        self._dlg_res_combo.set_on_selection_changed(
            lambda a, b: self._on_resolution_preset_changed(a, b, resolution_presets))
        res_row.add_child(self._dlg_res_combo)
        layout.add_child(res_row)
        
        layout.add_child(self._key_lbl("Select from real Basler ACE2 cameras or standard formats."))

        pos_block, (self._dlg_cx, self._dlg_cy, self._dlg_cz) = self._xyz_block(
            "Position (mm)", cam.position)
        layout.add_child(pos_block)

        look_block, (self._dlg_lx, self._dlg_ly, self._dlg_lz) = self._xyz_block(
            "Look-At  (mm)", cam.lookat)
        layout.add_child(look_block)

        self._dlg_fl = self._float_ne(cam.focal_length, 78)
        layout.add_child(self._kv_row("Focal length (mm)", self._dlg_fl))

        self._dlg_sw = self._float_ne(cam.sensor_width, 78)
        layout.add_child(self._kv_row("Sensor width  (mm)", self._dlg_sw))

        self._dlg_rw = self._int_ne(cam.image_width,  150)
        self._dlg_rh = self._int_ne(cam.image_height, 150)
        res_row = gui.Horiz(4)
        res_row.add_child(self._key_lbl("Resolution (px)"))
        res_row.add_stretch()
        res_row.add_child(self._dlg_rw)
        res_row.add_child(gui.Label("x"))
        res_row.add_child(self._dlg_rh)
        layout.add_child(res_row)

        note = gui.Label("Use width x height values to set the camera resolution.")
        note.text_color = gui.Color(0.36, 0.36, 0.36)
        layout.add_child(note)

        self._dlg_near = self._float_ne(cam.near, 78)
        layout.add_child(self._kv_row("Near plane   (mm)", self._dlg_near))

        self._dlg_far = self._float_ne(cam.far, 78)
        layout.add_child(self._kv_row("Far plane    (mm)", self._dlg_far))

        color_row = gui.Horiz(4)
        color_row.add_child(self._key_lbl("Camera color"))
        color_row.add_stretch()
        self._dlg_color_picker = gui.ColorEdit()
        self._dlg_color_picker.color_value = gui.Color(
            float(cam_color[0]), float(cam_color[1]), float(cam_color[2]))
        color_row.add_child(self._dlg_color_picker)
        layout.add_child(color_row)

        btn_row = gui.Horiz(4)
        btn_ok = gui.Button("Update Camera")
        btn_ok.set_on_clicked(self._on_update_camera_from_dialog)
        btn_cancel = gui.Button("Cancel")
        btn_cancel.set_on_clicked(lambda: self.win.close_dialog())
        btn_row.add_child(btn_ok)
        btn_row.add_child(btn_cancel)
        layout.add_child(btn_row)

        layout.preferred_width = 480
        dlg.add_child(layout)
        self.win.show_dialog(dlg)
        self._camera_dialog = dlg

        # Reflect current camera configuration in preset dropdowns.
        sensor_idx = self._find_sensor_preset_index(cam.sensor_width, sensor_presets)
        if sensor_idx >= 0:
            self._dlg_sensor_combo.selected_index = sensor_idx
        res_idx = self._find_resolution_preset_index(
            cam.image_width, cam.image_height, resolution_presets)
        if res_idx >= 0:
            self._dlg_res_combo.selected_index = res_idx

    def _find_sensor_preset_index(self, sensor_width: float, presets: list) -> int:
        best_idx = -1
        best_err = 1e9
        for i, (_, width, _ratio) in enumerate(presets):
            err = abs(float(width) - float(sensor_width))
            if err < best_err:
                best_err = err
                best_idx = i
        return best_idx if best_err <= 0.25 else -1

    def _find_resolution_preset_index(self, width: int, height: int, presets: list) -> int:
        for i, (_name, w, h) in enumerate(presets):
            if int(w) == int(width) and int(h) == int(height):
                return i
        return -1

    def _combo_to_index(self, a, b, n_items: int) -> int:
        """Extract selected index from Open3D combobox callback args robustly."""
        for v in (a, b):
            if isinstance(v, int):
                if 0 <= v < n_items:
                    return v
            elif isinstance(v, str):
                try:
                    iv = int(v)
                    if 0 <= iv < n_items:
                        return iv
                except ValueError:
                    continue
        return -1

    def _on_sensor_preset_changed(self, a, b, presets: list):
        """Update sensor_width when sensor preset is changed."""
        if not hasattr(self, "_dlg_sw"):
            return
        idx = self._combo_to_index(a, b, len(presets))
        if 0 <= idx < len(presets):
            _, sensor_width, _ = presets[idx]
            self._dlg_sw.double_value = float(sensor_width)

    def _on_resolution_preset_changed(self, a, b, presets: list):
        """Update image_width and image_height when resolution preset is changed."""
        if not hasattr(self, "_dlg_rw") or not hasattr(self, "_dlg_rh"):
            return
        idx = self._combo_to_index(a, b, len(presets))
        if 0 <= idx < len(presets):
            _, width, height = presets[idx]
            self._dlg_rw.int_value = int(width)
            self._dlg_rh.int_value = int(height)

    def _on_create_camera_from_dialog(self, *args):
        cam = CameraModel()
        cam.position = np.array([
            self._dlg_cx.double_value, self._dlg_cy.double_value, self._dlg_cz.double_value])
        cam.lookat = np.array([
            self._dlg_lx.double_value, self._dlg_ly.double_value, self._dlg_lz.double_value])
        cam.focal_length = max(1.0, self._dlg_fl.double_value)
        cam.sensor_width = max(1.0, self._dlg_sw.double_value)
        cam.image_width = max(1, self._dlg_rw.int_value)
        cam.image_height = max(1, self._dlg_rh.int_value)
        cam.near = max(1.0, self._dlg_near.double_value)
        cam.far = max(cam.near + 1.0, self._dlg_far.double_value)
        cam_color = self._dialog_color_value()
        self.cameras.append(cam)
        self.camera_colors.append(cam_color)
        self.active_camera_idx = len(self.cameras) - 1
        self._set_selection("camera", self.active_camera_idx)
        self._update_camera_visuals()
        self._update_camera_panel()
        self.win.close_dialog()
        self._set_status("New camera added.")
        self._autosave_now(show_status=False)

    def _on_update_camera_from_dialog(self, *args):
        """Update the selected camera from dialog values."""
        cam = self._cam()
        cam.position = np.array([
            self._dlg_cx.double_value, self._dlg_cy.double_value, self._dlg_cz.double_value])
        cam.lookat = np.array([
            self._dlg_lx.double_value, self._dlg_ly.double_value, self._dlg_lz.double_value])
        cam.focal_length = max(1.0, self._dlg_fl.double_value)
        cam.sensor_width = max(1.0, self._dlg_sw.double_value)
        cam.image_width = max(1, self._dlg_rw.int_value)
        cam.image_height = max(1, self._dlg_rh.int_value)
        cam.near = max(1.0, self._dlg_near.double_value)
        cam.far = max(cam.near + 1.0, self._dlg_far.double_value)
        
        cam_color = self._dialog_color_value()
        self.camera_colors[self.active_camera_idx] = cam_color
        
        self._set_selection("camera", self.active_camera_idx)
        self._update_camera_visuals()
        self._update_camera_panel()
        self.win.close_dialog()
        self._set_status(f"Camera {self.active_camera_idx + 1} updated.")
        self._autosave_now(show_status=False)

    def _on_update_object(self):
        self._read_object_ui()
        self._apply_object_transform()
        self._update_object_panel()
        self._set_status("Object transformed.")
        self._autosave_now(show_status=False)

    def _on_quick_rotate(self, axis: str, direction: float):
        self._read_object_ui()
        step = max(0.001, float(self._rot_step.double_value))
        delta = direction * step
        if axis == "X":
            self.object_rotation[0] += delta
        elif axis == "Y":
            self.object_rotation[1] += delta
        else:
            self.object_rotation[2] += delta
        self._apply_object_transform()
        self._update_object_panel()
        self._set_status(f"Rotated {axis} by {delta:.2f} deg")
        self._autosave_now(show_status=False)

    def _on_quick_scale(self, factor: float):
        """Apply a quick scale factor to the current object."""
        self._read_object_ui()
        self.object_scale = max(0.01, self.object_scale * factor)
        self._apply_object_transform()
        self._update_object_panel()
        self._set_status(f"Object scaled by {factor:g}x (now {self.object_scale:.3f})")
        self._autosave_now(show_status=False)

    # -- Section: Object --------------------------------------

    def _build_mesh_section(self):
        self.panel.add_child(self._section_lbl("[ OBJECT ]"))

        btn_load = gui.Button("Load Object (Mesh or Point Cloud)")
        btn_load.set_on_clicked(self._on_load_mesh)
        self.panel.add_child(btn_load)
        self.panel.add_child(self._key_lbl("Supported: STL/OBJ/PLY/GLB and COLMAP points3D.txt"))

        self._glb_unit_scale = self._float_ne(1000.0, 78)
        self.panel.add_child(self._kv_row("GLB mm per unit", self._glb_unit_scale))
        self.panel.add_child(self._key_lbl("Use 1000 for meter-based GLB, 1 for mm-based GLB."))

        self.panel.add_child(self._key_lbl("-- or create a cuboid --"))

        # L / W / H on one row with plain letter labels
        lwh_row = gui.Horiz(4)
        self._cube_lx = self._float_ne(1.0, 58)
        self._cube_ly = self._float_ne(0.5, 58)
        self._cube_lz = self._float_ne(0.5, 58)
        for label, ne in (("L", self._cube_lx),
                          ("W", self._cube_ly),
                          ("H", self._cube_lz)):
            sub = gui.Horiz(2)
            al = gui.Label(label)
            al.text_color = gui.Color(0.40, 0.80, 0.40)
            sub.add_child(al)
            sub.add_child(ne)
            lwh_row.add_child(sub)
        self.panel.add_child(lwh_row)

        btn_cube = gui.Button("Create Cuboid")
        btn_cube.set_on_clicked(self._on_create_cuboid)
        self.panel.add_child(btn_cube)

        self._mesh_status = gui.Label("No object loaded")
        self._mesh_status.text_color = gui.Color(0.52, 0.52, 0.52)
        self.panel.add_child(self._mesh_status)
        
        # Scaling info
        scale_info = gui.Label("To scale mesh: adjust Scale field or use Quick scale buttons below")
        scale_info.text_color = gui.Color(0.70, 0.70, 0.70)
        self.panel.add_child(scale_info)

    # -- Section: Camera --------------------------------------

    def _build_camera_section(self):
        self.panel.add_child(self._section_lbl("[ CAMERA ]"))

        sel_row = gui.Horiz(6)
        self._cam_select_lbl = gui.Label("Camera 1 / 1")
        self._cam_select_lbl.text_color = gui.Color(0.65, 0.85, 1.0)
        sel_row.add_child(self._cam_select_lbl)
        sel_row.add_stretch()
        btn_config = gui.Button("Edit")
        btn_config.set_on_clicked(self._on_edit_selected_camera)
        sel_row.add_child(btn_config)
        btn_delete = gui.Button("Delete")
        btn_delete.set_on_clicked(self._on_delete_selected_camera)
        sel_row.add_child(btn_delete)
        self.panel.add_child(sel_row)

        # Display current sensor & resolution info
        self._cam_sensor_info = gui.Label("Sensor: --")
        self._cam_sensor_info.text_color = gui.Color(0.60, 0.80, 0.95)
        self.panel.add_child(self._cam_sensor_info)

        self._cam_resolution_info = gui.Label("Resolution: --")
        self._cam_resolution_info.text_color = gui.Color(0.60, 0.80, 0.95)
        self.panel.add_child(self._cam_resolution_info)

        pos_block, (self._cx, self._cy, self._cz) = self._xyz_block(
            "Position (mm)", self._cam().position)
        self.panel.add_child(pos_block)

        lat_block, (self._lx, self._ly, self._lz) = self._xyz_block(
            "Look-At  (mm)", self._cam().lookat)
        self.panel.add_child(lat_block)

        self._fl = self._float_ne(self._cam().focal_length, 78)
        self.panel.add_child(self._kv_row("Focal length (mm)", self._fl))

        self._sw = self._float_ne(self._cam().sensor_width, 78)
        self.panel.add_child(self._kv_row("Sensor width  (mm)", self._sw))

        self._rw = self._int_ne(self._cam().image_width,  90)
        self._rh = self._int_ne(self._cam().image_height, 90)
        res_row = gui.Horiz(6)
        res_row.add_child(self._key_lbl("Resolution (px)"))
        res_row.add_stretch()
        res_row.add_child(self._rw)
        res_row.add_child(gui.Label("x"))
        res_row.add_child(self._rh)
        self.panel.add_child(res_row)
        self.panel.add_child(self._key_lbl("Set resolution width and height for the selected camera."))

        self._near = self._float_ne(self._cam().near, 78)
        self.panel.add_child(self._kv_row("Near plane   (mm)", self._near))

        self._far = self._float_ne(self._cam().far, 78)
        self.panel.add_child(self._kv_row("Far plane    (mm)", self._far))

        self._marker_radius = self._float_ne(self.camera_marker_radius, 78)
        self.panel.add_child(self._kv_row("Camera marker radius (mm)", self._marker_radius))

        roll_row = gui.Horiz(4)
        roll_row.add_child(self._key_lbl("Roll around optical axis"))
        roll_row.add_stretch()
        btn_roll_m = gui.Button("Roll -90")
        btn_roll_p = gui.Button("Roll +90")
        btn_roll_m.set_on_clicked(lambda: self._on_roll_camera(-90.0))
        btn_roll_p.set_on_clicked(lambda: self._on_roll_camera(+90.0))
        roll_row.add_child(btn_roll_m)
        roll_row.add_child(btn_roll_p)
        self.panel.add_child(roll_row)

    # -- Section: Actions -------------------------------------

    def _build_action_section(self):
        self.panel.add_child(self._section_lbl("[ ACTIONS ]"))

        sel_row = gui.Horiz(6)
        self._selection_dot = gui.Label("###")
        self._selection_dot.text_color = gui.Color(0.10, 0.75, 0.95)
        self._selection_label = gui.Label("Selected: Camera 1")
        self._selection_label.text_color = self._ui["key"]
        sel_row.add_child(self._selection_dot)
        sel_row.add_child(self._selection_label)
        self.panel.add_child(sel_row)

        self.panel.add_child(self._section_lbl("[ STATUS CHIPS ]"))
        chip_row1 = gui.Horiz(6)
        self._chip_mode = gui.Label("Mode: Idle")
        self._chip_mode.text_color = self._ui["muted"]
        self._chip_cov = gui.Label("Coverage: --")
        self._chip_cov.text_color = self._ui["muted"]
        chip_row1.add_child(self._chip_mode)
        chip_row1.add_child(self._chip_cov)
        self.panel.add_child(chip_row1)

        chip_row2 = gui.Horiz(6)
        self._chip_detect = gui.Label("Detectability: --")
        self._chip_detect.text_color = self._ui["muted"]
        chip_row2.add_child(self._chip_detect)
        self.panel.add_child(chip_row2)

        self._build_camera_legend_section()

        self.panel.add_child(self._section_lbl("[ PRIMARY ]"))
        btn_cov = gui.Button("Compute Coverage")
        btn_cov.set_on_clicked(self._on_compute_coverage)
        self.panel.add_child(btn_cov)

        btn_upd = gui.Button("Update Camera")
        btn_upd.set_on_clicked(self._on_update_camera)
        self.panel.add_child(btn_upd)

        self.panel.add_child(self._section_lbl("[ SECONDARY ]"))
        btn_add = gui.Button("Add Camera")
        btn_add.set_on_clicked(self._on_add_camera)
        self.panel.add_child(btn_add)

        btn_next = gui.Button("Switch Camera")
        btn_next.set_on_clicked(self._on_next_camera)
        self.panel.add_child(btn_next)

        btn_show_overall = gui.Button("Show Overall Coverage")
        btn_show_overall.set_on_clicked(self._on_show_overall_coverage)
        self.panel.add_child(btn_show_overall)

        btn_reset_cov = gui.Button("Reset Coverage View")
        btn_reset_cov.set_on_clicked(self._on_reset_coverage_view)
        self.panel.add_child(btn_reset_cov)

        self._status_lbl = gui.Label("Ready.")
        self._status_lbl.text_color = self._ui["muted"]
        self.panel.add_child(self._status_lbl)

        self._n_samples = self._int_ne(20000, 80)
        self.panel.add_child(self._kv_row("Sample points", self._n_samples))

        self.panel.add_child(self._hsep())
        self.panel.add_child(self._section_lbl("[ REPORTS & TRIALS ]"))
        btn_export = gui.Button("Export Report")
        btn_export.set_on_clicked(self._on_export_report)
        self.panel.add_child(btn_export)
        trial_row = gui.Horiz(4)
        btn_save_trial = gui.Button("Save Trial")
        btn_save_trial.set_on_clicked(self._on_save_trial)
        btn_load_trial = gui.Button("Load Trial")
        btn_load_trial.set_on_clicked(self._on_load_trial)
        trial_row.add_child(btn_save_trial)
        trial_row.add_child(btn_load_trial)
        self.panel.add_child(trial_row)
        self.panel.add_child(self._key_lbl("Report: screenshot + HTML with camera config."))
        self.panel.add_child(self._key_lbl("Trial: full state to resume experiment later."))

    def _build_camera_legend_section(self):
        self.panel.add_child(self._section_lbl("[ CAMERA LEGEND ]"))
        self._legend_rows = []
        self._legend_max = 20
        for i in range(self._legend_max):
            row = gui.Horiz(4)
            swatch = gui.Label("###")
            swatch.text_color = gui.Color(0.7, 0.7, 0.7)
            name_btn = gui.Button(f"Camera {i + 1}")
            name_btn.set_on_clicked(lambda idx=i: self._on_legend_camera_click(idx))
            row.add_child(swatch)
            row.add_child(name_btn)
            row.visible = False
            self.panel.add_child(row)
            self._legend_rows.append((row, swatch, name_btn))

    def _on_legend_camera_click(self, idx: int):
        if idx < len(self.cameras):
            self._select_camera(idx)
            self._update_camera_visuals()

    def _refresh_camera_legend(self):
        self._ensure_camera_color_count()
        for i, (row, swatch, name_btn) in enumerate(self._legend_rows):
            if i < len(self.cameras):
                c = self._camera_color(i)
                row.visible = True
                swatch.text = "###"
                swatch.text_color = gui.Color(float(c[0]), float(c[1]), float(c[2]))
                if i == self.active_camera_idx:
                    name_btn.text = f"> Camera {i + 1} (selected)"
                else:
                    name_btn.text = f"Camera {i + 1}"
            else:
                row.visible = False

    # -- Section: Coverage stats ------------------------------

    def _build_stats_section(self):
        self.panel.add_child(self._section_lbl("[ COVERAGE STATS ]"))
        self._sv = {}

        stat_defs = [
            ("coverage", "Overall coverage"),
            ("camera_coverage", "Selected coverage"),
            ("visible", "Overall visible pts"),
            ("camera_visible", "Selected visible pts"),
            ("total", "Sample points"),
            ("avg_dist", "Avg distance (mm)"),
            ("mm_per_px", "Resolution (mm/px)"),
            ("fov_h_deg", "FOV H (deg)"),
            ("fov_v_deg", "FOV V (deg)"),
        ]
        for key, name in stat_defs:
            row = gui.Horiz(6)
            k = gui.Label(name)
            k.text_color = self._ui["key"]
            row.add_child(k)
            row.add_stretch()
            v = self._val_lbl("--")
            row.add_child(v)
            self.panel.add_child(row)
            self._sv[key] = v

        self.panel.add_child(self._hsep())
        self.panel.add_child(self._section_lbl("[ DEFECT DETECTABILITY ]"))
        
        self._defect_size = self._float_ne(1.0, 78)
        self.panel.add_child(self._kv_row("Defect size (mm)", self._defect_size))
        self.panel.add_child(self._key_lbl("Recompute coverage to apply detectability mode."))
        
        # Thresholds display
        self.panel.add_child(self._key_lbl("Detection thresholds:"))
        self.panel.add_child(self._key_lbl("RED: < 3 px (undetectable)"))
        self.panel.add_child(self._key_lbl("YELLOW: 3-8 px (marginal)"))
        self.panel.add_child(self._key_lbl("GREEN: >= 8 px (detectable)"))
        
        btn_refresh_defect = gui.Button("Refresh Detectability")
        btn_refresh_defect.set_on_clicked(self._on_refresh_detectability)
        self.panel.add_child(btn_refresh_defect)

    # -- Section: Navigation hints ----------------------------

    def _build_nav_section(self):
        self.panel.add_child(self._section_lbl("[ 3D NAVIGATION ]"))
        hints = [
            ("Orbit",        "Left-drag"),
            ("Pan",          "Right-drag"),
            ("Zoom",         "2-finger scroll"),
            ("Zoom (mouse)", "Scroll wheel"),
            ("Reset view",   "F key"),
        ]
        for action, gesture in hints:
            row = gui.Horiz(4)
            a = gui.Label(action)
            a.text_color = gui.Color(0.58, 0.58, 0.58)
            g = gui.Label(gesture)
            g.text_color = gui.Color(0.48, 0.74, 0.95)
            row.add_child(a)
            row.add_stretch()
            row.add_child(g)
            self.panel.add_child(row)

        # Plain ASCII - no Unicode warning sign
        note = gui.Label(
            "Note: native pinch-to-zoom is not\n"
            "supported by Open3D. Use 2-finger\n"
            "scroll to zoom on the trackpad.")
        note.text_color = gui.Color(0.58, 0.56, 0.38)
        self.panel.add_child(note)

    # -- Scene materials --------------------------------------

    def _mat_lit(self, color, alpha=1.0):
        m = rendering.MaterialRecord()
        m.base_color = [*color, alpha]
        m.shader = "defaultLit"
        return m

    def _mat_unlit(self, color, alpha=1.0):
        m = rendering.MaterialRecord()
        m.base_color = [*color, alpha]
        m.shader = "defaultUnlit"
        return m

    def _mat_line(self, color, width=1.5):
        m = rendering.MaterialRecord()
        m.base_color = [*color, 1.0]
        m.shader = "unlitLine"
        m.line_width = width
        return m

    def _mat_point(self, size=4.5):
        m = rendering.MaterialRecord()
        m.shader = "defaultUnlit"
        m.point_size = size
        return m

    def _remove(self, name: str):
        try:
            self.scene_widget.scene.remove_geometry(name)
        except Exception:
            pass

    # -- Reference grid ---------------------------------------

    def _add_reference_grid(self, size=4.0, n=8):
        pts, lines = [], []
        step = size / n
        for i in range(n + 1):
            x = -size / 2 + i * step
            p0 = len(pts)
            pts += [[-size/2, x, 0.0], [size/2, x, 0.0]]
            lines.append([p0, p0 + 1])
            p1 = len(pts)
            pts += [[x, -size/2, 0.0], [x, size/2, 0.0]]
            lines.append([p1, p1 + 1])
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines  = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector([[0.75, 0.75, 0.75]] * len(lines))
        self.scene_widget.scene.add_geometry(
            self._GRID, ls, self._mat_line([0.60, 0.60, 0.60], width=1.0))

    def _add_reference_axes(self, size=4.0):
        pts = [[0.0, 0.0, 0.0], [size, 0.0, 0.0], [0.0, size, 0.0], [0.0, 0.0, size]]
        lines = [[0, 1], [0, 2], [0, 3]]
        colors = [[1.0, 0.2, 0.2], [0.2, 0.7, 0.2], [0.2, 0.3, 1.0]]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector(colors)
        self.scene_widget.scene.add_geometry(
            self._AXES, ls, self._mat_line([0.0, 0.0, 0.0], width=3.5))
        # Arrowhead cones at axis tips (create_cone apex is along +Z by default)
        cr, ch = size * 0.04, size * 0.10        # cone radius and height
        base_d = size - ch * 0.3                 # base slightly inside line tip
        # Rotation matrices to align cone (+Z) to each world axis
        Ry90   = np.array([[0,0,1],[0,1,0],[-1,0,0]], dtype=float)  # Z → X
        Rx_n90 = np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=float)  # Z → Y
        for name, rot, base_pos, col in [
            ("axes_cx", Ry90,      [base_d, 0.0,    0.0   ], [1.0, 0.2, 0.2]),
            ("axes_cy", Rx_n90,    [0.0,    base_d, 0.0   ], [0.2, 0.7, 0.2]),
            ("axes_cz", np.eye(3), [0.0,    0.0,    base_d], [0.2, 0.3, 1.0]),
        ]:
            cone = o3d.geometry.TriangleMesh.create_cone(radius=cr, height=ch, resolution=16)
            cone.compute_vertex_normals()
            cone.paint_uniform_color(col)
            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3]  = np.array(base_pos)
            cone.transform(T)
            self.scene_widget.scene.add_geometry(name, cone, self._mat_lit(col))
        # 3-D axis labels – placed slightly beyond each cone apex
        ld = base_d + ch * 1.25
        lx = self.scene_widget.add_3d_label([ld,  0.0, 0.0], "X")
        ly = self.scene_widget.add_3d_label([0.0, ld,  0.0], "Y")
        lz = self.scene_widget.add_3d_label([0.0, 0.0, ld ], "Z")
        lx.color = gui.Color(1.0, 0.3, 0.3)
        ly.color = gui.Color(0.3, 0.85, 0.3)
        lz.color = gui.Color(0.4, 0.5, 1.0)
        for lbl in (lx, ly, lz):
            try:
                lbl.font_size = 24
            except AttributeError:
                pass
        self._axis_labels = [lx, ly, lz]

    # -- Mesh management --------------------------------------

    def _set_mesh(self, mesh: o3d.geometry.TriangleMesh, label: str):
        self.base_mesh = copy.deepcopy(mesh)
        self.base_points = None
        self.base_point_colors = None
        self.point_cloud = None
        self.mesh_transform = np.eye(4)
        self.object_translation = np.array([0.0, 0.0, 0.0], dtype=float)
        self.object_rotation = np.array([0.0, 0.0, 0.0], dtype=float)
        self.object_scale = 1.0
        self.mesh = mesh
        self.surface_pts = None
        self.vis_engine = None
        self._remove(self._MESH)
        self._remove(self._POINTS)
        self._remove(self._CAMCOVERAGE)
        self._remove(self._COVERAGE)
        self.scene_widget.scene.add_geometry(
            self._MESH, mesh, self._mat_lit([0.55, 0.60, 0.65]))
        bb = mesh.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(60, bb, bb.get_center())
        self._mesh_status.text = f"{label}  ({len(mesh.triangles)} tris)"
        self._set_status("Mesh loaded. Ready.")
        self._update_object_panel()

    def _set_point_cloud(self, pcd: o3d.geometry.PointCloud, label: str):
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            raise ValueError("Point cloud is empty")

        cols = None
        if len(pcd.colors) == len(pts):
            cols = np.asarray(pcd.colors).copy()

        self.base_mesh = None
        self.mesh = None
        self.base_points = pts.copy()
        self.base_point_colors = cols
        self.point_cloud = copy.deepcopy(pcd)

        self.mesh_transform = np.eye(4)
        self.object_translation = np.array([0.0, 0.0, 0.0], dtype=float)
        self.object_rotation = np.array([0.0, 0.0, 0.0], dtype=float)
        self.object_scale = 1.0

        self.surface_pts = None
        self.vis_engine = None
        self._remove(self._MESH)
        self._remove(self._POINTS)
        self._remove(self._CAMCOVERAGE)
        self._remove(self._COVERAGE)
        self.scene_widget.scene.add_geometry(self._POINTS, pcd, self._mat_point(3.5))

        bb = pcd.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(60, bb, bb.get_center())
        self._mesh_status.text = f"{label}  ({len(pts)} pts)"
        self._set_status("Point cloud loaded. Coverage will run in no-occlusion mode.")
        self._update_object_panel()

    def _apply_pointcloud_transform(self):
        if self.base_points is None:
            return
        M = self._compute_object_transform()
        pts_h = np.concatenate(
            [self.base_points, np.ones((len(self.base_points), 1), dtype=float)], axis=1)
        pts_t = (M @ pts_h.T).T[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_t)
        if self.base_point_colors is not None and len(self.base_point_colors) == len(pts_t):
            pcd.colors = o3d.utility.Vector3dVector(self.base_point_colors)
        self.point_cloud = pcd
        self.surface_pts = None
        self.vis_engine = None
        self._remove(self._POINTS)
        self.scene_widget.scene.add_geometry(self._POINTS, pcd, self._mat_point(3.5))

    def _apply_object_transform(self):
        if self.base_mesh is not None:
            self._apply_mesh_transform()
        elif self.base_points is not None:
            self._apply_pointcloud_transform()

    def _load_mesh_from_path(self, path: str):
        try:
            ext = os.path.splitext(path)[1].lower()
            import_note = ""
            mesh = None
            
            if ext == ".glb":
                if not TRIMESH_AVAILABLE:
                    raise RuntimeError("GLB loading requires trimesh. Install with: pip install trimesh")
                loaded = trimesh.load(path, force="scene")
                if isinstance(loaded, trimesh.Scene):
                    if len(loaded.geometry) == 0:
                        raise ValueError("No mesh geometry found in GLB.")
                    # Export a concatenated mesh with scene graph transforms applied.
                    # Use to_geometry() if available (trimesh >= 3.17), fallback to dump()
                    try:
                        merged = loaded.to_geometry()
                    except (AttributeError, TypeError):
                        merged = loaded.dump(concatenate=True)
                else:
                    merged = loaded
                if merged.faces is None or len(merged.faces) == 0:
                    raise ValueError("No triangles found.")
                mesh = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(np.asarray(merged.vertices, dtype=np.float64)),
                    o3d.utility.Vector3iVector(np.asarray(merged.faces, dtype=np.int32)),
                )
                glb_scale = max(1e-9, float(self._glb_unit_scale.double_value))
                if abs(glb_scale - 1.0) > 1e-9:
                    bb = mesh.get_axis_aligned_bounding_box()
                    mesh.scale(glb_scale, center=bb.get_center())
                    import_note = f" (GLB scale x{glb_scale:g})"
            else:
                mesh = o3d.io.read_triangle_mesh(path)
            
            # Handle mesh if loaded
            if mesh is not None and len(mesh.triangles) > 0:
                mesh.compute_vertex_normals()
                mesh.paint_uniform_color([0.55, 0.60, 0.65])
                self._set_mesh(mesh, os.path.basename(path) + import_note)
                self._mesh_source = {"type": "mesh", "path": os.path.abspath(path)}
            else:
                # Fall back to point cloud import.
                lower_name = os.path.basename(path).lower()
                if lower_name == "points3d.txt":
                    pcd = parse_colmap_points3d_txt(path)
                else:
                    pcd = o3d.io.read_point_cloud(path)
                if len(pcd.points) == 0:
                    raise ValueError("No triangles or points found.")
                self._set_point_cloud(pcd, os.path.basename(path))
                self._mesh_source = {"type": "pointcloud", "path": os.path.abspath(path)}
            self._autosave_now(show_status=False)
        except Exception as e:
            self._set_status(f"Error: {e}")

    def _create_default_cuboid(self):
        mesh = create_cuboid_mesh(1000.0, 500.0, 500.0)
        self._set_mesh(mesh, "Cuboid 1000 x 500 x 500 mm")
        self._mesh_source = {"type": "cuboid", "dims": [1000.0, 500.0, 500.0]}

    # -- Camera visuals ---------------------------------------

    def _update_camera_visuals(self):
        for i in range(10):  # simple cleanup
            self._remove(f"{self._FRUSTUM}_{i}")
            self._remove(f"{self._MARKER}_{i}")

        # Remove previous coordinate labels before redrawing cameras.
        for lbl in getattr(self, "_camera_coord_labels", []):
            try:
                self.scene_widget.remove_3d_label(lbl)
            except Exception:
                pass
        self._camera_coord_labels = []

        # draw all cameras
        for i, cam in enumerate(self.cameras):
            base = self._camera_color(i)
            if i == self.active_camera_idx:
                color = np.clip(base * 1.0, 0.0, 1.0).tolist()
                marker_color = np.clip(base * 1.0, 0.0, 1.0).tolist()
            else:
                color = np.clip(base * 0.6, 0.0, 1.0).tolist()
                marker_color = np.clip(base * 0.8, 0.0, 1.0).tolist()
            self.scene_widget.scene.add_geometry(
                f"{self._FRUSTUM}_{i}",
                build_frustum_lineset(cam, color=color),
                self._mat_line(color)
            )

            self.scene_widget.scene.add_geometry(
                f"{self._MARKER}_{i}",
                build_camera_marker(cam.position, radius=self.camera_marker_radius, color=marker_color),
                self._mat_unlit(marker_color)
            )

            # Show camera coordinates in the 3D viewing pane near each marker.
            pos = cam.position
            label_pos = [float(pos[0]), float(pos[1]), float(pos[2] + self.camera_marker_radius * 1.4)]
            lbl = self.scene_widget.add_3d_label(
                label_pos,
                f"C{i + 1}: ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})")
            lbl.color = gui.Color(float(marker_color[0]), float(marker_color[1]), float(marker_color[2]))
            self._camera_coord_labels.append(lbl)
        self._refresh_selection_label()
        self._refresh_camera_legend()

    # -- Callbacks --------------------------------------------

    def _on_add_camera(self, *args):
        try:
            self._show_camera_dialog()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._set_status(f"Add Camera failed: {e}")

    def _on_next_camera(self):
        self.active_camera_idx = (self.active_camera_idx + 1) % len(self.cameras)
        self._set_selection("camera", self.active_camera_idx)
        self._update_camera_panel()
        self._update_camera_visuals()

    def _guess_sensor_size_name(self, sensor_width: float) -> str:
        """Try to match sensor width to a friendly name."""
        sensor_presets = {
            36.0: "Full-frame 35mm",
            23.5: "APS-C",
            25.4: "1\"",
            27.9: "1.1\"",
            30.5: "1.2\"",
            8.8: "2/3\"",
            13.2: "1/1.1\" (Basler)",
            7.6: "1/1.8\" (Basler)",
            5.76: "1/2.3\" (Basler)",
            4.8: "1/2.8\" (Basler)",
        }
        for width, name in sensor_presets.items():
            if abs(sensor_width - width) < 0.1:
                return name
        return f"{sensor_width:.1f}mm"

    def _update_camera_info_display(self):
        """Update sensor and resolution info display for current camera."""
        cam = self._cam()
        sensor_name = self._guess_sensor_size_name(cam.sensor_width)
        mp = (cam.image_width * cam.image_height) / 1_000_000
        self._cam_sensor_info.text = f"Sensor: {sensor_name}"
        self._cam_resolution_info.text = f"Resolution: {cam.image_width}x{cam.image_height} ({mp:.1f}MP)"
        self._cam_select_lbl.text = f"Camera {self.active_camera_idx + 1} / {len(self.cameras)}"

    def _on_edit_selected_camera(self):
        """Open camera dialog to edit selected camera."""
        try:
            self._show_camera_dialog_for_edit()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._set_status(f"Edit camera failed: {e}")

    def _on_delete_selected_camera(self):
        """Delete selected camera (with confirmation if more than 1)."""
        if len(self.cameras) <= 1:
            self._set_status("Error: must keep at least 1 camera.")
            return
        
        # Show confirmation dialog
        dlg = gui.Dialog("Confirm Delete")
        layout = gui.Vert(8)
        msg = gui.Label(f"Delete Camera {self.active_camera_idx + 1}?")
        msg.text_color = gui.Color(0.95, 0.85, 0.2)
        layout.add_child(msg)
        
        btn_row = gui.Horiz(4)
        btn_delete = gui.Button("Delete")
        btn_delete.set_on_clicked(lambda: self._confirm_delete_camera(dlg))
        btn_cancel = gui.Button("Cancel")
        btn_cancel.set_on_clicked(lambda: self.win.close_dialog())
        btn_row.add_child(btn_delete)
        btn_row.add_child(btn_cancel)
        layout.add_child(btn_row)
        
        dlg.add_child(layout)
        self.win.show_dialog(dlg)

    def _confirm_delete_camera(self, dlg):
        """Perform camera deletion after confirmation."""
        self.win.close_dialog()
        idx = self.active_camera_idx
        self.cameras.pop(idx)
        self.camera_colors.pop(idx)
        
        # Adjust active index if needed
        if self.active_camera_idx >= len(self.cameras):
            self.active_camera_idx = len(self.cameras) - 1
        
        self._set_selection("camera", self.active_camera_idx)
        self._update_camera_panel()
        self._update_camera_visuals()
        self._set_status(f"Camera {idx + 1} deleted.")
        self._autosave_now(show_status=False)

    def _on_load_mesh(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Select mesh file", self.win.theme)
        dlg.add_filter(
            ".stl .obj .ply .glb .pcd .xyz .xyzn .xyzrgb .pts .txt",
            "Object files (Mesh/PointCloud/COLMAP points3D.txt)")
        dlg.add_filter("", "All files")
        dlg.set_on_cancel(lambda: self.win.close_dialog())
        dlg.set_on_done(self._file_selected)
        self.win.show_dialog(dlg)

    def _file_selected(self, path: str):
        self.win.close_dialog()
        self._load_mesh_from_path(path)

    def _on_create_cuboid(self):
        lx = self._cube_lx.double_value
        ly = self._cube_ly.double_value
        lz = self._cube_lz.double_value
        if lx <= 0 or ly <= 0 or lz <= 0:
            self._set_status("Error: dimensions must be > 0")
            return
        self._set_mesh(create_cuboid_mesh(lx, ly, lz),
                       f"Cuboid {lx:.2f} x {ly:.2f} x {lz:.2f}")
        self._mesh_source = {"type": "cuboid", "dims": [float(lx), float(ly), float(lz)]}
        self._autosave_now(show_status=False)

    def _camera_to_dict(self, cam: CameraModel) -> dict:
        return {
            "position": cam.position.tolist(),
            "lookat": cam.lookat.tolist(),
            "focal_length": float(cam.focal_length),
            "sensor_width": float(cam.sensor_width),
            "image_width": int(cam.image_width),
            "image_height": int(cam.image_height),
            "near": float(cam.near),
            "far": float(cam.far),
            "up": cam.up.tolist(),
        }

    def _camera_from_dict(self, data: dict) -> CameraModel:
        cam = CameraModel()
        cam.position = np.array(data.get("position", cam.position.tolist()), dtype=float)
        cam.lookat = np.array(data.get("lookat", cam.lookat.tolist()), dtype=float)
        cam.focal_length = float(data.get("focal_length", cam.focal_length))
        cam.sensor_width = float(data.get("sensor_width", cam.sensor_width))
        cam.image_width = int(data.get("image_width", cam.image_width))
        cam.image_height = int(data.get("image_height", cam.image_height))
        cam.near = float(data.get("near", cam.near))
        cam.far = float(data.get("far", cam.far))
        cam.up = np.array(data.get("up", cam.up.tolist()), dtype=float)
        return cam

    def _build_project_state(self) -> dict:
        self._read_camera_ui()
        self._read_object_ui()
        return {
            "version": 1,
            "mesh": self._mesh_source,
            "object": {
                "translation": self.object_translation.tolist(),
                "rotation": self.object_rotation.tolist(),
                "scale": float(self.object_scale),
            },
            "cameras": [self._camera_to_dict(c) for c in self.cameras],
            "camera_colors": [self._camera_color(i).tolist() for i in range(len(self.cameras))],
            "active_camera_idx": int(self.active_camera_idx),
            "sample_points": int(self._n_samples.int_value),
            "camera_marker_radius": float(self.camera_marker_radius),
        }

    def _slugify_project_name(self, name: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
        return slug or "untitled-project"

    def _managed_project_dir(self, name: str) -> str:
        return os.path.join(self._projects_root, self._slugify_project_name(name))

    def _managed_project_json_path(self, project_dir: str) -> str:
        return os.path.join(project_dir, "project.json")

    def _managed_project_meta_path(self, project_dir: str) -> str:
        return os.path.join(project_dir, "metadata.json")

    def _timestamp_now(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _safe_relpath(self, path: str) -> str:
        try:
            return os.path.relpath(path, self._workspace_root)
        except Exception:
            return path

    def _build_project_metadata(self, project_name: str, state: dict,
                                project_dir: str, created_at: str | None = None) -> dict:
        mesh_info = state.get("mesh", {})
        mesh_type = mesh_info.get("type", "unknown")
        mesh_path = mesh_info.get("path", "")
        now = self._timestamp_now()
        return {
            "name": project_name,
            "slug": os.path.basename(project_dir),
            "created_at": created_at or now,
            "updated_at": now,
            "project_file": self._safe_relpath(self._managed_project_json_path(project_dir)),
            "object_type": mesh_type,
            "object_source": os.path.basename(mesh_path) if mesh_path else mesh_type,
            "camera_count": len(state.get("cameras", [])),
            "sample_points": int(state.get("sample_points", 0)),
        }

    def _read_project_metadata(self, project_dir: str) -> dict | None:
        meta_path = self._managed_project_meta_path(project_dir)
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _write_project_metadata(self, project_dir: str, meta: dict):
        with open(self._managed_project_meta_path(project_dir), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _list_managed_projects(self) -> list[dict]:
        projects = []
        if not os.path.isdir(self._projects_root):
            return projects
        for entry in os.scandir(self._projects_root):
            if not entry.is_dir():
                continue
            project_json = self._managed_project_json_path(entry.path)
            if not os.path.exists(project_json):
                continue
            meta = self._read_project_metadata(entry.path) or {}
            if not meta:
                created_ts = datetime.fromtimestamp(os.path.getctime(project_json)).isoformat(timespec="seconds")
                updated_ts = datetime.fromtimestamp(os.path.getmtime(project_json)).isoformat(timespec="seconds")
                meta = {
                    "name": entry.name,
                    "slug": entry.name,
                    "created_at": created_ts,
                    "updated_at": updated_ts,
                    "project_file": self._safe_relpath(project_json),
                    "object_type": "unknown",
                    "object_source": "unknown",
                    "camera_count": 0,
                    "sample_points": 0,
                }
            meta["project_dir"] = entry.path
            meta["project_json"] = project_json
            projects.append(meta)
        projects.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return projects

    def _format_project_timestamp(self, value: str) -> str:
        if not value:
            return "--"
        try:
            return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return value

    def _project_sort_options(self) -> list[tuple[str, str]]:
        return [
            ("Last updated", "updated_desc"),
            ("Created date", "created_desc"),
            ("Name", "name_asc"),
            ("Camera count", "camera_desc"),
        ]

    def _filter_sort_projects(self, projects: list[dict], query: str, sort_mode: str) -> list[dict]:
        q = (query or "").strip().lower()
        filtered = []
        for item in projects:
            hay = " ".join([
                str(item.get("name", "")),
                str(item.get("slug", "")),
                str(item.get("object_source", "")),
                str(item.get("object_type", "")),
            ]).lower()
            if not q or q in hay:
                filtered.append(item)

        if sort_mode == "created_desc":
            filtered.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        elif sort_mode == "name_asc":
            filtered.sort(key=lambda item: item.get("name", "").lower())
        elif sort_mode == "camera_desc":
            filtered.sort(key=lambda item: int(item.get("camera_count", 0)), reverse=True)
        else:
            filtered.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return filtered

    def _project_sort_mode_from_combo(self) -> str:
        if not hasattr(self, "_project_sort_combo"):
            return "updated_desc"
        idx = int(getattr(self._project_sort_combo, "selected_index", 0))
        options = self._project_sort_options()
        idx = max(0, min(idx, len(options) - 1))
        return options[idx][1]

    def _project_list_labels(self, projects: list[dict]) -> list[str]:
        labels = []
        for item in projects:
            labels.append(
                f"{item.get('name', 'Unnamed')}  |  {item.get('camera_count', 0)} cams  |  {item.get('object_source', 'unknown')}"
            )
        return labels

    def _update_project_browser_details(self, item: dict | None):
        if not hasattr(self, "_project_detail_label"):
            return
        if not item:
            self._project_detail_label.text = "No project selected."
            return
        self._project_detail_label.text = (
            f"Name: {item.get('name', '--')}\n"
            f"Created: {self._format_project_timestamp(item.get('created_at', ''))}\n"
            f"Updated: {self._format_project_timestamp(item.get('updated_at', ''))}\n"
            f"Object: {item.get('object_source', 'unknown')} ({item.get('object_type', 'unknown')})\n"
            f"Cameras: {item.get('camera_count', 0)}\n"
            f"Sample points: {item.get('sample_points', 0)}"
        )

    def _refresh_project_browser(self):
        all_projects = self._list_managed_projects()
        query = self._project_search_input.text_value if hasattr(self, "_project_search_input") else ""
        sort_mode = self._project_sort_mode_from_combo()
        self._project_browser_items = self._filter_sort_projects(all_projects, query, sort_mode)
        labels = self._project_list_labels(self._project_browser_items)
        self._project_list_view.set_items(labels)
        if self._project_browser_items:
            self._project_list_view.selected_index = 0
            self._update_project_browser_details(self._project_browser_items[0])
        else:
            self._update_project_browser_details(None)

    def _on_project_browser_selection_changed(self, _value, is_double_click):
        idx = int(getattr(self._project_list_view, "selected_index", -1))
        if 0 <= idx < len(self._project_browser_items):
            self._update_project_browser_details(self._project_browser_items[idx])
            if is_double_click:
                self._load_managed_project_selected(self._project_browser_items[idx]["project_json"])

    def _on_project_browser_load_clicked(self):
        idx = int(getattr(self._project_list_view, "selected_index", -1))
        if 0 <= idx < len(self._project_browser_items):
            self._load_managed_project_selected(self._project_browser_items[idx]["project_json"])
        else:
            self._set_status("No project selected.")

    def _save_project_named(self, project_name: str, show_status: bool = True):
        clean_name = project_name.strip() or "Untitled Project"
        project_dir = self._managed_project_dir(clean_name)
        os.makedirs(project_dir, exist_ok=True)
        project_json = self._managed_project_json_path(project_dir)
        state = self._build_project_state()
        with open(project_json, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        existing_meta = self._read_project_metadata(project_dir)
        created_at = existing_meta.get("created_at") if existing_meta else None
        meta = self._build_project_metadata(clean_name, state, project_dir, created_at=created_at)
        self._write_project_metadata(project_dir, meta)

        self._current_project_name = clean_name
        self._current_project_path = project_json
        if show_status:
            self._set_status(f"Project saved: {clean_name}")
        return project_json

    def _save_project_to_path(self, path: str, show_status: bool = True):
        out_path = path if path.lower().endswith(".json") else path + ".json"
        state = self._build_project_state()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        if show_status:
            self._set_status(f"Project saved: {os.path.basename(out_path)}")
        return out_path

    def _load_project_from_path(self, path: str, show_status: bool = True):
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        self._suspend_autosave = True
        try:
            mesh_info = state.get("mesh", {"type": "cuboid", "dims": [1.0, 0.5, 0.5]})
            mesh_type = mesh_info.get("type", "cuboid")
            if mesh_type == "mesh":
                mesh_path = mesh_info.get("path", "")
                if mesh_path and not os.path.isabs(mesh_path):
                    mesh_path = os.path.join(os.path.dirname(path), mesh_path)
                if not mesh_path or not os.path.exists(mesh_path):
                    raise FileNotFoundError(f"Mesh path not found: {mesh_path}")
                self._load_mesh_from_path(mesh_path)
            elif mesh_type == "pointcloud":
                cloud_path = mesh_info.get("path", "")
                if cloud_path and not os.path.isabs(cloud_path):
                    cloud_path = os.path.join(os.path.dirname(path), cloud_path)
                if not cloud_path or not os.path.exists(cloud_path):
                    raise FileNotFoundError(f"Point cloud path not found: {cloud_path}")
                self._load_mesh_from_path(cloud_path)
            else:
                dims = mesh_info.get("dims", [1.0, 0.5, 0.5])
                lx, ly, lz = [float(v) for v in dims[:3]]
                self._set_mesh(create_cuboid_mesh(lx, ly, lz),
                               f"Cuboid {lx:.2f} x {ly:.2f} x {lz:.2f}")
                self._mesh_source = {"type": "cuboid", "dims": [lx, ly, lz]}

            obj = state.get("object", {})
            self.object_translation = np.array(obj.get("translation", [0.0, 0.0, 0.0]), dtype=float)
            self.object_rotation = np.array(obj.get("rotation", [0.0, 0.0, 0.0]), dtype=float)
            self.object_scale = float(obj.get("scale", 1.0))
            self._apply_object_transform()
            self._update_object_panel()

            cams = state.get("cameras", [])
            if len(cams) > 0:
                self.cameras = [self._camera_from_dict(c) for c in cams]
            else:
                self.cameras = [CameraModel()]

            colors = state.get("camera_colors", [])
            self.camera_colors = [np.array(c, dtype=float) for c in colors[:len(self.cameras)]]
            self._ensure_camera_color_count()

            self.active_camera_idx = int(state.get("active_camera_idx", 0))
            self.active_camera_idx = max(0, min(self.active_camera_idx, len(self.cameras) - 1))
            self._n_samples.int_value = int(state.get("sample_points", self._n_samples.int_value))
            self.camera_marker_radius = float(state.get("camera_marker_radius", self.camera_marker_radius))

            self._last_visible_all = None
            self._last_visible_cam = None
            self._last_camera_coverages = []
            self._remove(self._COVERAGE)
            self._remove(self._CAMCOVERAGE)

            self._set_selection("camera", self.active_camera_idx)
            self._update_camera_panel()
            self._update_camera_visuals()
            self._current_project_path = path
            meta = None
            project_dir = os.path.dirname(path)
            if os.path.basename(path).lower() == "project.json":
                meta = self._read_project_metadata(project_dir)
            self._current_project_name = (meta or {}).get("name", os.path.splitext(os.path.basename(path))[0])
            if show_status:
                self._set_status(f"Project loaded: {self._current_project_name}")
        finally:
            self._suspend_autosave = False

    def _autosave_now(self, show_status: bool = False):
        if self._suspend_autosave:
            return
        try:
            self._save_project_to_path(self._autosave_path, show_status=show_status)
        except Exception:
            pass

    def _reset_project_state(self):
        self._suspend_autosave = True
        try:
            self.object_translation = np.array([0.0, 0.0, 0.0], dtype=float)
            self.object_rotation    = np.array([0.0, 0.0, 0.0], dtype=float)
            self.object_scale       = 1.0
            self._last_visible_all  = None
            self._last_visible_cam  = None
            self._last_camera_coverages = []
            self._remove(self._COVERAGE)
            self._remove(self._CAMCOVERAGE)
            self._remove(self._MESH)
            self._remove(self._POINTS)
            self.base_mesh = None
            self.base_points = None
            self.point_cloud = None
            self.mesh = None
            self.cameras            = [CameraModel()]
            self.camera_colors      = []
            self.camera_marker_radius = 50.0
            self._ensure_camera_color_count()
            self.active_camera_idx  = 0
            self._set_selection("camera", 0)
            self._update_camera_panel()
            self._update_object_panel()
            self._update_camera_visuals()
            self._refresh_camera_legend()
        finally:
            self._suspend_autosave = False

    def _on_new_project(self):
        dlg = gui.Dialog("Create New Project")
        layout = gui.Vert(6)
        layout.add_child(self._key_lbl("Project name"))
        self._new_project_name_input = gui.TextEdit()
        self._new_project_name_input.placeholder_text = "Enter project name"
        self._new_project_name_input.text_value = "Untitled Project"
        layout.add_child(self._new_project_name_input)

        source_row = gui.Horiz(6)
        source_row.add_child(self._key_lbl("Source"))
        self._new_project_source_combo = gui.Combobox()
        self._new_project_source_combo.add_item("Custom cuboid")
        self._new_project_source_combo.add_item("Load mesh / point cloud")
        self._new_project_source_combo.add_item("Create point cloud from video (COLMAP)")
        self._new_project_source_combo.selected_index = 0
        source_row.add_child(self._new_project_source_combo)
        layout.add_child(source_row)

        dims_row = gui.Horiz(6)
        dims_row.add_child(self._key_lbl("Cuboid L/W/H (mm)"))
        self._new_cube_lx = self._float_ne(1000.0, 72)
        self._new_cube_ly = self._float_ne(500.0, 72)
        self._new_cube_lz = self._float_ne(500.0, 72)
        dims_row.add_child(self._new_cube_lx)
        dims_row.add_child(self._new_cube_ly)
        dims_row.add_child(self._new_cube_lz)
        layout.add_child(dims_row)

        self._new_force_rebuild = gui.Checkbox("Force rebuild (delete existing scene for selected video)")
        self._new_force_rebuild.checked = False
        layout.add_child(self._new_force_rebuild)

        layout.add_child(self._key_lbl("For file/video sources, a file chooser opens after Create."))

        btn_row = gui.Horiz(6)
        btn_create = gui.Button("Create")
        btn_create.set_on_clicked(self._on_new_project_create_clicked)
        btn_cancel = gui.Button("Cancel")
        btn_cancel.set_on_clicked(lambda: self.win.close_dialog())
        btn_row.add_child(btn_create)
        btn_row.add_child(btn_cancel)
        layout.add_child(btn_row)

        layout.preferred_width = 520
        dlg.add_child(layout)
        self.win.show_dialog(dlg)

    def _on_new_project_create_clicked(self):
        name = self._new_project_name_input.text_value.strip() or "Untitled Project"
        source_idx = int(self._new_project_source_combo.selected_index)
        dims = [
            max(1e-6, float(self._new_cube_lx.double_value)),
            max(1e-6, float(self._new_cube_ly.double_value)),
            max(1e-6, float(self._new_cube_lz.double_value)),
        ]
        force_rebuild = bool(self._new_force_rebuild.checked)
        self.win.close_dialog()
        self._pending_new_project = {
            "name": name,
            "source": source_idx,
            "dims": dims,
            "force_rebuild": force_rebuild,
        }

        if source_idx == 0:
            self._create_new_project_from_cuboid(name, dims)
        elif source_idx == 1:
            dlg = gui.FileDialog(gui.FileDialog.OPEN, "Select mesh or point cloud", self.win.theme)
            dlg.add_filter(
                ".stl .obj .ply .glb .pcd .xyz .xyzn .xyzrgb .pts .txt",
                "Object files (Mesh/PointCloud/COLMAP points3D.txt)")
            dlg.add_filter("", "All files")
            dlg.set_on_cancel(lambda: self.win.close_dialog())
            dlg.set_on_done(self._new_project_mesh_selected)
            self.win.show_dialog(dlg)
        else:
            dlg = gui.FileDialog(gui.FileDialog.OPEN, "Select video for COLMAP", self.win.theme)
            dlg.add_filter(
                ".mp4 .mov .avi .mkv .wmv .m4v",
                "Video files")
            dlg.add_filter("", "All files")
            dlg.set_on_cancel(lambda: self.win.close_dialog())
            dlg.set_on_done(self._new_project_video_selected)
            self.win.show_dialog(dlg)

    def _create_new_project_from_cuboid(self, name: str, dims: list[float]):
        self._reset_project_state()
        lx, ly, lz = dims
        self._set_mesh(create_cuboid_mesh(lx, ly, lz),
                       f"Cuboid {lx:.2f} x {ly:.2f} x {lz:.2f}")
        self._mesh_source = {"type": "cuboid", "dims": [float(lx), float(ly), float(lz)]}
        self._current_project_name = name
        self._save_project_named(name, show_status=False)
        self._set_status(f"Project created: {name}")
        self._pending_new_project = None
        self._autosave_now(show_status=False)

    def _new_project_mesh_selected(self, path: str):
        self.win.close_dialog()
        try:
            if not self._pending_new_project:
                return
            name = self._pending_new_project["name"]
            self._reset_project_state()
            self._load_mesh_from_path(path)
            self._current_project_name = name
            self._save_project_named(name, show_status=False)
            self._set_status(f"Project created from file: {name}")
            self._pending_new_project = None
        except Exception as e:
            self._set_status(f"Create project failed: {e}")

    def _new_project_video_selected(self, path: str):
        self.win.close_dialog()
        try:
            if not self._pending_new_project:
                return
            name = self._pending_new_project["name"]
            self._reset_project_state()
            self._create_default_cuboid()
            self._current_project_name = name
            force_rebuild = bool(self._pending_new_project.get("force_rebuild", False))
            self._start_colmap_reconstruction(name, path, force_rebuild=force_rebuild)
            self._pending_new_project = None
        except Exception as e:
            self._set_status(f"Create project failed: {e}")
        self._autosave_now(show_status=False)

    def _on_save_project(self):
        dlg = gui.Dialog("Save Project")
        layout = gui.Vert(6)
        layout.add_child(self._key_lbl("Project name"))
        self._project_name_input = gui.TextEdit()
        self._project_name_input.placeholder_text = "Enter project name"
        self._project_name_input.text_value = self._current_project_name or "Untitled Project"
        layout.add_child(self._project_name_input)
        layout.add_child(self._key_lbl(f"Saved under: {self._safe_relpath(self._projects_root)}"))

        btn_row = gui.Horiz(6)
        btn_save = gui.Button("Save")
        btn_save.set_on_clicked(self._save_project_name_selected)
        btn_cancel = gui.Button("Cancel")
        btn_cancel.set_on_clicked(lambda: self.win.close_dialog())
        btn_row.add_child(btn_save)
        btn_row.add_child(btn_cancel)
        layout.add_child(btn_row)

        layout.preferred_width = 420
        dlg.add_child(layout)
        self.win.show_dialog(dlg)

    def _save_project_selected(self, path: str):
        self.win.close_dialog()
        try:
            self._save_project_to_path(path, show_status=True)
        except Exception as e:
            self._set_status(f"Save failed: {e}")

    def _save_project_name_selected(self):
        self.win.close_dialog()
        try:
            self._save_project_named(self._project_name_input.text_value, show_status=True)
        except Exception as e:
            self._set_status(f"Save failed: {e}")

    def _on_load_project(self):
        dlg = gui.Dialog("Load Project")
        layout = gui.Vert(6)
        layout.add_child(self._key_lbl(f"Projects in {self._safe_relpath(self._projects_root)}"))

        filter_row = gui.Horiz(6)
        filter_row.add_child(self._key_lbl("Search"))
        self._project_search_input = gui.TextEdit()
        self._project_search_input.placeholder_text = "Name, object, type"
        self._project_search_input.text_value = ""
        filter_row.add_child(self._project_search_input)
        layout.add_child(filter_row)

        sort_row = gui.Horiz(6)
        sort_row.add_child(self._key_lbl("Sort by"))
        self._project_sort_combo = gui.Combobox()
        for label, _mode in self._project_sort_options():
            self._project_sort_combo.add_item(label)
        self._project_sort_combo.selected_index = 0
        sort_row.add_child(self._project_sort_combo)
        btn_apply = gui.Button("Apply")
        btn_apply.set_on_clicked(self._refresh_project_browser)
        sort_row.add_child(btn_apply)
        layout.add_child(sort_row)

        self._project_list_view = gui.ListView()
        self._project_list_view.set_max_visible_items(12)
        self._project_list_view.set_on_selection_changed(self._on_project_browser_selection_changed)
        layout.add_child(self._project_list_view)

        self._project_detail_label = gui.Label("No project selected.")
        self._project_detail_label.text_color = self._ui["muted"]
        layout.add_child(self._project_detail_label)

        btn_row = gui.Horiz(6)
        btn_load = gui.Button("Load Selected")
        btn_load.set_on_clicked(self._on_project_browser_load_clicked)
        btn_close = gui.Button("Close")
        btn_close.set_on_clicked(lambda: self.win.close_dialog())
        btn_row.add_child(btn_load)
        btn_row.add_child(btn_close)
        layout.add_child(btn_row)

        layout.preferred_width = 560
        dlg.add_child(layout)
        self.win.show_dialog(dlg)
        self._refresh_project_browser()

    def _load_project_selected(self, path: str):
        self.win.close_dialog()
        try:
            self._load_project_from_path(path, show_status=True)
            self._autosave_now(show_status=False)
        except Exception as e:
            self._set_status(f"Load failed: {e}")

    def _load_managed_project_selected(self, path: str):
        self.win.close_dialog()
        try:
            self._load_project_from_path(path, show_status=True)
            self._autosave_now(show_status=False)
        except Exception as e:
            self._set_status(f"Load failed: {e}")

    def _show_reconstruction_dialog(self, project_name: str, video_path: str):
        dlg = gui.Dialog("COLMAP Reconstruction")
        layout = gui.Vert(6)
        layout.add_child(self._key_lbl(f"Project: {project_name}"))
        layout.add_child(self._key_lbl(f"Video: {os.path.basename(video_path)}"))

        self._recon_progress_label = gui.Label("Stage: Starting... (0%)")
        self._recon_progress_label.text_color = self._ui["value"]
        layout.add_child(self._recon_progress_label)

        self._recon_progress_bar = gui.ProgressBar()
        self._recon_progress_bar.value = 0.0
        layout.add_child(self._recon_progress_bar)

        self._recon_log_text = gui.TextEdit()
        self._recon_log_text.text_value = ""
        layout.add_child(self._recon_log_text)

        btn_row = gui.Horiz(6)
        btn_close = gui.Button("Close")
        btn_close.set_on_clicked(lambda: self.win.close_dialog())
        btn_row.add_child(btn_close)
        layout.add_child(btn_row)

        layout.preferred_width = 760
        dlg.add_child(layout)
        self.win.show_dialog(dlg)

    def _update_recon_progress_from_line(self, line: str):
        if not hasattr(self, "_recon_progress_label"):
            return
        text = line.lower()
        stage = None
        pct = None
        if "[1/4]" in text or "extracting frames" in text:
            stage = "Stage: 1/4 Extracting frames"
            pct = 0.15
        elif "[2/4]" in text or "feature_extractor" in text:
            stage = "Stage: 2/4 Feature extraction"
            pct = 0.35
        elif "[3/4]" in text or "sequential_matcher" in text:
            stage = "Stage: 3/4 Sequential matching"
            pct = 0.60
        elif "[4/4]" in text or "mapper" in text:
            stage = "Stage: 4/4 Sparse mapping"
            pct = 0.85
        elif "model_converter" in text:
            stage = "Stage: Exporting point cloud"
            pct = 0.95
        elif "finished" in text:
            stage = "Stage: Finished"
            pct = 1.0
        if stage:
            if hasattr(self, "_recon_progress_bar") and pct is not None:
                prev = float(self._recon_progress_bar.value)
                self._recon_progress_bar.value = max(prev, pct)
                cur_pct = int(round(float(self._recon_progress_bar.value) * 100.0))
            else:
                cur_pct = 0
            self._recon_progress_label.text = f"{stage} ({cur_pct}%)"

    def _append_recon_log(self, line: str):
        if not hasattr(self, "_recon_log_text"):
            return
        self._recon_log_lines.append(line.rstrip("\n"))
        if len(self._recon_log_lines) > 600:
            self._recon_log_lines = self._recon_log_lines[-600:]
        self._recon_log_text.text_value = "\n".join(self._recon_log_lines)
        self._update_recon_progress_from_line(line)

    def _start_colmap_reconstruction(self, project_name: str, video_path: str, force_rebuild: bool = False):
        if self._recon_running:
            self._set_status("Reconstruction already running.")
            return
        if not os.path.exists(video_path):
            self._set_status(f"Video not found: {video_path}")
            return
        if not os.path.exists(self._colmap_script_path):
            self._set_status(f"COLMAP script missing: {self._colmap_script_path}")
            return

        self._recon_running = True
        self._recon_log_lines = []
        self._show_reconstruction_dialog(project_name, video_path)
        self._set_status("COLMAP reconstruction started in background...")

        scene_name = os.path.splitext(os.path.basename(video_path))[0]
        scene_dir = os.path.join(self._colmap_scenes_dir, scene_name)

        if force_rebuild and os.path.isdir(scene_dir):
            try:
                shutil.rmtree(scene_dir)
                self._append_recon_log(f"[INFO] Force rebuild: deleted existing scene {scene_dir}")
            except Exception as e:
                self._recon_running = False
                self._set_status(f"Force rebuild failed: {e}")
                self._append_recon_log(f"[ERROR] Force rebuild failed: {e}")
                if hasattr(self, "_recon_progress_label"):
                    self._recon_progress_label.text = "Stage: Failed (0%)"
                if hasattr(self, "_recon_progress_bar"):
                    self._recon_progress_bar.value = 0.0
                return

        def _worker():
            env = os.environ.copy()
            env["BATCH_NO_PAUSE"] = "1"
            script_dir = os.path.dirname(self._colmap_script_path)
            cmd = ["cmd", "/c", self._colmap_script_path, video_path]

            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=script_dir,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                self._recon_process = proc

                if proc.stdout is not None:
                    for line in proc.stdout:
                        gui.Application.instance.post_to_main_thread(
                            self.win, lambda l=line: self._append_recon_log(l))

                return_code = proc.wait()
            except Exception as e:
                return_code = -1
                gui.Application.instance.post_to_main_thread(
                    self.win, lambda: self._append_recon_log(f"[ERROR] {e}"))

            pointcloud_path = os.path.join(self._colmap_scenes_dir, scene_name, "pointcloud.ply")

            def _finish():
                self._recon_running = False
                self._recon_process = None
                if return_code == 0 and os.path.exists(pointcloud_path):
                    self._append_recon_log(f"[INFO] Output: {pointcloud_path}")
                    self._load_mesh_from_path(pointcloud_path)
                    self._current_project_name = project_name
                    self._save_project_named(project_name, show_status=False)
                    self._set_status(f"Reconstruction complete. Project saved: {project_name}")
                    if hasattr(self, "_recon_progress_label"):
                        self._recon_progress_label.text = "Stage: Finished (100%)"
                    if hasattr(self, "_recon_progress_bar"):
                        self._recon_progress_bar.value = 1.0
                else:
                    self._set_status("Reconstruction failed. Check log dialog for details.")
                    if hasattr(self, "_recon_progress_label"):
                        self._recon_progress_label.text = "Stage: Failed"
                    if hasattr(self, "_recon_progress_bar"):
                        self._recon_progress_bar.value = max(0.0, float(self._recon_progress_bar.value))

            gui.Application.instance.post_to_main_thread(self.win, _finish)

        threading.Thread(target=_worker, daemon=True).start()

    def _read_camera_ui(self):
        self._cam().position = np.array([
            self._cx.double_value, self._cy.double_value, self._cz.double_value])
        self._cam().lookat = np.array([
            self._lx.double_value, self._ly.double_value, self._lz.double_value])
        self._cam().focal_length = max(1.0, self._fl.double_value)
        self._cam().sensor_width = max(1.0, self._sw.double_value)
        self._cam().image_width  = max(1,   self._rw.int_value)
        self._cam().image_height = max(1,   self._rh.int_value)
        self._cam().near = max(1.0, self._near.double_value)
        self._cam().far  = max(self._cam().near + 1.0, self._far.double_value)
        self.camera_marker_radius = max(1.0, self._marker_radius.double_value)

    def _on_update_camera(self):
        self._read_camera_ui()
        self._update_camera_visuals()
        self._refresh_fov_labels()
        self._set_status("Camera updated.")
        self._autosave_now(show_status=False)

    def _on_roll_camera(self, angle_deg: float):
        """Rotate selected camera around its principal axis (forward/look direction)."""
        self._read_camera_ui()
        cam = self._cam()

        fwd = cam.lookat - cam.position
        fn = np.linalg.norm(fwd)
        if fn < 1e-9:
            self._set_status("Cannot roll camera: look-at equals position.")
            return
        axis = fwd / fn

        # Ensure current up is valid and not parallel to forward.
        up = cam.up.copy()
        up -= axis * np.dot(up, axis)
        un = np.linalg.norm(up)
        if un < 1e-9:
            alt = np.array([0.0, 1.0, 0.0], dtype=float)
            if abs(np.dot(axis, alt)) > 0.99:
                alt = np.array([1.0, 0.0, 0.0], dtype=float)
            up = alt - axis * np.dot(alt, axis)
            un = np.linalg.norm(up)
        up /= max(un, 1e-9)

        # Rodrigues' rotation formula for rotating up-vector around optical axis.
        a = np.radians(angle_deg)
        c, s = np.cos(a), np.sin(a)
        up_rot = up * c + np.cross(axis, up) * s + axis * np.dot(axis, up) * (1.0 - c)
        cam.up = up_rot / max(np.linalg.norm(up_rot), 1e-9)

        self._update_camera_panel()
        self._update_camera_visuals()
        self._set_status(f"Camera rolled by {angle_deg:.0f} deg around optical axis.")
        self._autosave_now(show_status=False)

    def _on_compute_coverage(self):
        if self.mesh is None and self.point_cloud is None:
            self._set_status("No object loaded.")
            return
        self._set_chip_mode("Analyzing", "warn")
        self._set_status("Working...")
        threading.Thread(target=self._coverage_thread, daemon=True).start()

    def _on_show_overall_coverage(self):
        if self.surface_pts is None or self._last_visible_all is None:
            self._set_status("Compute coverage first.")
            return

        self._remove(self._CAMCOVERAGE)
        self._remove(self._COVERAGE)
        pcd_all = build_coverage_pointcloud(
            self.surface_pts, self._last_visible_all,
            visible_color=(0.1, 0.9, 0.2),
            hidden_color=(0.9, 0.1, 0.1))
        self.scene_widget.scene.add_geometry(
            self._COVERAGE, pcd_all, self._mat_point(4.8))
        self._set_chip_mode("Overall view", "good")
        self._set_status("Showing overall coverage (all cameras).")

    def _on_reset_coverage_view(self):
        self._remove(self._CAMCOVERAGE)
        self._remove(self._COVERAGE)

        # Re-add base object visuals to ensure original colors/material are restored.
        if self.mesh is not None:
            self._remove(self._MESH)
            self.scene_widget.scene.add_geometry(
                self._MESH, self.mesh, self._mat_lit([0.55, 0.60, 0.65]))
        elif self.point_cloud is not None:
            self._remove(self._POINTS)
            self.scene_widget.scene.add_geometry(
                self._POINTS, self.point_cloud, self._mat_point(3.5))

        self._set_chip_mode("Object view", "muted")
        self._set_status("Coverage colors reset. Showing original object.")

    def _compute_detectability_colors(self, surface_pts: np.ndarray,
                                       camera: CameraModel,
                                       defect_size_mm: float) -> np.ndarray:
        """
        Compute detectability heatmap colors for surface points.
        
        For each point: defect_px = defect_mm / mm_per_pixel(distance)
        - RED (< 3 px): undetectable
        - YELLOW (3-8 px): marginal/borderline
        - GREEN (>= 8 px): detectable/classifiable
        """
        colors = np.zeros((len(surface_pts), 3), dtype=np.float64)
        
        # Compute distances and mm_per_px for each point
        distances = np.linalg.norm(surface_pts - camera.position, axis=1)
        mm_per_px = camera.resolution_mm_per_pixel(distances)
        
        # Compute projected defect size in pixels
        defect_px = defect_size_mm / np.maximum(mm_per_px, 1e-9)
        
        # Color mapping
        red    = np.array([0.95, 0.15, 0.15], dtype=np.float64)      # undetectable
        yellow = np.array([0.95, 0.85, 0.15], dtype=np.float64)      # marginal
        green  = np.array([0.15, 0.90, 0.25], dtype=np.float64)      # detectable
        gray   = np.array([0.60, 0.60, 0.60], dtype=np.float64)      # out of view
        
        # Points out of frustum/image: gray
        in_image = camera.points_in_image(surface_pts)
        colors[~in_image] = gray
        
        # Points in image: color by detectability
        colors[in_image & (defect_px < self._defect_threshold_red_px)] = red
        colors[in_image & (defect_px >= self._defect_threshold_red_px) & 
               (defect_px < self._defect_threshold_yellow_px)] = yellow
        colors[in_image & (defect_px >= self._defect_threshold_yellow_px)] = green
        
        return colors

    def _on_refresh_detectability(self):
        """Re-render detectability overlay with current defect size."""
        if self.surface_pts is None:
            self._set_status("Compute coverage first.")
            return
        
        self._defect_size_mm = max(0.01, self._defect_size.double_value)
        
        self._remove(self._COVERAGE)
        self._remove(self._CAMCOVERAGE)
        
        try:
            # Compute detectability for selected camera
            colors_cam = self._compute_detectability_colors(
                self.surface_pts, self._cam(), self._defect_size_mm)
            
            pcd_cam = o3d.geometry.PointCloud()
            pcd_cam.points = o3d.utility.Vector3dVector(self.surface_pts)
            pcd_cam.colors = o3d.utility.Vector3dVector(colors_cam)
            
            self.scene_widget.scene.add_geometry(
                self._CAMCOVERAGE, pcd_cam, self._mat_point(6.5))
            
            # Also show all-camera detectability
            colors_all = np.ones((len(self.surface_pts), 3), dtype=np.float64) * 0.60
            best_defect_px = np.zeros(len(self.surface_pts))
            
            for cam in self.cameras:
                distances = np.linalg.norm(self.surface_pts - cam.position, axis=1)
                mm_per_px = cam.resolution_mm_per_pixel(distances)
                defect_px = self._defect_size_mm / np.maximum(mm_per_px, 1e-9)
                best_defect_px = np.maximum(best_defect_px, defect_px)
            
            in_image_any = self._cam().points_in_image(self.surface_pts)
            for cam in self.cameras:
                in_image_any |= cam.points_in_image(self.surface_pts)
            
            colors_all[~in_image_any] = [0.45, 0.45, 0.45]
            colors_all[in_image_any & (best_defect_px < self._defect_threshold_red_px)] = [0.95, 0.15, 0.15]
            colors_all[in_image_any & (best_defect_px >= self._defect_threshold_red_px) & 
                      (best_defect_px < self._defect_threshold_yellow_px)] = [0.95, 0.85, 0.15]
            colors_all[in_image_any & (best_defect_px >= self._defect_threshold_yellow_px)] = [0.15, 0.90, 0.25]
            
            pcd_all = o3d.geometry.PointCloud()
            pcd_all.points = o3d.utility.Vector3dVector(self.surface_pts)
            pcd_all.colors = o3d.utility.Vector3dVector(colors_all)
            
            self.scene_widget.scene.add_geometry(
                self._COVERAGE, pcd_all, self._mat_point(4.5))
            
            self._update_camera_visuals()
            self._set_chip_mode("Detectability view", "good")
            self._set_status(f"Detectability map updated for {self._defect_size_mm:.2f} mm defect size.")
        except Exception as e:
            self._set_status(f"Error: {e}")

    def _coverage_thread(self):
        try:
            self._read_camera_ui()
            n = max(1000, min(50000, self._n_samples.int_value))

            if self.mesh is not None:
                if self.surface_pts is None or len(self.surface_pts) != n:
                    self._post_status("Sampling surface...")
                    self.surface_pts = sample_surface_points(self.mesh, n)
                    self.vis_engine  = VisibilityEngine(self.mesh)

                self._post_status("Raycasting...")
                visible_all = np.zeros(len(self.surface_pts), dtype=bool)
                vis_cam = self.vis_engine.compute_visibility(self._cam(), self.surface_pts)

                for cam in self.cameras:
                    vis = self.vis_engine.compute_visibility(cam, self.surface_pts)
                    visible_all |= vis
                mode_note = ""
            else:
                if self.point_cloud is None:
                    raise RuntimeError("No object available for coverage.")
                self._post_status("Sampling point cloud...")
                pts = np.asarray(self.point_cloud.points, dtype=np.float32)
                self.surface_pts = sample_pointcloud_points(pts, n)
                self.vis_engine = None

                self._post_status("Projecting points (no occlusion)...")
                visible_all = np.zeros(len(self.surface_pts), dtype=bool)
                vis_cam = self._cam().points_in_image(self.surface_pts)
                for cam in self.cameras:
                    visible_all |= cam.points_in_image(self.surface_pts)
                mode_note = " [point-cloud mode: no occlusion]"

            stats = compute_statistics(self._cam(), self.surface_pts, vis_cam)
            stats["camera_visible"] = int(vis_cam.sum())
            stats["camera_coverage"] = 100.0 * stats["camera_visible"] / len(self.surface_pts) if len(self.surface_pts) > 0 else 0.0
            stats["coverage"] = 100.0 * int(visible_all.sum()) / len(self.surface_pts) if len(self.surface_pts) > 0 else 0.0
            stats["visible"] = int(visible_all.sum())
            per_camera_coverages = []
            if len(self.surface_pts) > 0:
                if self.vis_engine is not None:
                    for i, cam in enumerate(self.cameras):
                        vis_i = self.vis_engine.compute_visibility(cam, self.surface_pts)
                        n_vis_i = int(vis_i.sum())
                        per_camera_coverages.append({
                            "camera_index": i + 1,
                            "visible_points": n_vis_i,
                            "coverage_pct": 100.0 * n_vis_i / len(self.surface_pts),
                        })
                else:
                    for i, cam in enumerate(self.cameras):
                        vis_i = cam.points_in_image(self.surface_pts)
                        n_vis_i = int(vis_i.sum())
                        per_camera_coverages.append({
                            "camera_index": i + 1,
                            "visible_points": n_vis_i,
                            "coverage_pct": 100.0 * n_vis_i / len(self.surface_pts),
                        })
            self._last_visible_all = visible_all.copy()
            self._last_visible_cam = vis_cam.copy()
            self._last_camera_coverages = per_camera_coverages

            def _finish():
                self._remove(self._COVERAGE)
                self._remove(self._CAMCOVERAGE)
                cam_color = self._camera_color(self.active_camera_idx)
                pcd_all = build_coverage_pointcloud(
                    self.surface_pts, visible_all,
                    visible_color=(0.1, 0.9, 0.2),
                    hidden_color=(0.9, 0.1, 0.1))
                self.scene_widget.scene.add_geometry(
                    self._COVERAGE, pcd_all, self._mat_point(4.5))

                pcd_cam = build_coverage_pointcloud(
                    self.surface_pts, vis_cam,
                    visible_color=(float(cam_color[0]), float(cam_color[1]), float(cam_color[2])),
                    hidden_color=(0.85, 0.85, 0.85))
                self.scene_widget.scene.add_geometry(
                    self._CAMCOVERAGE, pcd_cam, self._mat_point(7.0))

                self._update_camera_visuals()
                self._show_stats(stats)
                self._set_status(
                    f"Done  {stats['coverage']:.1f}% overall  "
                    f"({stats['visible']}/{stats['total']} pts)  "
                    f"Selected camera {stats['camera_coverage']:.1f}%"
                    f"{mode_note}")

            gui.Application.instance.post_to_main_thread(self.win, _finish)

        except Exception as e:
            import traceback; traceback.print_exc()
            self._post_status(f"Error: {e}")

    # -- Thread-safe helpers ----------------------------------

    def _post_status(self, msg: str):
        gui.Application.instance.post_to_main_thread(
            self.win, lambda: self._set_status(msg))

    def _set_chip_mode(self, mode_text: str, tone: str = "muted"):
        if not hasattr(self, "_chip_mode"):
            return
        tone_map = {
            "good": gui.Color(0.40, 0.90, 0.55),
            "warn": gui.Color(0.95, 0.82, 0.30),
            "bad": gui.Color(0.93, 0.35, 0.35),
            "muted": self._ui["muted"],
        }
        self._chip_mode.text = f"Mode: {mode_text}"
        self._chip_mode.text_color = tone_map.get(tone, self._ui["muted"])

    def _set_chip_detectability(self, mmpp: float):
        if not hasattr(self, "_chip_detect"):
            return
        if mmpp <= 0.25:
            self._chip_detect.text = "Detectability: Good"
            self._chip_detect.text_color = gui.Color(0.40, 0.90, 0.55)
        elif mmpp <= 0.75:
            self._chip_detect.text = "Detectability: Marginal"
            self._chip_detect.text_color = gui.Color(0.95, 0.82, 0.30)
        else:
            self._chip_detect.text = "Detectability: Poor"
            self._chip_detect.text_color = gui.Color(0.93, 0.35, 0.35)

    def _set_chip_coverage(self, coverage_pct: float):
        if not hasattr(self, "_chip_cov"):
            return
        if coverage_pct >= 80.0:
            self._chip_cov.text = f"Coverage: Good ({coverage_pct:.1f}%)"
            self._chip_cov.text_color = gui.Color(0.40, 0.90, 0.55)
        elif coverage_pct >= 50.0:
            self._chip_cov.text = f"Coverage: Marginal ({coverage_pct:.1f}%)"
            self._chip_cov.text_color = gui.Color(0.95, 0.82, 0.30)
        else:
            self._chip_cov.text = f"Coverage: Poor ({coverage_pct:.1f}%)"
            self._chip_cov.text_color = gui.Color(0.93, 0.35, 0.35)

    def _set_status(self, msg: str):
        self._status_lbl.text = msg

    def _refresh_fov_labels(self):
        self._sv["fov_h_deg"].text = f"{np.degrees(self._cam().fov_h):.2f}"
        self._sv["fov_v_deg"].text = f"{np.degrees(self._cam().fov_v):.2f}"

    def _show_stats(self, s: dict):
        self._last_stats = dict(s)
        self._sv["coverage" ].text = f"{s['coverage']:.2f}%"
        self._sv["camera_visible"].text = f"{s['camera_visible']:,}"
        self._sv["visible"  ].text = f"{s['visible']:,}"
        self._sv["total"    ].text = f"{s['total']:,}"
        self._sv["camera_coverage"].text = f"{s['camera_coverage']:.2f}%"
        self._sv["avg_dist" ].text = f"{s['avg_dist']:.4f} mm"
        self._sv["mm_per_px"].text = f"{s['mm_per_px']:.4f}"
        self._sv["fov_h_deg"].text = f"{s['fov_h_deg']:.2f}"
        self._sv["fov_v_deg"].text = f"{s['fov_v_deg']:.2f}"

        # Quality cues for key metrics.
        cov = float(s.get("coverage", 0.0))
        cam_cov = float(s.get("camera_coverage", 0.0))
        mmpp = float(s.get("mm_per_px", 0.0))

        good = gui.Color(0.40, 0.90, 0.55)
        warn = gui.Color(0.95, 0.82, 0.30)
        bad = gui.Color(0.93, 0.35, 0.35)
        neutral = self._ui["value"]

        self._sv["coverage"].text_color = good if cov >= 80.0 else (warn if cov >= 50.0 else bad)
        self._sv["camera_coverage"].text_color = good if cam_cov >= 80.0 else (warn if cam_cov >= 50.0 else bad)
        self._sv["mm_per_px"].text_color = good if mmpp <= 0.25 else (warn if mmpp <= 0.75 else bad)
        self._sv["visible"].text_color = neutral
        self._sv["camera_visible"].text_color = neutral
        self._sv["total"].text_color = neutral
        self._sv["avg_dist"].text_color = neutral
        self._sv["fov_h_deg"].text_color = neutral
        self._sv["fov_v_deg"].text_color = neutral

        self._set_chip_mode("Coverage view", "good")
        self._set_chip_coverage(cov)
        self._set_chip_detectability(mmpp)

    # -- Reports & Trials ------------------------------------

    def _build_report_data(self, trial_name: str = "") -> dict:
        """Assemble all configuration and coverage data for a report or trial."""
        self._read_camera_ui()
        state = self._build_project_state()

        cameras_info = []
        for i, cam in enumerate(self.cameras):
            sensor_name = self._guess_sensor_size_name(cam.sensor_width)
            cameras_info.append({
                "index": i + 1,
                "sensor_name": sensor_name,
                "position_mm": cam.position.tolist(),
                "lookat_mm": cam.lookat.tolist(),
                "focal_length_mm": float(cam.focal_length),
                "sensor_width_mm": float(cam.sensor_width),
                "sensor_height_mm": float(cam.sensor_height),
                "image_width_px": int(cam.image_width),
                "image_height_px": int(cam.image_height),
                "near_mm": float(cam.near),
                "far_mm": float(cam.far),
                "fov_h_deg": float(np.degrees(cam.fov_h)),
                "fov_v_deg": float(np.degrees(cam.fov_v)),
                "up": cam.up.tolist(),
            })

        self._read_object_ui()
        return {
            "trial_name": trial_name or "Unnamed Trial",
            "timestamp": self._timestamp_now(),
            "project_name": self._current_project_name or "Unsaved",
            "object": {
                "type": self._mesh_source.get("type", "unknown"),
                "source": self._mesh_source.get("path", "") or str(self._mesh_source.get("dims", "")),
                "translation_mm": self.object_translation.tolist(),
                "rotation_deg": self.object_rotation.tolist(),
                "scale": float(self.object_scale),
            },
            "cameras": cameras_info,
            "coverage_stats": self._last_stats or {},
            "camera_coverages": self._last_camera_coverages or [],
            "defect_size_mm": self._defect_size_mm,
            "sample_points": int(self._n_samples.int_value),
            "state": state,  # full project state for trial reload
        }

    def _cov_class(self, pct: float) -> str:
        if pct >= 80.0:
            return "good"
        if pct >= 50.0:
            return "warn"
        return "bad"

    def _mmpp_class(self, mmpp: float) -> str:
        if mmpp <= 0.25:
            return "good"
        if mmpp <= 0.75:
            return "warn"
        return "bad"

    def _generate_html_report(self, data: dict, report_images: list) -> str:
        """Generate a standalone HTML report with configuration images."""
        ts = data.get("timestamp", "")
        project = data.get("project_name", "--")
        trial = data.get("trial_name", "--")

        cameras_html = ""
        for cam in data.get("cameras", []):
            pos = cam["position_mm"]
            lat = cam["lookat_mm"]
            cameras_html += (
                f"<tr>"
                f"<td>{cam['index']}</td>"
                f"<td>{cam.get('sensor_name', '')}</td>"
                f"<td>{cam['focal_length_mm']:.1f}</td>"
                f"<td>{cam['sensor_width_mm']:.2f}</td>"
                f"<td>{cam['image_width_px']}x{cam['image_height_px']}</td>"
                f"<td>{cam['fov_h_deg']:.1f}&deg; x {cam['fov_v_deg']:.1f}&deg;</td>"
                f"<td>{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}</td>"
                f"<td>{lat[0]:.1f}, {lat[1]:.1f}, {lat[2]:.1f}</td>"
                f"<td>{cam['near_mm']:.0f} - {cam['far_mm']:.0f}</td>"
                f"</tr>"
            )

        stats = data.get("coverage_stats", {})
        obj = data.get("object", {})

        gallery_parts = []
        for item in (report_images or []):
            title = str(item.get("title", "Image"))
            src = str(item.get("src", "")).strip()
            if not src:
                continue
            gallery_parts.append(
                f"<div class='img-card'><div class='img-title'>{title}</div>"
                f"<img src=\"{src}\"/></div>")
        img_section = "".join(gallery_parts) if gallery_parts else "<p><em>No screenshots available.</em></p>"

        obj_trans = obj.get("translation_mm", [0, 0, 0])
        obj_rot   = obj.get("rotation_deg",   [0, 0, 0])

        coverage_no_stats = "<p><em>Coverage was not computed for this trial.</em></p>"
        if stats:
            coverage_table = (
                f"<table>"
                f"<tr><th>Metric</th><th>Value</th></tr>"
                f"<tr><td>Overall Coverage</td>"
                f"<td class='{self._cov_class(stats.get('coverage', 0))}'>"
                f"{stats.get('coverage', 0):.2f}%</td></tr>"
                f"<tr><td>Selected Camera Coverage</td>"
                f"<td class='{self._cov_class(stats.get('camera_coverage', 0))}'>"
                f"{stats.get('camera_coverage', 0):.2f}%</td></tr>"
                f"<tr><td>Overall Visible Points</td><td>{stats.get('visible', 0):,}</td></tr>"
                f"<tr><td>Selected Camera Visible</td><td>{stats.get('camera_visible', 0):,}</td></tr>"
                f"<tr><td>Total Sample Points</td><td>{stats.get('total', 0):,}</td></tr>"
                f"<tr><td>Avg Distance (mm)</td><td>{stats.get('avg_dist', 0):.2f}</td></tr>"
                f"<tr><td>Resolution (mm/px)</td>"
                f"<td class='{self._mmpp_class(stats.get('mm_per_px', 0))}'>"
                f"{stats.get('mm_per_px', 0):.4f}</td></tr>"
                f"<tr><td>FOV Horizontal</td><td>{stats.get('fov_h_deg', 0):.2f}&deg;</td></tr>"
                f"<tr><td>FOV Vertical</td><td>{stats.get('fov_v_deg', 0):.2f}&deg;</td></tr>"
                f"<tr><td>Defect Size (mm)</td><td>{data.get('defect_size_mm', 1.0):.2f}</td></tr>"
                f"</table>"
            )
        else:
            coverage_table = coverage_no_stats

        camera_cov_rows = ""
        for row in data.get("camera_coverages", []):
            camera_cov_rows += (
                f"<tr><td>Camera {row.get('camera_index', '--')}</td>"
                f"<td>{int(row.get('visible_points', 0)):,}</td>"
                f"<td class='{self._cov_class(float(row.get('coverage_pct', 0.0)))}'>"
                f"{float(row.get('coverage_pct', 0.0)):.2f}%</td></tr>")
        camera_cov_table = (
            "<table><tr><th>Camera</th><th>Visible Points</th><th>Coverage</th></tr>"
            f"{camera_cov_rows}</table>" if camera_cov_rows else "")

        n_cams = len(data.get("cameras", []))

        return (
            "<!DOCTYPE html>\n"
            "<html><head><meta charset=\"UTF-8\"/>\n"
            f"<title>Camera Planner Report - {trial}</title>\n"
            "<style>\n"
            "body { font-family: monospace; background: #1a1c1f; color: #d8dce6; margin: 24px; }\n"
            "h1 { color: #91CFF9; margin-bottom: 4px; }\n"
            "h2 { color: #91CFF9; border-bottom: 1px solid #333; padding-bottom: 4px; margin-top: 28px; }\n"
            "table { border-collapse: collapse; width: 100%; margin-bottom: 16px; }\n"
            "th { background: #252730; color: #c0c8d8; text-align: left; padding: 6px 10px; }\n"
            "td { padding: 5px 10px; border-bottom: 1px solid #2a2d34; }\n"
            ".img-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; margin-bottom: 12px; }\n"
            ".img-card { border: 1px solid #31343b; border-radius: 6px; background: #17191d; padding: 8px; }\n"
            ".img-card img { width: 100%; border-radius: 4px; border: 1px solid #444; }\n"
            ".img-title { color: #c0c8d8; margin-bottom: 6px; }\n"
            ".good { color: #66e699; font-weight: bold; }\n"
            ".warn { color: #f0d44a; font-weight: bold; }\n"
            ".bad  { color: #ee5555; font-weight: bold; }\n"
            ".meta { color: #888; font-size: 0.9em; }\n"
            "</style></head>\n"
            "<body>\n"
            f"<h1>Camera Coverage Report</h1>\n"
            f"<p class=\"meta\">Project: <strong>{project}</strong> &nbsp;|&nbsp; "
            f"Trial: <strong>{trial}</strong> &nbsp;|&nbsp; Generated: {ts}</p>\n"
            "<h2>Configuration Images</h2>\n"
            f"<div class='img-grid'>{img_section}</div>\n"
            "<h2>Object</h2>\n"
            "<table><tr><th>Property</th><th>Value</th></tr>\n"
            f"<tr><td>Type</td><td>{obj.get('type', '--')}</td></tr>\n"
            f"<tr><td>Source</td><td>{obj.get('source', '--')}</td></tr>\n"
            f"<tr><td>Translation (mm)</td>"
            f"<td>X={obj_trans[0]:.2f}, Y={obj_trans[1]:.2f}, Z={obj_trans[2]:.2f}</td></tr>\n"
            f"<tr><td>Rotation (deg)</td>"
            f"<td>Rx={obj_rot[0]:.2f}, Ry={obj_rot[1]:.2f}, Rz={obj_rot[2]:.2f}</td></tr>\n"
            f"<tr><td>Scale</td><td>{obj.get('scale', 1.0):.4f}</td></tr>\n"
            "</table>\n"
            f"<h2>Camera Configurations ({n_cams} camera(s))</h2>\n"
            "<table>\n"
            "<tr><th>#</th><th>Sensor</th><th>FL&nbsp;(mm)</th>"
            "<th>Sensor&nbsp;W&nbsp;(mm)</th><th>Resolution</th>"
            "<th>FOV H x V</th><th>Position (mm)</th>"
            "<th>Look-At (mm)</th><th>Near-Far (mm)</th></tr>\n"
            f"{cameras_html}\n"
            "</table>\n"
            "<h2>Coverage Statistics</h2>\n"
            f"{coverage_table}\n"
            f"{camera_cov_table}\n"
            "</body></html>"
        )

    def _report_output_dir(self, subdir: str, ts_slug: str) -> str:
        """Return path for a report/trial output directory."""
        if self._current_project_path and \
                os.path.basename(self._current_project_path).lower() == "project.json":
            base = os.path.join(os.path.dirname(self._current_project_path), subdir)
        elif self._current_project_name:
            slug = self._slugify_project_name(self._current_project_name)
            base = os.path.join(self._projects_root, slug, subdir)
        else:
            base = os.path.join(self._workspace_root, subdir)
        return os.path.join(base, ts_slug)

    def _build_report_images(self, image, out_dir: str) -> list:
        """Save multiple report images from current view and return relative references."""
        images = []
        overview_path = os.path.join(out_dir, "scene_overview.png")
        focus_path = os.path.join(out_dir, "scene_focus.png")
        try:
            o3d.io.write_image(overview_path, image)
            images.append({"title": "Scene Overview", "src": "scene_overview.png"})
        except Exception:
            pass

        try:
            arr = np.asarray(image)
            if arr.ndim == 3 and arr.shape[0] > 20 and arr.shape[1] > 20:
                h, w = arr.shape[0], arr.shape[1]
                y0, y1 = int(h * 0.2), int(h * 0.8)
                x0, x1 = int(w * 0.2), int(w * 0.8)
                crop = np.ascontiguousarray(arr[y0:y1, x0:x1, :])
                focus_img = o3d.geometry.Image(crop)
                o3d.io.write_image(focus_path, focus_img)
                images.append({"title": "Center Focus", "src": "scene_focus.png"})
        except Exception:
            pass
        return images

    def _collect_manual_report_images(self, out_dir: str, auto_names: set[str]) -> list:
        """Collect user-provided screenshots from report dir and manual_images folder."""
        items = []
        seen = set()
        search_dirs = [out_dir, os.path.join(os.path.dirname(out_dir), "manual_images")]
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for name in sorted(os.listdir(d)):
                lower = name.lower()
                if not (lower.endswith(".png") or lower.endswith(".jpg") or
                        lower.endswith(".jpeg") or lower.endswith(".webp")):
                    continue
                if name in auto_names:
                    continue
                abs_path = os.path.join(d, name)
                if not os.path.isfile(abs_path):
                    continue
                rel_src = os.path.relpath(abs_path, out_dir).replace("\\", "/")
                key = rel_src.lower()
                if key in seen:
                    continue
                seen.add(key)
                items.append({"title": f"Manual - {name}", "src": rel_src})
            return items

    def _on_export_report(self):
        """Export a report: configuration images + HTML + JSON summary."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self._report_output_dir("reports", ts)
        os.makedirs(report_dir, exist_ok=True)
        data = self._build_report_data(trial_name=f"Report {ts}")

        def _write_report_files(report_images):
            try:
                auto_names = {item.get("src", "") for item in (report_images or [])}
                report_images = list(report_images or []) + self._collect_manual_report_images(report_dir, auto_names)
                summary = {k: v for k, v in data.items() if k != "state"}
                report_json = os.path.join(report_dir, "report.json")
                with open(report_json, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                html = self._generate_html_report(data, report_images)
                report_html = os.path.join(report_dir, "report.html")
                with open(report_html, "w", encoding="utf-8") as f:
                    f.write(html)
                rel = os.path.relpath(report_dir, self._workspace_root)
                gui.Application.instance.post_to_main_thread(
                    self.win, lambda r=rel: self._set_status(f"Report saved: {r}"))
                try:
                    subprocess.Popen(["explorer", os.path.abspath(report_dir)])
                except Exception:
                    pass
            except Exception as e:
                import traceback; traceback.print_exc()
                gui.Application.instance.post_to_main_thread(
                    self.win, lambda err=str(e): self._set_status(f"Report error: {err}"))

        def _on_image(image):
            try:
                report_images = self._build_report_images(image, report_dir)
                threading.Thread(target=_write_report_files, args=(report_images,), daemon=True).start()
            except Exception as e:
                import traceback; traceback.print_exc()
                threading.Thread(target=_write_report_files, args=([],), daemon=True).start()

        try:
            self.scene_widget.scene.render_to_image(_on_image)
        except Exception:
            threading.Thread(target=_write_report_files, args=([],), daemon=True).start()
        self._set_status("Rendering screenshot for report...")

    def _on_save_trial(self):
        """Open dialog to name and save a trial (full state + screenshot + report)."""
        dlg = gui.Dialog("Save Trial")
        layout = gui.Vert(6)
        layout.add_child(self._key_lbl("Trial name"))
        self._trial_name_input = gui.TextEdit()
        ts_display = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._trial_name_input.placeholder_text = "Enter trial name"
        self._trial_name_input.text_value = f"Trial {ts_display}"
        layout.add_child(self._trial_name_input)
        note = gui.Label(
            "Saves complete state + coverage results.\n"
            "Load it later to resume this exact experiment.")
        note.text_color = self._ui["muted"]
        layout.add_child(note)
        btn_row = gui.Horiz(4)
        btn_save = gui.Button("Save")
        btn_save.set_on_clicked(self._do_save_trial)
        btn_cancel = gui.Button("Cancel")
        btn_cancel.set_on_clicked(lambda: self.win.close_dialog())
        btn_row.add_child(btn_save)
        btn_row.add_child(btn_cancel)
        layout.add_child(btn_row)
        layout.preferred_width = 440
        dlg.add_child(layout)
        self.win.show_dialog(dlg)

    def _do_save_trial(self):
        """Perform the trial save after dialog confirmation."""
        trial_name = (self._trial_name_input.text_value.strip()
                      or f"Trial {datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.win.close_dialog()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        trial_slug = self._slugify_project_name(trial_name)[:40]
        trial_folder_name = f"{ts}_{trial_slug}"
        trial_dir = self._report_output_dir("trials", trial_folder_name)
        os.makedirs(trial_dir, exist_ok=True)

        data = self._build_report_data(trial_name=trial_name)

        def _write_trial_files(report_images):
            try:
                auto_names = {item.get("src", "") for item in (report_images or [])}
                report_images = list(report_images or []) + self._collect_manual_report_images(trial_dir, auto_names)
                trial_json = os.path.join(trial_dir, "trial.json")
                with open(trial_json, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                html = self._generate_html_report(data, report_images)
                report_html = os.path.join(trial_dir, "report.html")
                with open(report_html, "w", encoding="utf-8") as f:
                    f.write(html)
                rel = os.path.relpath(trial_dir, self._workspace_root)
                tname = trial_name
                gui.Application.instance.post_to_main_thread(
                    self.win, lambda r=rel, n=tname: self._set_status(f"Trial saved: {n}  ({r})"))
            except Exception as e:
                import traceback; traceback.print_exc()
                gui.Application.instance.post_to_main_thread(
                    self.win, lambda err=str(e): self._set_status(f"Trial save error: {err}"))

        def _on_image(image):
            try:
                report_images = self._build_report_images(image, trial_dir)
                threading.Thread(target=_write_trial_files, args=(report_images,), daemon=True).start()
            except Exception as e:
                import traceback; traceback.print_exc()
                threading.Thread(target=_write_trial_files, args=([],), daemon=True).start()

        try:
            self.scene_widget.scene.render_to_image(_on_image)
        except Exception:
            threading.Thread(target=_write_trial_files, args=([],), daemon=True).start()
        self._set_status("Saving trial...")

    def _list_trials(self) -> list:
        """List all saved trials for the current project, sorted newest-first."""
        trials_dirs = []

        if self._current_project_path and \
                os.path.basename(self._current_project_path).lower() == "project.json":
            d = os.path.join(os.path.dirname(self._current_project_path), "trials")
            if os.path.isdir(d):
                trials_dirs.append(d)

        if self._current_project_name:
            slug = self._slugify_project_name(self._current_project_name)
            d2 = os.path.join(self._projects_root, slug, "trials")
            if os.path.isdir(d2) and d2 not in trials_dirs:
                trials_dirs.append(d2)

        # Fallback: workspace root trials folder
        d3 = os.path.join(self._workspace_root, "trials")
        if os.path.isdir(d3) and d3 not in trials_dirs:
            trials_dirs.append(d3)

        trials = []
        for tdir in trials_dirs:
            for entry in os.scandir(tdir):
                if not entry.is_dir():
                    continue
                trial_json = os.path.join(entry.path, "trial.json")
                if not os.path.exists(trial_json):
                    continue
                try:
                    with open(trial_json, "r", encoding="utf-8") as f:
                        trial_data = json.load(f)
                    trial_data["_trial_dir"] = entry.path
                    trial_data["_trial_json"] = trial_json
                    screenshot = os.path.join(entry.path, "scene_overview.png")
                    trial_data["_has_screenshot"] = os.path.exists(screenshot)
                    trials.append(trial_data)
                except Exception:
                    continue

        trials.sort(key=lambda t: t.get("timestamp", ""), reverse=True)
        return trials

    def _on_load_trial(self):
        """Browse saved trials and load one back into the tool."""
        trials = self._list_trials()

        dlg = gui.Dialog("Load Trial")
        layout = gui.Vert(6)

        if not trials:
            layout.add_child(self._key_lbl("No trials found for this project."))
            layout.add_child(self._key_lbl("Use 'Save Trial' to create one."))
            btn_close = gui.Button("Close")
            btn_close.set_on_clicked(lambda: self.win.close_dialog())
            layout.add_child(btn_close)
            layout.preferred_width = 420
            dlg.add_child(layout)
            self.win.show_dialog(dlg)
            return

        proj_name = self._current_project_name or "Unsaved"
        layout.add_child(self._key_lbl(f"Trials for: {proj_name}"))

        self._trial_browser_items = trials
        labels = []
        for t in trials:
            ts = t.get("timestamp", "--")[:16]
            name = t.get("trial_name", "--")
            stats = t.get("coverage_stats", {})
            cov = stats.get("coverage", None)
            cov_str = f"  cov={cov:.1f}%" if cov is not None else ""
            labels.append(f"{name}  |  {ts}{cov_str}")

        self._trial_list_view = gui.ListView()
        self._trial_list_view.set_items(labels)
        self._trial_list_view.set_max_visible_items(12)
        self._trial_list_view.set_on_selection_changed(self._on_trial_selection_changed)
        layout.add_child(self._trial_list_view)

        self._trial_detail_label = gui.Label("Select a trial to see details.")
        self._trial_detail_label.text_color = self._ui["muted"]
        layout.add_child(self._trial_detail_label)

        if trials:
            self._trial_list_view.selected_index = 0
            self._update_trial_detail(trials[0])

        btn_row = gui.Horiz(4)
        btn_load = gui.Button("Load Selected")
        btn_load.set_on_clicked(self._on_trial_load_clicked)
        btn_close = gui.Button("Close")
        btn_close.set_on_clicked(lambda: self.win.close_dialog())
        btn_row.add_child(btn_load)
        btn_row.add_child(btn_close)
        layout.add_child(btn_row)

        layout.preferred_width = 560
        dlg.add_child(layout)
        self.win.show_dialog(dlg)

    def _on_trial_selection_changed(self, _val, is_double_click):
        idx = int(getattr(self._trial_list_view, "selected_index", -1))
        if 0 <= idx < len(self._trial_browser_items):
            self._update_trial_detail(self._trial_browser_items[idx])
            if is_double_click:
                self._load_trial_from_data(self._trial_browser_items[idx])

    def _update_trial_detail(self, trial: dict):
        if not hasattr(self, "_trial_detail_label"):
            return
        stats = trial.get("coverage_stats", {})
        obj   = trial.get("object", {})
        n_cams = len(trial.get("cameras", []))
        has_screenshot = trial.get("_has_screenshot", False)
        cov = stats.get("coverage", None)
        cam_cov = stats.get("camera_coverage", None)
        mmpp = stats.get("mm_per_px", None)

        lines = [
            f"Trial: {trial.get('trial_name', '--')}",
            f"Time:  {trial.get('timestamp', '--')}",
            f"Object: {obj.get('type', '--')} - {obj.get('source', '--')}",
            f"Cameras: {n_cams}",
        ]
        if cov is not None:
            lines.append(f"Coverage: {cov:.1f}%   Camera: {cam_cov:.1f}%")
        else:
            lines.append("Coverage: not computed")
        if mmpp is not None:
            lines.append(f"Resolution: {mmpp:.4f} mm/px")
        lines.append(f"Screenshot: {'yes' if has_screenshot else 'no'}")

        self._trial_detail_label.text = "\n".join(lines)

    def _on_trial_load_clicked(self):
        idx = int(getattr(self._trial_list_view, "selected_index", -1))
        if 0 <= idx < len(self._trial_browser_items):
            self._load_trial_from_data(self._trial_browser_items[idx])
        else:
            self._set_status("No trial selected.")

    def _load_trial_from_data(self, trial_data: dict):
        """Restore full tool state from a saved trial."""
        self.win.close_dialog()
        try:
            state = trial_data.get("state", {})
            if not state:
                self._set_status("Trial has no state data.")
                return

            import tempfile
            with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
                json.dump(state, tf)
                temp_path = tf.name

            try:
                self._load_project_from_path(temp_path, show_status=False)
            finally:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

            # Restore coverage statistics display if available
            stats = trial_data.get("coverage_stats", {})
            if stats:
                self._last_stats = dict(stats)
                self._show_stats(stats)

            self._set_status(f"Trial loaded: {trial_data.get('trial_name', '--')}")
        except Exception as e:
            self._set_status(f"Load trial failed: {e}")

    def run(self):
        self.app.run()


# --------------------------------------------------------------
#  Entry point
# --------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Camera Coverage & Planning Tool")
    p.add_argument("--mesh", type=str, default=None,
                   help="Path to STL/OBJ/PLY mesh file")
    args = p.parse_args()
    gui.Application.instance  # ensure singleton
    CameraPlanner(initial_mesh_path=args.mesh).run()


if __name__ == "__main__":
    main()