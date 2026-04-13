# # import open3d as o3d
# # import numpy as np

# # # ---------------------------
# # # 1. Load GLB mesh
# # # ---------------------------
# # mesh = o3d.io.read_triangle_mesh(r"C:\Prem_Varma\Camera-Coverage-Studio\3d\Wheel Mesh.glb")
# # mesh.compute_vertex_normals()

# # # ---------------------------
# # # 2. Sample point cloud
# # # ---------------------------
# # pcd = mesh.sample_points_poisson_disk(number_of_points=80000)

# # print("Initial points:", np.asarray(pcd.points).shape[0])

# # # ---------------------------
# # # 3. Downsample (optional but recommended)
# # # ---------------------------
# # pcd = pcd.voxel_down_sample(voxel_size=0.01)

# # # ---------------------------
# # # 4. Remove statistical outliers
# # # ---------------------------
# # pcd, ind = pcd.remove_statistical_outlier(
# #     nb_neighbors=20,
# #     std_ratio=2.0
# # )

# # print("After SOR:", np.asarray(pcd.points).shape[0])

# # # ---------------------------
# # # 5. Remove plane (wall)
# # # ---------------------------
# # plane_model, inliers = pcd.segment_plane(
# #     distance_threshold=0.01,
# #     ransac_n=3,
# #     num_iterations=1000
# # )

# # pcd_no_plane = pcd.select_by_index(inliers, invert=True)

# # print("After plane removal:", np.asarray(pcd_no_plane.points).shape[0])

# # # ---------------------------
# # # 6. Cluster (DBSCAN)
# # # ---------------------------
# # labels = np.array(
# #     pcd_no_plane.cluster_dbscan(
# #         eps=0.05,
# #         min_points=20,
# #         print_progress=True
# #     )
# # )

# # # Remove noise label (-1)
# # valid = labels >= 0
# # labels = labels[valid]
# # points = np.asarray(pcd_no_plane.points)[valid]

# # # ---------------------------
# # # 7. Extract largest cluster (wheel)
# # # ---------------------------
# # largest_cluster = np.argmax(np.bincount(labels))

# # wheel_indices = np.where(labels == largest_cluster)[0]

# # wheel_pcd = o3d.geometry.PointCloud()
# # wheel_pcd.points = o3d.utility.Vector3dVector(points[wheel_indices])

# # print("Wheel points:", len(wheel_indices))

# # # ---------------------------
# # # 8. Save result
# # # ---------------------------
# # o3d.io.write_point_cloud("seat_only.pcd", wheel_pcd)

# # # ---------------------------
# # # 9. Visualize
# # # ---------------------------
# # o3d.visualization.draw_geometries([wheel_pcd])



# # import open3d as o3d
# # import numpy as np

# # # ---------------------------
# # # 1. Load GLB mesh
# # # ---------------------------
# # mesh = o3d.io.read_triangle_mesh(r"C:\Prem_Varma\Camera-Coverage-Studio\3d\Seat Top.glb")
# # mesh.compute_vertex_normals()

# # # ---------------------------
# # # 2. Sample point cloud
# # # ---------------------------
# # pcd = mesh.sample_points_poisson_disk(number_of_points=100000)
# # print("Initial points:", len(pcd.points))

# # # ---------------------------
# # # 3. Downsample
# # # ---------------------------
# # pcd = pcd.voxel_down_sample(voxel_size=0.01)

# # # ---------------------------
# # # 4. Remove statistical outliers
# # # ---------------------------
# # pcd, ind = pcd.remove_statistical_outlier(
# #     nb_neighbors=20,
# #     std_ratio=2.0
# # )
# # print("After SOR:", len(pcd.points))

# # # ---------------------------
# # # 5. Remove plane (wall)
# # # ---------------------------
# # plane_model, inliers = pcd.segment_plane(
# #     distance_threshold=0.01,
# #     ransac_n=3,
# #     num_iterations=1000
# # )

# # pcd_no_plane = pcd.select_by_index(inliers, invert=True)
# # print("After plane removal:", len(pcd_no_plane.points))

# # # ---------------------------
# # # 6. DBSCAN Clustering (robust)
# # # ---------------------------
# # labels = np.array(
# #     pcd_no_plane.cluster_dbscan(
# #         eps=0.08,        # 🔁 tune if needed
# #         min_points=10,   # 🔁 tune if needed
# #         print_progress=True
# #     )
# # )

# # print("Unique labels:", np.unique(labels))

# # # Handle case: no clusters found
# # if np.all(labels == -1):
# #     raise ValueError("❌ No clusters found. Increase eps or reduce min_points.")

# # # Keep only valid clusters
# # valid = labels >= 0
# # labels_valid = labels[valid]
# # points_valid = np.asarray(pcd_no_plane.points)[valid]

# # # ---------------------------
# # # 7. Extract largest cluster (wheel)
# # # ---------------------------
# # largest_cluster = np.argmax(np.bincount(labels_valid))

# # wheel_indices = np.where(labels_valid == largest_cluster)[0]

# # wheel_pcd = o3d.geometry.PointCloud()
# # wheel_pcd.points = o3d.utility.Vector3dVector(points_valid[wheel_indices])

# # print("Wheel points:", len(wheel_pcd.points))

# # # ---------------------------
# # # 8. Estimate normals (needed for mesh)
# # # ---------------------------
# # wheel_pcd.estimate_normals(
# #     search_param=o3d.geometry.KDTreeSearchParamHybrid(
# #         radius=0.05,
# #         max_nn=30
# #     )
# # )

# # # ---------------------------
# # # 9. Mesh reconstruction (Poisson)
# # # ---------------------------
# # mesh_out, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
# #     wheel_pcd,
# #     depth=9
# # )

# # # ---------------------------
# # # 10. Remove low-density vertices
# # # ---------------------------
# # densities = np.asarray(densities)
# # threshold = np.quantile(densities, 0.1)
# # vertices_to_remove = densities < threshold

# # mesh_out.remove_vertices_by_mask(vertices_to_remove)

# # print("Final mesh vertices:", len(mesh_out.vertices))

# # # ---------------------------
# # # 11. Save outputs
# # # ---------------------------
# # o3d.io.write_point_cloud("seat_only.pcd", wheel_pcd)
# # o3d.io.write_triangle_mesh("seat_only.glb", mesh_out)

# # print("✅ Saved: seat_only.pcd and seat_only.glb")

# # # ---------------------------
# # # 12. Visualize
# # # ---------------------------
# # o3d.visualization.draw_geometries([mesh_out])

# # import open3d as o3d
# # import numpy as np

# # # ---------------------------
# # # 1. Load GLB mesh
# # # ---------------------------
# # mesh = o3d.io.read_triangle_mesh(r"C:\Prem_Varma\Camera-Coverage-Studio\3d\Seat Top.glb")
# # mesh.compute_vertex_normals()

# # # ---------------------------
# # # 2. Sample point cloud
# # # ---------------------------
# # pcd = mesh.sample_points_poisson_disk(number_of_points=60000)
# # print("Initial points:", len(pcd.points))

# # # ---------------------------
# # # 3. Downsample
# # # ---------------------------
# # pcd = pcd.voxel_down_sample(voxel_size=0.01)

# # # ---------------------------
# # # 4. Remove statistical outliers
# # # ---------------------------
# # pcd, ind = pcd.remove_statistical_outlier(
# #     nb_neighbors=20,
# #     std_ratio=2.0
# # )
# # print("After SOR:", len(pcd.points))

# # # ---------------------------
# # # 5. Remove plane
# # # ---------------------------
# # plane_model, inliers = pcd.segment_plane(
# #     distance_threshold=0.01,
# #     ransac_n=3,
# #     num_iterations=1000
# # )

# # pcd_no_plane = pcd.select_by_index(inliers, invert=True)
# # print("After plane removal:", len(pcd_no_plane.points))

# # # ---------------------------
# # # 6. DBSCAN Clustering
# # # ---------------------------
# # labels = np.array(
# #     pcd_no_plane.cluster_dbscan(
# #         eps=0.12,
# #         min_points=10,
# #         print_progress=True
# #     )
# # )

# # if np.all(labels == -1):
# #     raise ValueError("❌ No clusters found. Increase eps.")

# # # ---------------------------
# # # 7. Smart cluster selection
# # # ---------------------------
# # points_np = np.asarray(pcd_no_plane.points)

# # valid = labels >= 0
# # labels_valid = labels[valid]

# # unique_labels, counts = np.unique(labels_valid, return_counts=True)
# # sorted_clusters = unique_labels[np.argsort(-counts)]

# # center = points_np.mean(axis=0)

# # best_cluster = None
# # best_score = -np.inf

# # for lbl in sorted_clusters:
# #     cluster_pts = points_np[labels == lbl]

# #     if len(cluster_pts) < 500:
# #         continue

# #     cluster_center = cluster_pts.mean(axis=0)
# #     dist = np.linalg.norm(cluster_center - center)

# #     score = len(cluster_pts) - 1000 * dist

# #     if score > best_score:
# #         best_score = score
# #         best_cluster = lbl

# # seat_indices = np.where(labels == best_cluster)[0]

# # seat_pcd = o3d.geometry.PointCloud()
# # seat_points = points_np[seat_indices]
# # seat_pcd.points = o3d.utility.Vector3dVector(seat_points)

# # print("Seat points:", len(seat_pcd.points))

# # # ---------------------------
# # # 7.5 Crop bounding box
# # # ---------------------------
# # bbox = seat_pcd.get_axis_aligned_bounding_box()
# # # bbox = bbox.scale(0.95, bbox.get_center())
# # seat_pcd = seat_pcd.crop(bbox)

# # # ---------------------------
# # # ✅ 8. ADD COLORS (IMPORTANT FIX)
# # # ---------------------------

# # # Option 1: Height-based coloring (BEST)
# # points = np.asarray(seat_pcd.points)
# # z_vals = points[:, 2]

# # z_min, z_max = z_vals.min(), z_vals.max()
# # z_norm = (z_vals - z_min) / (z_max - z_min + 1e-8)

# # colors = np.zeros((len(points), 3))
# # colors[:, 0] = z_norm        # Red gradient
# # colors[:, 1] = 1 - z_norm    # Green gradient
# # colors[:, 2] = 0.5           # Blue constant

# # seat_pcd.colors = o3d.utility.Vector3dVector(colors)

# # # ---------------------------
# # # 9. Estimate normals
# # # ---------------------------
# # seat_pcd.estimate_normals(
# #     search_param=o3d.geometry.KDTreeSearchParamHybrid(
# #         radius=0.05,
# #         max_nn=30
# #     )
# # )

# # # ---------------------------
# # # 10. Mesh reconstruction
# # # ---------------------------
# # mesh_out, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
# #     seat_pcd,
# #     depth=11
# # )

# # # ---------------------------
# # # 11. Remove low-density vertices
# # # ---------------------------
# # densities = np.asarray(densities)
# # threshold = np.quantile(densities, 0.1)

# # mesh_out.remove_vertices_by_mask(densities < threshold)

# # # ---------------------------
# # # 12. Save outputs
# # # ---------------------------
# # o3d.io.write_point_cloud("seat_only_colored.pcd", seat_pcd)
# # o3d.io.write_triangle_mesh("seat_only.glb", mesh_out)

# # print("✅ Saved colored point cloud + mesh")

# # # ---------------------------
# # # 13. Visualize (POINT CLOUD + MESH)
# # # ---------------------------
# # o3d.visualization.draw_geometries([seat_pcd])
# import open3d as o3d
# import numpy as np

# # ---------------------------
# # 1. Load GLB mesh
# # ---------------------------
# mesh = o3d.io.read_triangle_mesh(r"C:\Prem_Varma\Camera-Coverage-Studio\3d\Seat Bottom.glb")
# mesh.compute_vertex_normals()

# # ---------------------------
# # 2. Sample point cloud
# # ---------------------------
# # Increased points for better edge definition
# pcd = mesh.sample_points_poisson_disk(number_of_points=100000)

# # ---------------------------
# # 3. Smart Downsample
# # ---------------------------
# # Note: 0.00001 might be too small depending on your unit scale. 
# # Adjusted to a more stable value.
# pcd = pcd.voxel_down_sample(voxel_size=0.001) 

# # ---------------------------
# # 4. Remove statistical outliers (Tightened)
# # ---------------------------
# pcd, ind = pcd.remove_statistical_outlier(
#     nb_neighbors=30,
#     std_ratio=1.5  # More aggressive to clean edges
# )

# # ---------------------------
# # 5. Remove plane
# # ---------------------------
# plane_model, inliers = pcd.segment_plane(
#     distance_threshold=0.01,
#     ransac_n=3,
#     num_iterations=1000
# )
# pcd_no_plane = pcd.select_by_index(inliers, invert=True)

# # ---------------------------
# # 6. DBSCAN Clustering
# # ---------------------------
# labels = np.array(pcd_no_plane.cluster_dbscan(eps=0.04, min_points=10))

# # [Selection logic remains same but filtered for cleaner output]
# if len(labels) > 0 and np.max(labels) >= 0:
#     counts = np.bincount(labels[labels >= 0])
#     best_cluster = np.argmax(counts)
#     seat_pcd = pcd_no_plane.select_by_index(np.where(labels == best_cluster)[0])
# else:
#     seat_pcd = pcd_no_plane

# # ---------------------------
# # 7. Color Assignment (Z-Gradient)
# # ---------------------------
# points = np.asarray(seat_pcd.points)
# z_vals = points[:, 2]
# z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-8)
# colors = np.zeros((len(points), 3))
# colors[:, 0] = z_norm
# colors[:, 1] = 1 - z_norm
# colors[:, 2] = 0.5
# seat_pcd.colors = o3d.utility.Vector3dVector(colors)

# # ---------------------------
# # 8. MESH RECONSTRUCTION (Alpha Shapes)
# # ---------------------------
# # Alpha shapes are better for "flat" or "thin" objects like seat covers.
# # A smaller alpha creates a tighter fit to the points.
# alpha = 0.01  # 🔁 Adjust based on scale and desired detail. Start small for sharper edges. 
# print(f"Generating Alpha Shape mesh with alpha={alpha}...")
# mesh_out = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(seat_pcd, alpha)

# # ---------------------------
# # 9. Clean up the Mesh
# # ---------------------------
# mesh_out.compute_vertex_normals()
# # Remove any tiny disconnected components created by the alpha shape
# mesh_out = mesh_out.remove_degenerate_triangles()
# mesh_out = mesh_out.remove_duplicated_triangles()
# mesh_out = mesh_out.remove_duplicated_vertices()
# mesh_out = mesh_out.remove_non_manifold_edges()

# # ---------------------------
# # 10. Transfer colors to mesh
# # ---------------------------
# pcd_tree = o3d.geometry.KDTreeFlann(seat_pcd)
# mesh_colors = []
# for v in mesh_out.vertices:
#     _, idx, _ = pcd_tree.search_knn_vector_3d(v, 1)
#     mesh_colors.append(seat_pcd.colors[idx[0]])
# mesh_out.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

# # ---------------------------
# # 11. Save and Visualize
# # ---------------------------
# o3d.io.write_triangle_mesh("seat_only_backside_clean2.glb", mesh_out)
# print("✅ Saved! Using Alpha Shapes for sharp edges.")
# o3d.visualization.draw_geometries([mesh_out], mesh_show_back_face=True)

import open3d as o3d
import numpy as np

# ---------------------------
# PARAMETERS
# ---------------------------
radius = 0.05
vertical_len = 1.0
horizontal_len = 1.0
bend_radius = 0.15

# ---------------------------
# Vertical Pipe
# ---------------------------
vertical = o3d.geometry.TriangleMesh.create_cylinder(radius, vertical_len)
vertical.compute_vertex_normals()
vertical.translate([0, 0, vertical_len / 2])

# ---------------------------
# Horizontal Pipe
# ---------------------------
horizontal = o3d.geometry.TriangleMesh.create_cylinder(radius, horizontal_len)
horizontal.compute_vertex_normals()

R = horizontal.get_rotation_matrix_from_xyz([0, np.pi/2, 0])
horizontal.rotate(R, center=(0, 0, 0))
horizontal.translate([bend_radius, 0, vertical_len])

# ---------------------------
# Bend (Quarter Torus)
# ---------------------------
torus = o3d.geometry.TriangleMesh.create_torus(
    torus_radius=bend_radius,
    tube_radius=radius
)
torus.compute_vertex_normals()

# Keep only quarter
v = np.asarray(torus.vertices)
t = np.asarray(torus.triangles)

mask = (v[:, 0] >= 0) & (v[:, 2] >= 0)
valid = np.where(mask)[0]

tri = [tri for tri in t if all(i in valid for i in tri)]

torus.vertices = o3d.utility.Vector3dVector(v)
torus.triangles = o3d.utility.Vector3iVector(tri)
torus.remove_unreferenced_vertices()

torus.translate([0, 0, vertical_len - bend_radius])

# ---------------------------
# Hemisphere Cap
# ---------------------------
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
sphere.compute_vertex_normals()

v = np.asarray(sphere.vertices)
t = np.asarray(sphere.triangles)

# Keep only half (x positive side)
mask = v[:, 0] >= 0
valid = np.where(mask)[0]

tri = [tri for tri in t if all(i in valid for i in tri)]

sphere.vertices = o3d.utility.Vector3dVector(v)
sphere.triangles = o3d.utility.Vector3iVector(tri)
sphere.remove_unreferenced_vertices()

# Position at pipe end
sphere.translate([bend_radius + horizontal_len, 0, vertical_len])

# ---------------------------
# Combine All
# ---------------------------
pipe = vertical + torus + horizontal + sphere
pipe.compute_vertex_normals()

# ---------------------------
# Save
# ---------------------------
o3d.io.write_triangle_mesh("pipe_with_round_end.glb", pipe)

# ---------------------------
# Visualize
# ---------------------------
o3d.visualization.draw_geometries([pipe])