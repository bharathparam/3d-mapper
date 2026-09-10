"""
Surface meshing, object-of-interest isolation, and density-centric alignment module.

Solves the core drone reconstruction challenges:
1. Distant background/sky/tree points distort bounding boxes and de-center 3D models.
2. COLMAP outputs coordinate frames with arbitrary diagonal tilts relative to the ground.
3. Centering must be anchored to the PEAK SPATIAL DENSITY of the primary landmark.

This module:
1. Calculates the Peak Spatial Density center (mode/medoid of 3D point cloud).
2. Performs RANSAC plane detection & PCA to align the ground plane horizontally (Y-up).
3. Translates the entire scene so the densest point is precisely at (0, 0, 0).
4. Isolates the primary structure from background clutter.
5. Runs high-resolution Poisson surface meshing & dense surface infilling.
6. Re-centers all artifacts (sparse, dense, mesh, confidence) around the densest core.
"""
from __future__ import annotations

import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class SurfaceMesher:
    """Isolates target objects and creates watertight 3D meshes & dense infilled clouds."""

    def __init__(self, depth: int = 9, point_weight: float = 2.0) -> None:
        self.depth = depth
        self.point_weight = point_weight

    def process(
        self,
        input_ply: Path,
        output_mesh_ply: Path,
        output_mesh_obj: Path,
        output_dense_ply: Path,
        output_primary_ply: Path | None = None,
        output_centered_sparse_ply: Path | None = None,
    ) -> dict:
        """Run density-centric alignment, segmentation, meshing, and hole filling."""
        input_ply = Path(input_ply)
        if not input_ply.exists():
            raise FileNotFoundError(f"Input PLY not found: {input_ply}")

        logger.info("Starting density-centric alignment & meshing on %s", input_ply)
        try:
            return self._process_with_open3d(
                input_ply,
                output_mesh_ply,
                output_mesh_obj,
                output_dense_ply,
                output_primary_ply,
                output_centered_sparse_ply,
            )
        except Exception as e:
            logger.warning("Open3D processing encountered an issue: %s. Falling back to SciPy.", e)
            return self._process_with_scipy(
                input_ply, output_mesh_ply, output_mesh_obj, output_dense_ply
            )

    def _process_with_open3d(
        self,
        input_ply: Path,
        output_mesh_ply: Path,
        output_mesh_obj: Path,
        output_dense_ply: Path,
        output_primary_ply: Path | None = None,
        output_centered_sparse_ply: Path | None = None,
    ) -> dict:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(input_ply))
        pts = np.asarray(pcd.points)
        if len(pts) < 10:
            raise ValueError(f"Too few points ({len(pts)}) for surface meshing.")

        # ── Step 1: Remove extreme statistical outliers ──────────────────────
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.4)
        inliers = pcd.select_by_index(ind)
        inlier_pts = np.asarray(inliers.points)

        # ── Step 2: Compute Exact Peak Spatial Density Center ────────────────
        pcd_tree = o3d.geometry.KDTreeFlann(inliers)
        k = min(30, max(5, len(inlier_pts) // 20))
        densities = []
        for i in range(len(inlier_pts)):
            [k_found, idx, d] = pcd_tree.search_knn_vector_3d(inlier_pts[i], k)
            mean_dist = np.mean(np.sqrt(d[1:])) if k_found > 1 else 1.0
            densities.append(1.0 / (mean_dist + 1e-6))

        densities = np.array(densities)
        # Take top 8% densest points as the core landmark center
        top_densest_count = max(10, int(len(inlier_pts) * 0.08))
        top_idx = np.argsort(densities)[-top_densest_count:]
        density_center = inlier_pts[top_idx].mean(axis=0)
        logger.info("Found peak spatial density center: %s", density_center)

        # ── Step 3: Upright & Ground-Plane Alignment (RANSAC Plane) ──────────
        R = np.eye(3)
        try:
            plane_model, plane_inliers = inliers.segment_plane(
                distance_threshold=0.3, ransac_n=3, num_iterations=1000
            )
            [a, b, c, d] = plane_model
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal)

            target_up = np.array([0.0, 1.0, 0.0])
            if normal[1] < 0:
                normal = -normal

            v = np.cross(normal, target_up)
            s = np.linalg.norm(v)
            c_val = np.dot(normal, target_up)
            if s > 1e-5:
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                R = np.eye(3) + vx + np.matmul(vx, vx) * ((1 - c_val) / (s ** 2))
        except Exception as e:
            logger.warning("Plane alignment warning: %s", e)

        # ── Step 4: Apply Density-Centric Alignment to Full Cloud ────────────
        pcd_aligned = o3d.geometry.PointCloud(pcd)
        pcd_aligned.translate(-density_center)
        pcd_aligned.rotate(R, center=(0, 0, 0))

        inliers_aligned = o3d.geometry.PointCloud(inliers)
        inliers_aligned.translate(-density_center)
        inliers_aligned.rotate(R, center=(0, 0, 0))

        # ── Step 5: DBSCAN Clustering for Primary Landmark Object ────────────
        labels = np.array(inliers_aligned.cluster_dbscan(eps=1.2, min_points=15, print_progress=False))
        if len(labels) > 0 and labels.max() >= 0:
            unique, counts = np.unique(labels[labels >= 0], return_counts=True)
            sorted_clusters = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
            dominant_id = sorted_clusters[0][0]
            main_indices = np.where(labels == dominant_id)[0]
            primary_cloud = inliers_aligned.select_by_index(main_indices)
        else:
            primary_cloud = inliers_aligned

        # ── Step 6: High-Fidelity Surface Meshing (Poisson -> Alpha/BPA Fallback) ──
        reg_cloud = primary_cloud.voxel_down_sample(voxel_size=0.08)
        reg_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.4, max_nn=30)
        )
        reg_cloud.orient_normals_consistent_tangent_plane(k=15)

        mesh = None
        mesher_used = "poisson_density_centered"
        try:
            p_mesh, mesh_densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                reg_cloud, depth=self.depth, scale=1.1, linear_fit=True
            )
            mesh_densities = np.asarray(mesh_densities)
            if len(mesh_densities) > 0:
                p_mesh.remove_vertices_by_mask(mesh_densities < np.quantile(mesh_densities, 0.08))
            p_mesh.remove_degenerate_triangles()
            p_mesh.remove_duplicated_triangles()
            p_mesh.remove_duplicated_vertices()
            p_mesh.remove_non_manifold_edges()
            if len(p_mesh.triangles) > 100:
                mesh = p_mesh
        except Exception as e:
            logger.warning("Poisson meshing fallback due to: %s", e)

        if mesh is None or len(mesh.triangles) == 0:
            distances = primary_cloud.compute_nearest_neighbor_distance()
            avg_dist = float(np.mean(distances)) if len(distances) > 0 else 0.05
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(primary_cloud, alpha=avg_dist * 4.5)
            mesher_used = "alpha_shape_density_centered"
            if len(mesh.triangles) == 0:
                radii = [avg_dist * 1.5, avg_dist * 3.0, avg_dist * 6.0]
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(primary_cloud, o3d.utility.DoubleVector(radii))
                mesher_used = "bpa_density_centered"

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()

        # Vertex coloring
        if primary_cloud.has_colors():
            mesh_pcd = o3d.geometry.PointCloud()
            mesh_pcd.points = mesh.vertices
            kd_tree = o3d.geometry.KDTreeFlann(primary_cloud)
            colors = []
            pcd_colors = np.asarray(primary_cloud.colors)
            for v in np.asarray(mesh.vertices):
                [_, idx, _] = kd_tree.search_knn_vector_3d(v, 3)
                if len(idx) > 0:
                    colors.append(np.mean(pcd_colors[idx], axis=0))
                else:
                    colors.append([0.8, 0.8, 0.8])
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(colors))

        # ── Step 7: Dense Infilled Points ────────────────────────────────────
        dense_points_count = max(len(primary_cloud.points) * 6, 35000)
        dense_pcd = mesh.sample_points_uniformly(number_of_points=dense_points_count)

        # ── Step 8: Save Aligned Artifacts ───────────────────────────────────
        output_mesh_ply.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(output_mesh_ply), mesh, write_ascii=False)
        o3d.io.write_triangle_mesh(str(output_mesh_obj), mesh)
        o3d.io.write_point_cloud(str(output_dense_ply), dense_pcd, write_ascii=False)

        if output_primary_ply:
            o3d.io.write_point_cloud(str(output_primary_ply), primary_cloud, write_ascii=False)

        # Update sparse.ply with the density-centered & upright aligned cloud
        if output_centered_sparse_ply:
            o3d.io.write_point_cloud(str(output_centered_sparse_ply), pcd_aligned, write_ascii=False)

        primary_pts = np.asarray(primary_cloud.points)
        dists = np.linalg.norm(primary_pts, axis=1)

        stats = {
            "total_scene_points": len(pts),
            "primary_object_points": len(primary_cloud.points),
            "dense_infilled_points": len(dense_pcd.points),
            "mesh_vertices": len(mesh.vertices),
            "mesh_triangles": len(mesh.triangles),
            "density_center_raw": density_center.tolist(),
            "target_radius_95pct": float(np.percentile(dists, 95)) if len(dists) > 0 else 5.0,
            "mesher": "poisson_density_centered",
        }
        logger.info("Density-centric meshing complete: %d points, radius %.1fm",
                    stats["primary_object_points"], stats["target_radius_95pct"])
        return stats

    def _process_with_scipy(
        self,
        input_ply: Path,
        output_mesh_ply: Path,
        output_mesh_obj: Path,
        output_dense_ply: Path,
    ) -> dict:
        import trimesh
        mesh_tri = trimesh.load(str(input_ply))
        pts = mesh_tri.vertices
        med = np.median(pts, axis=0)
        mesh_tri.vertices = pts - med
        hull = trimesh.convex.convex_hull(mesh_tri.vertices)
        output_mesh_ply.parent.mkdir(parents=True, exist_ok=True)
        hull.export(str(output_mesh_ply))
        hull.export(str(output_mesh_obj))
        dense_pts, _ = trimesh.sample.sample_surface(hull, count=max(len(pts) * 4, 15000))
        dense_cloud = trimesh.PointCloud(dense_pts)
        dense_cloud.export(str(output_dense_ply))
        return {
            "total_scene_points": len(pts),
            "primary_object_points": len(pts),
            "dense_infilled_points": len(dense_pts),
            "mesh_vertices": len(hull.vertices),
            "mesh_triangles": len(hull.faces),
            "mesher": "scipy_fallback",
        }
