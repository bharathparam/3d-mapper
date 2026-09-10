"""
Surface meshing, object-of-interest isolation, and hole-filling module.

Solves the core drone problem:
Drone videos reconstruct both the intended large landmark/monument and
hundreds of distant background/sky/tree points.

This module:
1. Uses Statistical Outlier Removal (SOR) + DBSCAN spatial density clustering
   to isolate the PRIMARY intended object/structure from background clutter.
2. Centers and bounds the 3D model tightly on the primary object.
3. Performs high-density Poisson Surface Reconstruction to fill holes and create
   a continuous 3D polygon mesh.
4. Generates a 35,000+ point infilled/densified point cloud of the target structure.
5. Exports both isolated object and full scene formats (PLY + OBJ).
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
    ) -> dict:
        """Run object segmentation, meshing, and hole filling."""
        input_ply = Path(input_ply)
        if not input_ply.exists():
            raise FileNotFoundError(f"Input PLY not found: {input_ply}")

        logger.info("Starting target object segmentation & meshing on %s", input_ply)
        try:
            return self._process_with_open3d(
                input_ply, output_mesh_ply, output_mesh_obj, output_dense_ply, output_primary_ply
            )
        except Exception as e:
            logger.warning("Open3D meshing encountered an issue: %s. Falling back to SciPy.", e)
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
    ) -> dict:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(input_ply))
        pts = np.asarray(pcd.points)
        if len(pts) < 10:
            raise ValueError(f"Too few points ({len(pts)}) for surface meshing.")

        # ── Step 1: Remove statistical outliers (floating sky/horizon points) ──
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.2)
        inlier_cloud = pcd.select_by_index(ind)

        # ── Step 2: DBSCAN Spatial Clustering to isolate the PRIMARY OBJECT ───
        target_cloud = inlier_cloud
        labels = np.array(inlier_cloud.cluster_dbscan(eps=1.2, min_points=15, print_progress=False))
        if len(labels) > 0 and labels.max() >= 0:
            unique, counts = np.unique(labels[labels >= 0], return_counts=True)
            sorted_clusters = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
            dominant_cluster_id = sorted_clusters[0][0]
            dominant_cluster_count = sorted_clusters[0][1]

            # If the dominant cluster contains significant points, treat it as the main object
            if dominant_cluster_count >= 50 or dominant_cluster_count > 0.3 * len(pts):
                main_indices = np.where(labels == dominant_cluster_id)[0]
                target_cloud = inlier_cloud.select_by_index(main_indices)
                logger.info("Isolated primary object: %d points (from %d total)",
                            len(target_cloud.points), len(pts))

        # ── Step 3: Center the Target Object Coordinate Frame ───────────────
        target_pts = np.asarray(target_cloud.points)
        center = target_pts.mean(axis=0)
        target_cloud_centered = o3d.geometry.PointCloud(target_cloud)
        target_cloud_centered.translate(-center)

        # ── Step 4: Estimate Normals on Target Object ─────────────────────────
        target_cloud_centered.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.4, max_nn=30)
        )
        target_cloud_centered.orient_normals_consistent_tangent_plane(k=15)

        # ── Step 5: Poisson Surface Reconstruction (Watertight Solid Model) ──
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            target_cloud_centered, depth=self.depth, scale=1.1, linear_fit=True
        )

        # Trim low-density boundary artifacts
        densities = np.asarray(densities)
        if len(densities) > 0:
            density_threshold = np.quantile(densities, 0.08)
            mesh.remove_vertices_by_mask(densities < density_threshold)

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        # Interpolate accurate RGB colors onto the mesh faces
        if target_cloud_centered.has_colors():
            mesh_pcd = o3d.geometry.PointCloud()
            mesh_pcd.points = mesh.vertices
            kd_tree = o3d.geometry.KDTreeFlann(target_cloud_centered)
            colors = []
            pcd_colors = np.asarray(target_cloud_centered.colors)
            for v in np.asarray(mesh.vertices):
                [_, idx, _] = kd_tree.search_knn_vector_3d(v, 3)
                if len(idx) > 0:
                    colors.append(np.mean(pcd_colors[idx], axis=0))
                else:
                    colors.append([0.8, 0.8, 0.8])
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(colors))

        # ── Step 6: Dense Surface Infilled Point Cloud ────────────────────────
        dense_points_count = max(len(target_pts) * 8, 35000)
        dense_pcd = mesh.sample_points_uniformly(number_of_points=dense_points_count)

        # ── Step 7: Export all Artifacts ─────────────────────────────────────
        output_mesh_ply.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(output_mesh_ply), mesh, write_ascii=False)
        o3d.io.write_triangle_mesh(str(output_mesh_obj), mesh)
        o3d.io.write_point_cloud(str(output_dense_ply), dense_pcd, write_ascii=False)

        if output_primary_ply:
            o3d.io.write_point_cloud(str(output_primary_ply), target_cloud_centered, write_ascii=False)

        stats = {
            "total_scene_points": len(pts),
            "primary_object_points": len(target_cloud.points),
            "dense_infilled_points": len(dense_pcd.points),
            "mesh_vertices": len(mesh.vertices),
            "mesh_triangles": len(mesh.triangles),
            "object_center": center.tolist(),
            "object_bounds_min": target_pts.min(axis=0).tolist(),
            "object_bounds_max": target_pts.max(axis=0).tolist(),
            "mesher": "poisson_object_focused",
        }
        logger.info("Object meshing complete: %d triangles, %d infilled points",
                    stats["mesh_triangles"], stats["dense_infilled_points"])
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
        hull = trimesh.convex.convex_hull(pts)
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
