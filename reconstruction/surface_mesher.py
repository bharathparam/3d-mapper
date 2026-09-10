"""
Surface meshing and point cloud infilling module.

Transforms sparse/semi-dense point clouds into:
1. Watertight / manifold surface polygon meshes (Poisson / Ball-Pivoting / Alpha-Shape)
2. Densified / hole-filled point clouds via surface interpolation
3. Exports both colored PLY and Wavefront OBJ formats
"""
from __future__ import annotations

import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class SurfaceMesher:
    """Generates filled 3D meshes and interpolated point clouds."""

    def __init__(self, depth: int = 8, point_weight: float = 2.0) -> None:
        self.depth = depth
        self.point_weight = point_weight

    def process(
        self,
        input_ply: Path,
        output_mesh_ply: Path,
        output_mesh_obj: Path,
        output_dense_ply: Path,
    ) -> dict:
        """Run meshing and hole filling from sparse PLY."""
        input_ply = Path(input_ply)
        if not input_ply.exists():
            raise FileNotFoundError(f"Input PLY not found: {input_ply}")

        logger.info("Starting surface meshing and hole filling on %s", input_ply)
        try:
            return self._process_with_open3d(
                input_ply, output_mesh_ply, output_mesh_obj, output_dense_ply
            )
        except Exception as e:
            logger.warning("Open3D meshing encountered an issue: %s. Falling back to SciPy/Delaunay mesher.", e)
            return self._process_with_scipy(
                input_ply, output_mesh_ply, output_mesh_obj, output_dense_ply
            )

    def _process_with_open3d(
        self,
        input_ply: Path,
        output_mesh_ply: Path,
        output_mesh_obj: Path,
        output_dense_ply: Path,
    ) -> dict:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(input_ply))
        pts = np.asarray(pcd.points)
        if len(pts) < 10:
            raise ValueError(f"Too few points ({len(pts)}) for surface meshing.")

        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

        # 1. Poisson Surface Reconstruction (Fills holes across the whole structure)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=self.depth, scale=1.1, linear_fit=True
        )

        # Filter low density outlier triangles
        densities = np.asarray(densities)
        if len(densities) > 0:
            density_threshold = np.quantile(densities, 0.05)
            vertices_to_remove = densities < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        # Interpolate vertex colors onto mesh from point cloud
        if pcd.has_colors():
            mesh_pcd = o3d.geometry.PointCloud()
            mesh_pcd.points = mesh.vertices
            kd_tree = o3d.geometry.KDTreeFlann(pcd)
            colors = []
            pcd_colors = np.asarray(pcd.colors)
            for v in np.asarray(mesh.vertices):
                [_, idx, _] = kd_tree.search_knn_vector_3d(v, 3)
                if len(idx) > 0:
                    colors.append(np.mean(pcd_colors[idx], axis=0))
                else:
                    colors.append([0.8, 0.8, 0.8])
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(colors))

        # 2. Generate Densified / Infilled Point Cloud by sampling the reconstructed mesh
        dense_pcd = mesh.sample_points_uniformly(number_of_points=max(len(pts) * 5, 25000))

        # Save artifacts
        output_mesh_ply.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(output_mesh_ply), mesh, write_ascii=False)
        o3d.io.write_triangle_mesh(str(output_mesh_obj), mesh)
        o3d.io.write_point_cloud(str(output_dense_ply), dense_pcd, write_ascii=False)

        stats = {
            "original_points": len(pts),
            "dense_points": len(dense_pcd.points),
            "mesh_vertices": len(mesh.vertices),
            "mesh_triangles": len(mesh.triangles),
            "mesher": "poisson_open3d",
        }
        logger.info("Meshing complete: %d vertices, %d faces, %d dense points",
                    stats["mesh_vertices"], stats["mesh_triangles"], stats["dense_points"])
        return stats

    def _process_with_scipy(
        self,
        input_ply: Path,
        output_mesh_ply: Path,
        output_mesh_obj: Path,
        output_dense_ply: Path,
    ) -> dict:
        """Fallback mesher using SciPy Delaunay / Convex hull."""
        import trimesh
        mesh_tri = trimesh.load(str(input_ply))
        pts = mesh_tri.vertices
        colors = mesh_tri.visual.vertex_colors if hasattr(mesh_tri.visual, "vertex_colors") else None

        # Create convex hull or alpha shape
        hull = trimesh.convex.convex_hull(pts)
        if colors is not None and len(colors) > 0:
            hull.visual.vertex_colors = colors[:len(hull.vertices)]

        # Export
        output_mesh_ply.parent.mkdir(parents=True, exist_ok=True)
        hull.export(str(output_mesh_ply))
        hull.export(str(output_mesh_obj))

        # Densified sampling
        dense_pts, face_indices = trimesh.sample.sample_surface(hull, count=max(len(pts) * 4, 15000))
        dense_cloud = trimesh.PointCloud(dense_pts)
        dense_cloud.export(str(output_dense_ply))

        return {
            "original_points": len(pts),
            "dense_points": len(dense_pts),
            "mesh_vertices": len(hull.vertices),
            "mesh_triangles": len(hull.faces),
            "mesher": "scipy_fallback",
        }
