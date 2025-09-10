"""
Core SAM + FoundationPose integration (no UI/visualization).

- Automatic zero-shot segmentation via Ultralytics SAM 2.1
- Multiple objects supported
- Direct mask-to-pose estimation with FoundationPose
- Simple state management (registration -> tracking)
- Multi-threaded workers for per-object registration/tracking

Usage (example):

    import trimesh
    from sam_foundation_pose import SAMFoundationPoseSystem

    mesh_list = [trimesh.load(p) for p in mesh_paths]
    system = SAMFoundationPoseSystem(sam_weights="sam2.1_b.pt", max_workers=4)

    poses = system.process_frame(color_image, depth_image, camera_matrix, mesh_list)

Notes:
- This module intentionally avoids any visualization or UI.
- Ensure the Ultralytics SAM weights are available locally or downloadable by Ultralytics.
- FoundationPose dependencies (nvdiffrast, torch, etc.) must be installed.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import trimesh

# External: Ultralytics SAM
from ultralytics import SAM

# FoundationPose imports
from Utils import *  # provides logging, dr (nvdiffrast), etc.
from estimater import FoundationPose


class SAMFoundationPoseSystem:
    """Integrates Ultralytics SAM with FoundationPose for zero-shot multi-object 6D pose.

    Threading model:
    - One worker per object for registration/tracking using ThreadPoolExecutor.
    - Each worker uses its own nvdiffrast context for safety.
    """

    def __init__(
        self,
        sam_weights: str = "sam2.1_b.pt",
        max_workers: int = 2,
        scorer=None,
        refiner=None,
        device: str = "cuda:0",
        gl_device: int = 0,
        min_mask_area: int = 200,  # filter tiny artifacts
    ) -> None:
        # SAM 2.1 initialization
        self.seg_model = SAM(sam_weights)

        # Pose related
        self.scorer = scorer
        self.refiner = refiner
        self.device = device
        self.gl_device = gl_device

        # Runtime
        self.max_workers = max_workers
        self.min_mask_area = min_mask_area

        # State: pose estimators keyed by object idx
        self.pose_estimators: Dict[int, Dict[str, object]] = {}

        # Guards
        self._lock = threading.Lock()

    # -------------------------------
    # Segmentation
    # -------------------------------
    def segment_objects_with_sam(self, color_image: np.ndarray) -> List[np.ndarray]:
        """
        Use SAM to automatically segment all objects in the image.
        Returns list of binary masks (uint8) with shape (H, W), values in {0, 255}.
        """
        H, W = color_image.shape[:2]

        # Ultralytics predict returns list[Results], grab first image's results
        results = self.seg_model.predict(color_image, verbose=False)
        if not results:
            return []
        result = results[0]

        if result.masks is None:
            return []

        detected_masks: List[np.ndarray] = []

        # Prefer polygons to create clean binary masks
        # result.masks.xy: list of (N_i, 2) arrays in image coordinates
        try:
            polys: List[np.ndarray] = result.masks.xy
        except Exception:
            polys = []

        if polys:
            for poly in polys:
                if poly is None or len(poly) < 3:
                    continue
                # Draw filled polygon into a binary mask
                mask = np.zeros((H, W), dtype=np.uint8)
                contour = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(mask, [contour], 255)
                if int(mask.sum() // 255) >= self.min_mask_area:
                    detected_masks.append(mask)
        else:
            # Fallback: rasterize SAM masks from bitmap tensor
            # result.masks.data: [n, H, W] boolean/float tensor
            try:
                data = result.masks.data.cpu().numpy()
                for i in range(data.shape[0]):
                    mask = (data[i] > 0.5).astype(np.uint8) * 255
                    if int(mask.sum() // 255) >= self.min_mask_area:
                        detected_masks.append(mask)
            except Exception:
                pass

        return detected_masks

    # -------------------------------
    # Pose estimators init
    # -------------------------------
    def initialize_pose_estimators(self, detected_masks: List[np.ndarray], mesh_list: List[trimesh.Trimesh]) -> Dict[int, Dict[str, object]]:
        """
        Initialize FoundationPose estimators for each detected mask.
        """
        pose_estimators: Dict[int, Dict[str, object]] = {}

        for idx, mask in enumerate(detected_masks):
            if idx >= len(mesh_list):
                break
            mesh = mesh_list[idx]

            # Ensure numpy arrays for points and normals
            model_pts = np.asarray(mesh.vertices, dtype=np.float32)
            model_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

            # Per-estimator GL context is safer under threads
            glctx = dr.RasterizeCudaContext(self.gl_device)

            pose_est = FoundationPose(
                model_pts=model_pts.copy(),
                model_normals=model_normals.copy(),
                symmetry_tfs=None,
                mesh=mesh,
                scorer=self.scorer,
                refiner=self.refiner,
                glctx=glctx,
                debug_dir=None,
                debug=0,
            )
            try:
                pose_est.to_device(self.device)
            except Exception:
                # Fallback to CPU if CUDA is unavailable
                pose_est.to_device("cpu")

            pose_estimators[idx] = {
                "pose_est": pose_est,
                "mask": mask,
                "mesh": mesh,
            }

        return pose_estimators

    # -------------------------------
    # Registration (multi-threaded)
    # -------------------------------
    def _register_worker(
        self,
        idx: int,
        color_image: np.ndarray,
        depth_image: Optional[np.ndarray],
        camera_matrix: np.ndarray,
        data: Dict[str, object],
        iterations: int = 4,
    ) -> Tuple[int, Optional[np.ndarray]]:
        pose_est: FoundationPose = data["pose_est"]  # type: ignore
        mask: np.ndarray = data["mask"]  # type: ignore

        try:
            pose = pose_est.register(
                K=camera_matrix,
                rgb=color_image,
                depth=depth_image,
                ob_mask=mask,
                iteration=iterations,
            )
            # Track state: mark as registered
            setattr(pose_est, "is_register", True)
            return idx, pose
        except Exception as e:
            logging.warning(f"Registration failed for idx={idx}: {e}")
            return idx, None

    def register_objects(
        self,
        color_image: np.ndarray,
        depth_image: Optional[np.ndarray],
        camera_matrix: np.ndarray,
        pose_estimators: Dict[int, Dict[str, object]],
        iterations: int = 4,
    ) -> Dict[int, np.ndarray]:
        """Register poses for all detected objects using their SAM masks."""
        results: Dict[int, np.ndarray] = {}
        if not pose_estimators:
            return results

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [
                ex.submit(self._register_worker, idx, color_image, depth_image, camera_matrix, data, iterations)
                for idx, data in pose_estimators.items()
            ]
            for fut in as_completed(futures):
                idx, pose = fut.result()
                if pose is not None:
                    results[idx] = pose
        return results

    # -------------------------------
    # Tracking (multi-threaded)
    # -------------------------------
    def _track_worker(
        self,
        idx: int,
        color_image: np.ndarray,
        depth_image: Optional[np.ndarray],
        camera_matrix: np.ndarray,
        data: Dict[str, object],
        iterations: int = 2,
    ) -> Tuple[int, Optional[np.ndarray]]:
        pose_est: FoundationPose = data["pose_est"]  # type: ignore

        try:
            # Only track if registered
            if not getattr(pose_est, "is_register", False):
                return idx, None

            pose = pose_est.track_one(
                rgb=color_image,
                depth=depth_image,
                K=camera_matrix,
                iteration=iterations,
            )
            return idx, pose
        except Exception as e:
            logging.warning(f"Tracking failed for idx={idx}: {e}")
            return idx, None

    def track_objects(
        self,
        color_image: np.ndarray,
        depth_image: Optional[np.ndarray],
        camera_matrix: np.ndarray,
        pose_estimators: Dict[int, Dict[str, object]],
        iterations: int = 2,
    ) -> Dict[int, np.ndarray]:
        """Track registered objects in subsequent frames."""
        estimated_poses: Dict[int, np.ndarray] = {}
        if not pose_estimators:
            return estimated_poses

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [
                ex.submit(self._track_worker, idx, color_image, depth_image, camera_matrix, data, iterations)
                for idx, data in pose_estimators.items()
            ]
            for fut in as_completed(futures):
                idx, pose = fut.result()
                if pose is not None:
                    estimated_poses[idx] = pose
        return estimated_poses

    # -------------------------------
    # Main per-frame pipeline
    # -------------------------------
    def process_frame(
        self,
        color_image: np.ndarray,
        depth_image: Optional[np.ndarray],
        camera_matrix: np.ndarray,
        mesh_list: List[trimesh.Trimesh],
    ) -> Dict[int, np.ndarray]:
        """
        Main processing function that integrates SAM with FoundationPose.
        Returns a dict of {idx: 4x4 pose} for the current frame.
        """
        # Step 1: Use SAM for automatic segmentation
        detected_masks = self.segment_objects_with_sam(color_image)
        if not detected_masks:
            return {}

        # Step 2: Initialize FoundationPose estimators using SAM masks (only once)
        if not self.pose_estimators:
            with self._lock:
                if not self.pose_estimators:  # double-checked locking
                    self.pose_estimators = self.initialize_pose_estimators(detected_masks, mesh_list)
            # Step 3: Register poses using SAM masks
            registered = self.register_objects(color_image, depth_image, camera_matrix, self.pose_estimators)
            return registered
        else:
            # Step 4: Track objects in subsequent frames
            estimated_poses = self.track_objects(color_image, depth_image, camera_matrix, self.pose_estimators)
            return estimated_poses


__all__ = [
    "SAMFoundationPoseSystem",
]
