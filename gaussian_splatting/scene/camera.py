"""
Pinhole Camera Model / 针孔相机模型
=====================================
For single-view sports video, we use a standard pinhole camera.
Since the video is captured from a fixed viewpoint, all frames
share the same intrinsic parameters.

针对单视角体育视频，使用标准针孔相机模型。
由于视频从固定视角拍摄，所有帧共享相同的内参。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CameraParams:
    """Intrinsic & extrinsic camera parameters / 相机内外参数"""

    width: int          # Image width / 图像宽度
    height: int         # Image height / 图像高度
    fx: float           # Focal length x (pixels) / x方向焦距
    fy: float           # Focal length y (pixels) / y方向焦距
    cx: float           # Principal point x / 主点 x
    cy: float           # Principal point y / 主点 y
    # Extrinsics (world-to-camera transform)
    # 外参（世界坐标系到相机坐标系的变换）
    R: Optional[torch.Tensor] = None   # Rotation matrix (3x3) / 旋转矩阵
    T: Optional[torch.Tensor] = None   # Translation vector (3,) / 平移向量
    znear: float = 0.01                # Near clipping plane / 近裁剪面
    zfar: float = 100.0                # Far clipping plane / 远裁剪面


class PinholeCamera(nn.Module):
    """
    Standard pinhole camera model for 3DGS rendering.
    用于 3DGS 渲染的标准针孔相机模型。

    Constructs the world-to-camera matrix, projection matrix,
    and full view-projection matrix used during rasterization.

    Attributes / 属性:
        params: Immutable camera parameters / 不可变相机参数
        world_view_transform: 4×4 world-to-camera matrix / 4×4 世界到相机矩阵
        projection_matrix: 4×4 OpenGL-style projection / 4×4 OpenGL 风格投影矩阵
        full_proj_transform: 4×4 combined transform / 4×4 组合变换矩阵
        camera_center: camera position in world coords / 相机在世界坐标的位置
    """

    def __init__(self, params: CameraParams, device: str = "cuda"):
        super().__init__()
        self.params = params
        self.device = device

        # Build transform matrices / 构建变换矩阵
        self._build_world_view_transform()
        self._build_projection_matrix()
        self._build_full_proj_transform()
        self._compute_camera_center()

    def _build_world_view_transform(self) -> None:
        """
        Build the 4×4 world-to-view (view) matrix.
        构建 4×4 世界到观察（视图）矩阵。

        For a fixed camera, R=Identity, T=origin is typical.
        If single-view: we place the camera along +Z looking at origin.
        """
        if self.params.R is None or self.params.T is None:
            # Default: camera sits at (0, 0, 2.0) looking at origin
            # 默认：相机位于 (0, 0, 2.0)，看向原点
            R = torch.eye(3, device=self.device, dtype=torch.float32)
            T = torch.tensor([0.0, 0.0, 2.0], device=self.device, dtype=torch.float32)
        else:
            R = self.params.R.to(self.device, dtype=torch.float32)
            T = self.params.T.to(self.device, dtype=torch.float32)

        # Build 4×4 view matrix: [R | t] augmented to 4×4
        # 构建 4×4 视图矩阵
        w2c = torch.eye(4, device=self.device, dtype=torch.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = T

        self.world_view_transform = w2c.T.contiguous()  # GL convention / OpenGL 惯例

    def _build_projection_matrix(self) -> None:
        """
        Build OpenGL-style perspective projection matrix.
        构建 OpenGL 风格的透视投影矩阵。
        """
        p = self.params
        fov_y = 2.0 * np.arctan(p.height / (2.0 * p.fy))
        fov_y_rad = float(fov_y)
        tan_half_fov = float(np.tan(fov_y_rad / 2.0))
        aspect = p.width / p.height
        z_sign = 1.0  # OpenGL convention

        proj = torch.zeros(4, 4, device=self.device, dtype=torch.float32)
        proj[0, 0] = 1.0 / (tan_half_fov * aspect)
        proj[1, 1] = 1.0 / tan_half_fov
        proj[2, 2] = z_sign * p.zfar / (p.zfar - p.znear)
        proj[2, 3] = -(p.zfar * p.znear) / (p.zfar - p.znear)
        proj[3, 2] = z_sign

        self.projection_matrix = proj

    def _build_full_proj_transform(self) -> None:
        """Compose view × projection / 组合视图 × 投影"""
        self.full_proj_transform = (
            self.world_view_transform @ self.projection_matrix
        ).contiguous()

    def _compute_camera_center(self) -> None:
        """
        Camera center in world space = inverse(view)[:3, 3].
        相机在世界空间的位置。
        """
        w2c = self.world_view_transform.T
        R_inv = w2c[:3, :3].T
        T = w2c[:3, 3]
        self.camera_center = (-R_inv @ T).float()

    def get_calib_matrix(self) -> torch.Tensor:
        """
        Return the 3×3 intrinsic matrix K / 返回 3×3 内参矩阵 K.
        """
        p = self.params
        K = torch.zeros(3, 3, device=self.device, dtype=torch.float32)
        K[0, 0] = p.fx
        K[1, 1] = p.fy
        K[0, 2] = p.cx
        K[1, 2] = p.cy
        K[2, 2] = 1.0
        return K

    def project_points(self, points_3d: torch.Tensor) -> torch.Tensor:
        """
        Project 3D world points to 2D image coordinates.
        将 3D 世界点投影到 2D 图像坐标。

        Args:
            points_3d: (N, 3) world coordinates / 世界坐标
        Returns:
            (N, 2) pixel coordinates / 像素坐标
        """
        K = self.get_calib_matrix()
        w2c = self.world_view_transform.T  # 4×4

        # Convert to homogeneous / 转为齐次坐标
        ones = torch.ones(points_3d.shape[0], 1, device=self.device)
        pts_homo = torch.cat([points_3d, ones], dim=1)  # (N, 4)

        # Transform to camera space / 变换到相机空间
        cam_pts = (w2c @ pts_homo.T).T[:, :3]  # (N, 3)

        # Perspective division and intrinsics / 透视除法与内参
        z = cam_pts[:, 2:3].clamp(min=1e-6)
        uv_homo = (K @ (cam_pts / z).T).T  # (N, 3)
        return uv_homo[:, :2]

    def get_fov_degrees(self) -> Tuple[float, float]:
        """
        Return horizontal and vertical FOV in degrees.
        返回水平和垂直视场角（度数）。
        """
        p = self.params
        fov_y = 2.0 * float(np.arctan(p.height / (2.0 * p.fy))) * 180.0 / np.pi
        fov_x = 2.0 * float(np.arctan(p.width / (2.0 * p.fx))) * 180.0 / np.pi
        return fov_x, fov_y
