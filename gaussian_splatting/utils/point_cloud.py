"""
Point Cloud Utilities / 点云工具函数
======================================
Unproject depth maps to 3D point clouds and fuse multiple views.
将深度图反投影为 3D 点云并融合多视角。
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Optional, Tuple
from pathlib import Path


def depth_to_point_cloud(
    depth: torch.Tensor,
    color: Optional[torch.Tensor] = None,
    fx: float = 500.0,
    fy: float = 500.0,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    max_depth: float = 80.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Unproject a depth map to a 3D point cloud.
    将深度图反投影为三维点云。

    Given depth z at pixel (u,v), the 3D point is:
    给定像素 (u,v) 处的深度 z，3D 点为:
      X = (u - cx) * z / fx
      Y = (v - cy) * z / fy
      Z = z

    Args:
        depth: (H, W) depth map in meters / 深度图（米）
        color: (3, H, W) optional RGB image / 可选的 RGB 图像
        fx, fy: Focal lengths / 焦距
        cx, cy: Principal point (default: image center) / 主点（默认：图像中心）
        max_depth: Maximum valid depth / 最大有效深度
    Returns:
        points: (N, 3) 3D coordinates / 三维坐标
        colors: (N, 3) RGB colors (None if no color provided) / RGB 颜色
    """
    H, W = depth.shape

    if cx is None:
        cx = W / 2.0
    if cy is None:
        cy = H / 2.0

    # Create pixel grid / 创建像素网格
    v, u = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=depth.device),
        torch.arange(W, dtype=torch.float32, device=depth.device),
        indexing="ij",
    )

    # Valid depth mask / 有效深度掩码
    valid = (depth > 0.01) & (depth < max_depth)

    z = depth[valid].float()
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy  # Y down in image coords / 图像坐标中 Y 向下

    # X right, Y down, Z forward / X 向右，Y 向下，Z 向前
    points = torch.stack([x, -y, z], dim=1)  # Flip Y for world-up / 翻转 Y 使世界上方为正

    if color is not None:
        colors = color.permute(1, 2, 0)[valid].float()  # (N, 3)
        return points, colors

    return points, None


def fuse_point_clouds(
    points_list: list,
    colors_list: list,
    voxel_size: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse multiple point clouds via voxel downsampling.
    通过体素下采样融合多个点云。

    Uses Open3D for efficient voxel grid filtering.
    使用 Open3D 进行高效的体素网格滤波。

    Args:
        points_list: List of (N_i, 3) point arrays / 点数组列表
        colors_list: List of (N_i, 3) color arrays / 颜色数组列表
        voxel_size: Voxel size for downsampling / 下采样体素大小
    Returns:
        Fused points and colors / 融合后的点和颜色
    """
    try:
        import open3d as o3d
    except ImportError:
        print("open3d not installed, concatenating directly / 未安装 open3d，直接拼接")
        all_pts = np.concatenate([p.cpu().numpy() if isinstance(p, torch.Tensor) else p
                                   for p in points_list], axis=0)
        all_cols = np.concatenate([c.cpu().numpy() if isinstance(c, torch.Tensor) else c
                                    for c in colors_list], axis=0)
        return all_pts, all_cols

    # Build Open3D point clouds / 构建 Open3D 点云
    pcds = []
    for pts, cols in zip(points_list, colors_list):
        if isinstance(pts, torch.Tensor):
            pts = pts.cpu().numpy()
        if isinstance(cols, torch.Tensor):
            cols = cols.cpu().numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        pcds.append(pcd)

    # Merge and downsample / 合并并下采样
    merged = pcds[0]
    for pcd in pcds[1:]:
        merged += pcd

    downsampled = merged.voxel_down_sample(voxel_size)
    return (
        np.asarray(downsampled.points),
        np.asarray(downsampled.colors),
    )
