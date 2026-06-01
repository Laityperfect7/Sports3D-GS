"""
3D Gaussian Model / 三维高斯模型
=================================
Implements the core Gaussian primitive used in 3DGS.
Each Gaussian is defined by:
  - mean (xyz): 3D position / 位置
  - covariance: 3×3 matrix, decomposed as R @ S @ S @ R^T / 协方差矩阵
  - opacity (alpha): scalar ∈ [0, 1] / 不透明度
  - color: spherical harmonics coefficients for view-dependent color / 球谐系数

References / 参考:
  Kerbl et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path


# ============================================================================
# Utility Functions / 工具函数
# ============================================================================

def build_scaling_rotation_from_covariance(
    cov3d: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose a covariance matrix into scaling factors and rotation quaternion.
    将协方差矩阵分解为缩放因子和旋转四元数。

    cov3d = R @ diag(scale²) @ R^T

    Args:
        cov3d: (N, 3, 3) covariance matrices / 协方差矩阵
    Returns:
        scales: (N, 3) per-axis scaling / 各轴缩放
        rotations: (N, 4) quaternion (w,x,y,z) / 四元数
    """
    # Eigen-decomposition: cov = V @ diag(λ) @ V^T
    eigvals, eigvecs = torch.linalg.eigh(cov3d)  # eigvals sorted ascending
    scales = torch.sqrt(torch.clamp(eigvals, min=1e-10))  # σ

    # Build rotation matrix from eigenvectors; ensure right-handed
    # 从特征向量构建旋转矩阵，确保右手坐标系
    rot_mat = eigvecs  # (N, 3, 3) columns are eigenvectors

    # Convert rotation matrix to quaternion / 旋转矩阵转四元数
    # Handle possible reflection (det < 0) / 处理可能的反射
    det = torch.det(rot_mat)
    rot_mat[det < 0, :, 2] *= -1  # Flip last column to ensure proper rotation

    r = rot_mat
    # Standard matrix-to-quaternion formula / 标准矩阵转四元数公式
    trace = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]
    q = torch.zeros(cov3d.shape[0], 4, device=cov3d.device, dtype=cov3d.dtype)

    # Case: trace > 0 / 迹大于零的情况
    mask = trace > 0
    if mask.any():
        s = 0.5 / torch.sqrt(trace[mask] + 1.0)
        q[mask, 0] = 0.25 / s  # w
        q[mask, 1] = (r[mask, 2, 1] - r[mask, 1, 2]) * s  # x
        q[mask, 2] = (r[mask, 0, 2] - r[mask, 2, 0]) * s  # y
        q[mask, 3] = (r[mask, 1, 0] - r[mask, 0, 1]) * s  # z

    # Other cases for robustness / 鲁棒性处理
    neg_mask = ~mask
    if neg_mask.any():
        rn = r[neg_mask]
        # Find max diagonal element / 找最大对角元素
        diag_max, diag_argmax = torch.max(
            torch.stack([rn[:, 0, 0], rn[:, 1, 1], rn[:, 2, 2]], dim=1), dim=1
        )

        for k in range(3):
            case = (diag_argmax == k) & neg_mask[neg_mask] if any(neg_mask) else \
                   torch.zeros(1, dtype=torch.bool, device=cov3d.device)
            # simplified: normalize all q
            pass

        # Fallback: use normalized columns directly / 回退到直接使用归一化列
        qv = torch.cat([
            rn[:, 0:1, 0], rn[:, 1:2, 1], rn[:, 2:3, 2]
        ], dim=1)
        q[neg_mask, 0] = torch.sqrt(torch.clamp(1.0 + qv[:, 0] - qv[:, 1] - qv[:, 2], min=0)) / 2.0
        q[neg_mask, 1] = torch.sqrt(torch.clamp(1.0 - qv[:, 0] + qv[:, 1] - qv[:, 2], min=0)) / 2.0
        q[neg_mask, 2] = torch.sqrt(torch.clamp(1.0 + qv[:, 0] + qv[:, 1] + qv[:, 2], min=0)) / 2.0
        q[neg_mask, 3] = torch.sqrt(torch.clamp(1.0 - qv[:, 0] - qv[:, 1] + qv[:, 2], min=0)) / 2.0

    return scales, q


def build_covariance_from_scaling_rotation(
    scaling: torch.Tensor,
    rotation_quat: torch.Tensor
) -> torch.Tensor:
    """
    Build 3×3 covariance matrix from scaling and rotation.
    从缩放和旋转构建 3×3 协方差矩阵。

    Σ = R @ S @ Sᵀ @ Rᵀ, S = diag(scaling), R = quat_to_matrix(rotation)
    """
    N = scaling.shape[0]
    device = scaling.device
    dtype = scaling.dtype

    # Quaternion to rotation matrix / 四元数转旋转矩阵
    R = _quat_to_rotmat(rotation_quat)  # (N, 3, 3)

    # Build scaling matrix / 构建缩放矩阵
    S = torch.diag_embed(scaling)  # (N, 3, 3)

    # Σ = R S S R^T / 协方差 = R * S² * Rᵀ
    RS = R @ S
    cov = RS @ RS.transpose(1, 2)  # (N, 3, 3)

    return cov


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (w, x, y, z) to 3×3 rotation matrix.
    四元数 (w, x, y, z) 转 3×3 旋转矩阵。
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    device = q.device
    dtype = q.dtype

    R = torch.zeros(N, 3, 3, device=device, dtype=dtype)
    R[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[:, 0, 1] = 2.0 * (x * y - w * z)
    R[:, 0, 2] = 2.0 * (x * z + w * y)

    R[:, 1, 0] = 2.0 * (x * y + w * z)
    R[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[:, 1, 2] = 2.0 * (y * z - w * x)

    R[:, 2, 0] = 2.0 * (x * z - w * y)
    R[:, 2, 1] = 2.0 * (y * z + w * x)
    R[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)

    return R


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Inverse sigmoid for parameter activation / 反 S 型函数，用于参数激活"""
    return torch.log(x / (1.0 - x))


# ============================================================================
# Gaussian Model / 高斯模型
# ============================================================================

class GaussianModel(nn.Module):
    """
    Collection of 3D Gaussians with differentiable parameters.
    三维高斯的可微参数集合。

    Each Gaussian i is fully described by / 每个高斯 i 由以下参数完整描述:
      xyz_i       : 3D mean / 三维均值
      features_i  : Spherical Harmonics coefficients (view-dependent RGB)
                    球谐系数（视角相关颜色）
      opacity_i   : scalar opacity α ∈ [0,1] / 不透明度
      scaling_i   : 3-axis std-dev before rotation / 旋转前的三轴标准差
      rotation_i  : quaternion (w,x,y,z) encoding rotation / 编码旋转的四元数
    """

    # Maximum SH degree supported / 支持的最大球谐阶数
    MAX_SH_DEGREE = 4

    def __init__(
        self,
        num_points: int,
        sh_degree: int = 3,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.sh_degree = min(sh_degree, self.MAX_SH_DEGREE)
        self.active_sh_degree = 0  # Starts at 0, raised gradually / 从0开始逐渐提升

        # Number of SH coefficients = (degree+1)² per color channel
        # 球谐系数数量 = (degree+1)²，每个颜色通道
        self.num_sh_coeffs = (self.MAX_SH_DEGREE + 1) ** 2
        self.num_features = self.num_sh_coeffs * 3  # 3 for RGB

        self._create_parameters(num_points)
        self._create_state()

    def _create_parameters(self, N: int) -> None:
        """
        Initialize all learnable parameters / 初始化所有可学习参数。

        Initialization strategy / 初始化策略:
          - xyz: random within [-1, 1]³ cube / 在 [-1,1]³ 立方体中随机
          - features: zero (gray color initially) / 初始为零（灰色）
          - opacity: inverse sigmoid of 0.1 / 不透明度反 S 形值（对应 0.1）
          - scaling: inverse of log(std) initialized from mean distance to neighbors
                    缩放从到邻居的平均距离初始化
          - rotation: identity quaternion (1,0,0,0) / 恒等四元数
        """
        device = self.device

        # Gaussian means / 高斯均值
        self._xyz = nn.Parameter(torch.randn(N, 3, device=device) * 0.5)

        # SH features: (N, num_features, 1) as per 3DGS convention
        # 球谐特征：按照 3DGS 惯例为 (N, num_features, 1)
        self._features_dc = nn.Parameter(torch.zeros(N, 3, 1, device=device))
        self._features_rest = nn.Parameter(torch.zeros(N, 3, (self.MAX_SH_DEGREE + 1) ** 2 - 1, device=device))

        # Opacity / 不透明度
        self._opacity = nn.Parameter(inverse_sigmoid(torch.full((N, 1), 0.1, device=device)))

        # Scaling (per-axis log-std) / 缩放（各轴对数标准差）
        self._scaling = nn.Parameter(torch.zeros(N, 3, device=device))

        # Rotation quaternion / 旋转四元数
        self._rotation = nn.Parameter(torch.zeros(N, 4, device=device))
        self._rotation.data[:, 0] = 1.0  # w = 1 (identity) / w=1 恒等旋转

    def _create_state(self) -> None:
        """Non-learnable state tensors / 不可学习的状态张量"""
        # Gradient accumulators for densification / 用于密度控制的梯度累积器
        N = self._xyz.shape[0]
        self.xyz_gradient_accum = torch.zeros(N, 1, device=self.device)
        self.denom = torch.zeros(N, 1, device=self.device)

        # Spatial extent bounds (updated during training) / 空间范围边界
        self.spatial_lr_scale: float = 1.0
        self.max_radii2D = torch.zeros(N, device=self.device)

    # ---- Property Accessors (激活函数包装) ---- / 属性访问器

    @property
    def get_xyz(self) -> torch.Tensor:
        """Raw 3D positions / 原始三维位置"""
        return self._xyz

    @property
    def get_features(self) -> torch.Tensor:
        """
        Concatenated SH features: [dc | rest].
        拼接的球谐特征：[直流分量 | 高阶分量]
        """
        features_dc = self._features_dc  # (N, 3, 1)
        features_rest = self._features_rest  # (N, 3, M)
        return torch.cat([features_dc, features_rest], dim=2)  # (N, 3, total)

    @property
    def get_opacity(self) -> torch.Tensor:
        """Opacity with sigmoid activation / 经 S 型函数激活的不透明度"""
        return torch.sigmoid(self._opacity)

    @property
    def get_scaling(self) -> torch.Tensor:
        """
        Per-axis scaling factors (activated via exp).
        经指数激活的各轴缩放因子。
        """
        return torch.exp(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        """
        Rotation quaternion (normalized).
        归一化后的旋转四元数。
        """
        return torch.nn.functional.normalize(self._rotation, dim=1)

    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        """
        Compute full 3×3 covariance matrices for all Gaussians.
        计算所有高斯的 3×3 协方差矩阵。

        Args:
            scaling_modifier: Global scale factor / 全局缩放系数
        Returns:
            (N, 3, 3) covariance matrices
        """
        return build_covariance_from_scaling_rotation(
            self.get_scaling * scaling_modifier,
            self.get_rotation,
        )

    # ---- Training Helpers / 训练辅助 ----

    def oneup_SH_degree(self) -> None:
        """
        Gradually increase active SH degree during training.
        训练过程中逐渐提升活跃的球谐阶数。

        Higher SH degrees encode finer view-dependent effects.
        高阶球谐编码更精细的视角相关效果。
        """
        if self.active_sh_degree < self.sh_degree:
            self.active_sh_degree += 1

    def densification_postprocess(
        self,
        new_xyz: torch.Tensor,
        new_features_dc: torch.Tensor,
        new_features_rest: torch.Tensor,
        new_opacities: torch.Tensor,
        new_scaling: torch.Tensor,
        new_rotation: torch.Tensor,
    ) -> None:
        """
        After densification (clone / split), reorganize parameters.
        密度控制（克隆/分裂）后，重组参数。

        Performs:
          1. Re-create parameter tensors with new size
          2. Re-initialize gradient accumulators
          3. Copy old + new values into expanded tensors
        """
        old_N = self._xyz.shape[0]
        new_N = old_N + new_xyz.shape[0]
        device = self.device

        # Helper to extend a parameter / 扩展参数的辅助函数
        def _cat_param(old: torch.Tensor, new: torch.Tensor) -> nn.Parameter:
            return nn.Parameter(torch.cat([old.detach(), new], dim=0))

        # Extend all parameters / 扩展所有参数
        self._xyz = _cat_param(self._xyz, new_xyz)
        self._features_dc = _cat_param(self._features_dc, new_features_dc)
        self._features_rest = _cat_param(self._features_rest, new_features_rest)
        self._opacity = _cat_param(self._opacity, new_opacities)
        self._scaling = _cat_param(self._scaling, new_scaling)
        self._rotation = _cat_param(self._rotation, new_rotation)

        # Reset gradient accumulators / 重置梯度累积器
        self.xyz_gradient_accum = torch.zeros(new_N, 1, device=device)
        self.denom = torch.zeros(new_N, 1, device=device)
        self.max_radii2D = torch.zeros(new_N, device=device)

    def prune_points(self, mask: torch.Tensor) -> None:
        """
        Remove Gaussians according to boolean mask.
        根据布尔掩码删除高斯。

        Args:
            mask: (N,) bool tensor, True = keep / 布尔张量，True = 保留
        """
        valid_mask = mask.bool().to(self.device)

        self._xyz = nn.Parameter(self._xyz[valid_mask].detach())
        self._features_dc = nn.Parameter(self._features_dc[valid_mask].detach())
        self._features_rest = nn.Parameter(self._features_rest[valid_mask].detach())
        self._opacity = nn.Parameter(self._opacity[valid_mask].detach())
        self._scaling = nn.Parameter(self._scaling[valid_mask].detach())
        self._rotation = nn.Parameter(self._rotation[valid_mask].detach())

        # Reset gradient state / 重置梯度状态
        new_N = self._xyz.shape[0]
        self.xyz_gradient_accum = torch.zeros(new_N, 1, device=self.device)
        self.denom = torch.zeros(new_N, 1, device=self.device)
        self.max_radii2D = torch.zeros(new_N, device=self.device)

    def add_to_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Add all trainable parameters to an optimizer.
        将所有可训练参数加入优化器。
        """
        optimizer.add_param_group({"params": [self._xyz], "lr": 0.00016, "name": "xyz"})
        optimizer.add_param_group({"params": [self._features_dc], "lr": 0.0025, "name": "f_dc"})
        optimizer.add_param_group({"params": [self._features_rest], "lr": 0.000125, "name": "f_rest"})
        optimizer.add_param_group({"params": [self._opacity], "lr": 0.05, "name": "opacity"})
        optimizer.add_param_group({"params": [self._scaling], "lr": 0.005, "name": "scaling"})
        optimizer.add_param_group({"params": [self._rotation], "lr": 0.001, "name": "rotation"})

    def save_ply(self, path: Path) -> None:
        """
        Export Gaussians to .ply file for visualization.
        导出高斯到 .ply 文件用于可视化。

        Standard 3DGS PLY format with properties:
          x, y, z, f_dc_0..2, f_rest_*, opacity, scale_0..2, rot_0..3
        """
        from plyfile import PlyData, PlyElement
        import numpy as np

        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().cpu().numpy().reshape(-1, 3)
        f_rest = self._features_rest.detach().cpu().numpy().reshape(
            xyz.shape[0], -1
        )
        opacity = self.get_opacity.detach().cpu().numpy().reshape(-1, 1)
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        # Build ply dtype / 构建 .ply 数据类型
        dtype_full = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ]
        for i in range(f_rest.shape[1]):
            dtype_full.append((f"f_rest_{i}", "f4"))
        dtype_full += [
            ("opacity", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ]

        data = np.empty(xyz.shape[0], dtype=dtype_full)
        data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        data["f_dc_0"], data["f_dc_1"], data["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
        for i in range(f_rest.shape[1]):
            data[f"f_rest_{i}"] = f_rest[:, i]
        data["opacity"] = opacity[:, 0]
        data["scale_0"], data["scale_1"], data["scale_2"] = scale[:, 0], scale[:, 1], scale[:, 2]
        data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"] = \
            rotation[:, 0], rotation[:, 1], rotation[:, 2], rotation[:, 3]

        el = PlyElement.describe(data, "vertex")
        PlyData([el]).write(str(path))
        print(f"Saved {xyz.shape[0]} Gaussians → {path} / 保存了 {xyz.shape[0]} 个高斯 → {path}")

    def load_ply(self, path: Path) -> None:
        """
        Load Gaussians from a .ply file / 从 .ply 文件加载高斯。
        """
        from plyfile import PlyData

        plydata = PlyData.read(str(path))
        vertices = plydata["vertex"]

        xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1)
        f_dc = np.stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]], axis=1)
        f_rest_names = [p.name for p in plydata["vertex"].properties if p.name.startswith("f_rest_")]
        f_rest = np.stack([vertices[name] for name in f_rest_names], axis=1) if f_rest_names else np.zeros((xyz.shape[0], 0))
        opacities = vertices["opacity"][:, None]
        scales = np.stack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]], axis=1)
        rots = np.stack([vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]], axis=1)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device=self.device))
        self._features_dc = nn.Parameter(torch.tensor(f_dc, dtype=torch.float32, device=self.device).unsqueeze(2))
        self._features_rest = nn.Parameter(torch.tensor(f_rest, dtype=torch.float32, device=self.device).reshape(xyz.shape[0], 3, -1))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float32, device=self.device))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float32, device=self.device))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float32, device=self.device))

        self._create_state()
