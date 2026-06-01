"""
Gaussian Rasterizer Wrapper / 高斯光栅化器封装
================================================
Abstract wrapper around the CUDA diff-gaussian-rasterization module.
Supports both the real CUDA backend and a pure-PyTorch fallback
for development without CUDA compilation.

封装 CUDA diff-gaussian-rasterization 模块的抽象接口。
在无 CUDA 编译时提供纯 PyTorch 回退方案以支持开发。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

# Try importing the real CUDA rasterizer / 尝试导入真实 CUDA 光栅化器
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer as _CURasterizer
    _HAS_CUDA_RASTERIZER = True
except ImportError:
    _HAS_CUDA_RASTERIZER = False


class GaussianRasterizer(nn.Module):
    """
    Differentiable Gaussian rasterizer that renders Gaussians to a 2D image.
    可微高斯光栅化器，将高斯渲染为 2D 图像。

    The rasterizer:
      1. Projects 3D Gaussians to 2D screen space (EWA splatting)
         将三维高斯投影到二维屏幕空间（EWA 溅射）
      2. Sorts by depth for proper alpha compositing
         按深度排序实现正确的 alpha 合成
      3. Produces rendered image and alpha map
         生成渲染图像和 alpha 图
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        tanfovx: float,
        tanfovy: float,
        bg_color: torch.Tensor,
        scale_modifier: float = 1.0,
        viewmatrix: Optional[torch.Tensor] = None,
        projmatrix: Optional[torch.Tensor] = None,
        sh_degree: int = 3,
        near_clip: float = 0.01,
        far_clip: float = 100.0,
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.tanfovx = tanfovx
        self.tanfovy = tanfovy
        self.bg_color = bg_color
        self.scale_modifier = scale_modifier
        self.viewmatrix = viewmatrix
        self.projmatrix = projmatrix
        self.sh_degree = sh_degree
        self.near_clip = near_clip
        self.far_clip = far_clip

        if not _HAS_CUDA_RASTERIZER:
            print(
                "⚠ CUDA rasterizer not found. Using PyTorch fallback (slow!). "
                "Install diff-gaussian-rasterization for real-time rendering."
                " / 未找到 CUDA 光栅化器。使用 PyTorch 回退方案（慢！）。"
            )

    def forward(
        self,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        opacity: torch.Tensor,
        sh_features: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        cov3D_precomp: Optional[torch.Tensor] = None,
        colors_precomp: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Render Gaussians to image / 将高斯渲染为图像。

        Args:
            means3D: (N, 3) Gaussian centers in world space / 世界空间的高斯中心
            means2D: (N, 2) projected 2D means (for CUDA rasterizer) / 投影后的 2D 均值
            opacity: (N, 1) opacity values / 不透明度值
            sh_features: (N, 3, C) SH coefficients / 球谐系数
            scales: (N, 3) per-axis scaling / 各轴缩放
            rotations: (N, 4) rotation quaternions / 旋转四元数
            cov3D_precomp: Precomputed 3D covariances (optional) / 预计算的三维协方差
            colors_precomp: Precomputed colors (optional, bypasses SH) / 预计算颜色

        Returns:
            dict with "render" (3, H, W) RGB image and "alpha" (1, H, W) alpha map
        """
        if _HAS_CUDA_RASTERIZER and self.viewmatrix is not None:
            return self._forward_cuda(
                means3D, means2D, opacity, sh_features,
                scales, rotations, cov3D_precomp, colors_precomp,
            )
        else:
            return self._forward_pytorch(
                means3D, opacity, sh_features, scales, rotations,
                cov3D_precomp, colors_precomp,
            )

    def _forward_cuda(
        self,
        means3D, means2D, opacity, sh_features,
        scales, rotations, cov3D_precomp, colors_precomp,
    ) -> dict:
        """Real CUDA rasterization path / 真实 CUDA 光栅化路径"""
        raster_settings = GaussianRasterizationSettings(
            image_height=self.image_height,
            image_width=self.image_width,
            tanfovx=self.tanfovx,
            tanfovy=self.tanfovy,
            bg=self.bg_color,
            scale_modifier=self.scale_modifier,
            viewmatrix=self.viewmatrix,
            projmatrix=self.projmatrix,
            sh_degree=self.sh_degree,
            campos=self.camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = _CURasterizer(raster_settings=raster_settings)

        # Rasterize / 光栅化
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=sh_features,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        # rendered_image shape: (N_rendered, H, W, 3) or (3, H, W)
        if rendered_image.dim() == 4:
            rendered_image = rendered_image[0].permute(2, 0, 1)

        return {
            "render": rendered_image,
            "alpha": None,  # CUDA rasterizer doesn't return alpha separately
            "radii": radii,
        }

    def _forward_pytorch(
        self,
        means3D, opacity, sh_features, scales, rotations,
        cov3D_precomp, colors_precomp,
    ) -> dict:
        """
        Pure-PyTorch fallback rasterizer (simplified, for development).
        纯 PyTorch 回退光栅化器（简化版，用于开发）。

        This is a simplified splatting implementation that:
          - Projects 3D means to 2D
          - Computes 2D Gaussian kernels
          - Alpha-composites front-to-back

        NOTE: This is ~100× slower than CUDA. For training, use the CUDA backend.
        注意：比 CUDA 慢约百倍。训练时请使用 CUDA 后端。
        """
        device = means3D.device
        H, W = self.image_height, self.image_width

        # Project 3D means to 2D using a simple pinhole model
        # 使用简单针孔模型将3D均值投影到2D
        if self.viewmatrix is not None:
            # Apply world-to-view transform / 应用世界到视图变换
            ones = torch.ones(means3D.shape[0], 1, device=device)
            pts_homo = torch.cat([means3D, ones], dim=1)  # (N, 4)
            view_pts = (self.viewmatrix @ pts_homo.T).T[:, :3]  # (N, 3)
        else:
            view_pts = means3D

        # Simple perspective projection / 简单透视投影
        fx = W / (2.0 * self.tanfovx) if self.tanfovx > 0 else 500.0
        fy = H / (2.0 * self.tanfovy) if self.tanfovy > 0 else 500.0

        z = torch.clamp(view_pts[:, 2], min=1e-6)
        u = fx * view_pts[:, 0] / z + W / 2.0
        v = -fy * view_pts[:, 1] / z + H / 2.0

        # Compute 2D covariance / 计算二维协方差
        # Simplified: use isotropic Gaussians in screen space
        # 简化版：在屏幕空间使用各向同性高斯
        sigma_2d = torch.clamp(scales[:, 0] * fx / z, min=0.3, max=50.0)

        # Alpha-blend front-to-back / 从前到后 alpha 混合
        # Sort by depth / 按深度排序
        depth = view_pts[:, 2]
        sorted_idx = torch.argsort(depth, descending=False)  # Near to far

        # Initialize canvas / 初始化画布
        render = self.bg_color.expand(3, H, W).clone()
        alpha_canvas = torch.zeros(1, H, W, device=device)

        # Create pixel grid / 创建像素网格
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )

        # Simplified splatting — process in batches for memory
        # 简化溅射 — 分批处理以节省内存
        batch_size = 1000
        for start in range(0, means3D.shape[0], batch_size):
            end = min(start + batch_size, means3D.shape[0])
            batch_idx = sorted_idx[start:end]

            mu_u = u[batch_idx]  # (B,)
            mu_v = v[batch_idx]  # (B,)
            sigma = sigma_2d[batch_idx]  # (B,)
            alpha = opacity[batch_idx, 0]  # (B,)

            if colors_precomp is not None:
                colors = colors_precomp[batch_idx]  # (B, 3)
            else:
                colors = torch.sigmoid(sh_features[batch_idx, :, 0])  # (B, 3) — DC only for fallback

            for j in range(len(batch_idx)):
                # 2D Gaussian kernel / 二维高斯核
                dist = (xx - mu_u[j]) ** 2 + (yy - mu_v[j]) ** 2
                gauss_weight = torch.exp(-0.5 * dist / (sigma[j] ** 2 + 1e-6))

                contrib_alpha = alpha[j] * gauss_weight.unsqueeze(0)  # (1, H, W)
                contrib_color = colors[j].unsqueeze(1).unsqueeze(1) * contrib_alpha  # (3, H, W)

                # Alpha compositing / Alpha 合成
                T = 1.0 - alpha_canvas  # Transmittance / 透射率
                render = render + contrib_color * T
                alpha_canvas = alpha_canvas + contrib_alpha * T

        return {"render": render, "alpha": alpha_canvas, "radii": None}
