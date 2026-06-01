"""
Adaptive Density Control / 自适应密度控制
===========================================
Implements the densification strategy from 3DGS:
  1. Clone: duplicate small Gaussians in under-reconstructed regions
     克隆：在重建不足区域复制小高斯
  2. Split: replace large Gaussians with smaller copies
     分裂：用小副本替换大高斯
  3. Prune: remove low-opacity or overly-large Gaussians
     剪枝：移除低不透明度或过大的高斯

Reference / 参考:
  Kerbl et al. "3D Gaussian Splatting" (2023), Section 5.2
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


class DensificationController:
    """
    Controls when and how to clone / split / prune Gaussians.
    控制何时以及如何克隆/分裂/剪枝高斯。
    """

    def __init__(
        self,
        densify_from_iter: int = 500,
        densify_until_iter: int = 15000,
        densification_interval: int = 100,
        opacity_reset_interval: int = 3000,
        densify_grad_threshold: float = 0.0002,
        densify_opacity_threshold: float = 0.005,
        densify_size_threshold: float = 0.01,  # as fraction of scene extent
        prune_opacity_threshold: float = 0.005,
        max_screen_size: int = 20,
    ):
        """
        Args:
            densify_from_iter: Step to start densification / 开始密度控制的步数
            densify_until_iter: Step to stop densification / 停止密度控制的步数
            densification_interval: Steps between densification attempts / 密度控制间隔
            opacity_reset_interval: Reset opacity values every N steps / 重置不透明度的间隔
            densify_grad_threshold: Gradient threshold to clone/split / 克隆/分裂的梯度阈值
            densify_opacity_threshold: Min opacity before pruning / 剪枝前的最小不透明度
            densify_size_threshold: Max Gaussian size before splitting (fraction of extent)
                                    分裂前的高斯最大尺寸（场景范围的分数）
            prune_opacity_threshold: Opacity below which Gaussians are pruned
                                     低于此不透明度的高斯被剪枝
            max_screen_size: Max 2D radius before splitting / 分裂前最大二维半径
        """
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densification_interval = densification_interval
        self.opacity_reset_interval = opacity_reset_interval
        self.densify_grad_threshold = densify_grad_threshold
        self.densify_opacity_threshold = densify_opacity_threshold
        self.densify_size_threshold = densify_size_threshold
        self.prune_opacity_threshold = prune_opacity_threshold
        self.max_screen_size = max_screen_size

    def should_densify(self, iteration: int) -> bool:
        """
        Check if densification should happen at this step.
        检查当前步数是否应进行密度控制。
        """
        return (
            self.densify_from_iter <= iteration <= self.densify_until_iter
            and iteration % self.densification_interval == 0
        )

    def should_reset_opacity(self, iteration: int) -> bool:
        """Check if opacity reset should happen at this step / 检查是否应重置不透明度"""
        return (
            iteration < self.densify_until_iter
            and iteration % self.opacity_reset_interval == 0
        )

    def compute_densification_mask(
        self,
        gaussian_model: nn.Module,
        extent: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Determine which Gaussians to clone, split, or prune.
        决定哪些高斯需要克隆、分裂或剪枝。

        Strategy / 策略:
          - High gradient + small size → clone (add detail)
            高梯度 + 小尺寸 → 克隆（增加细节）
          - High gradient + large size → split (reduce large blobs)
            高梯度 + 大尺寸 → 分裂（减少大块）
          - Low opacity → prune
            低不透明度 → 剪枝

        Args:
            gaussian_model: Current Gaussian model / 当前高斯模型
            extent: Spatial extent of the scene / 场景空间范围
        Returns:
            (clone_mask, split_mask, prune_mask): Boolean tensors
            布尔张量元组：(克隆掩码, 分裂掩码, 剪枝掩码)
        """
        N = gaussian_model._xyz.shape[0]
        device = gaussian_model.device

        # Average gradient per Gaussian since last densification
        # 自上次密度控制以来每个高斯的平均梯度
        grads = gaussian_model.xyz_gradient_accum / gaussian_model.denom.clamp(min=1)
        grads = grads.squeeze()  # (N,)

        # Current Gaussian sizes / 当前高斯尺寸
        scales = gaussian_model.get_scaling  # (N, 3)
        max_scales = scales.max(dim=1).values  # (N,) — max axis scale

        # Opacity values / 不透明度值
        opacity = gaussian_model.get_opacity.squeeze()  # (N,)

        # ---- Prune mask: low opacity or too large / 剪枝掩码 ----
        prune_mask = (opacity < self.prune_opacity_threshold) | \
                     (max_scales > extent * 2.0)

        # ---- Clone mask: small Gaussians needing more detail / 克隆掩码 ----
        clone_mask = (
            (grads >= self.densify_grad_threshold)
            & (max_scales <= self.densify_size_threshold * extent)
            & ~prune_mask
        )

        # ---- Split mask: large Gaussians needing subdivision / 分裂掩码 ----
        split_mask = (
            (grads >= self.densify_grad_threshold)
            & (max_scales > self.densify_size_threshold * extent)
            & ~prune_mask
        )

        return clone_mask, split_mask, prune_mask

    def clone_gaussians(
        self, gaussian_model: nn.Module, clone_mask: torch.Tensor
    ) -> dict:
        """
        Clone selected Gaussians (duplicate parameters exactly).
        克隆选中的高斯（精确复制参数）。

        Cloning adds detail in under-reconstructed regions.
        克隆在重建不足区域增加细节。
        """
        if not clone_mask.any():
            return {}

        selected = clone_mask.nonzero(as_tuple=True)[0]

        new_xyz = gaussian_model._xyz[selected].clone()
        new_features_dc = gaussian_model._features_dc[selected].clone()
        new_features_rest = gaussian_model._features_rest[selected].clone()
        new_opacity = gaussian_model._opacity[selected].clone()
        new_scaling = gaussian_model._scaling[selected].clone()
        new_rotation = gaussian_model._rotation[selected].clone()

        return {
            "xyz": new_xyz,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacity": new_opacity,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

    def split_gaussians(
        self, gaussian_model: nn.Module, split_mask: torch.Tensor
    ) -> dict:
        """
        Split large Gaussians into two smaller ones placed at ±σ from mean.
        将大高斯沿均值位置 ±σ 分裂为两个较小的高斯。

        The two children have:
          - Position: parent_mean ± (scaling vector in random direction)
            位置：父均值 ±（随机方向的缩放向量）
          - Scale: parent_scale / 1.6 (heuristic for volume preservation)
            缩放：父缩放 / 1.6（体积守恒的启发式）
          - Other params: inherited from parent
            其他参数：继承自父高斯
        """
        if not split_mask.any():
            return {}

        selected = split_mask.nonzero(as_tuple=True)[0]
        N_split = len(selected)
        device = gaussian_model.device

        parent_xyz = gaussian_model._xyz[selected]
        parent_scales = gaussian_model.get_scaling[selected]
        parent_rot = gaussian_model.get_rotation[selected]

        # Sample random directions for each Gaussian
        # 为每个高斯采样随机方向
        dirs = torch.randn(N_split, 3, device=device)
        dirs = torch.nn.functional.normalize(dirs, dim=1)

        # Apply rotation and scale to get offset in world coords
        # 应用旋转和缩放获得世界坐标下的偏移
        # Simplified: offset proportional to scale along random direction
        offset = dirs * parent_scales * 0.5  # Half a std-dev / 半个标准差

        child1_xyz = parent_xyz + offset
        child2_xyz = parent_xyz - offset

        # Reduce scale for children / 子高斯缩小尺寸
        child_scales = torch.log(parent_scales / 1.6)
        opacity_val = gaussian_model._opacity[selected]

        # Stack two children interleaved / 交错堆叠两个子高斯
        new_xyz = torch.cat([child1_xyz, child2_xyz], dim=0)
        new_features_dc = gaussian_model._features_dc[selected].repeat(2, 1, 1)
        new_features_rest = gaussian_model._features_rest[selected].repeat(2, 1, 1)
        new_opacity = opacity_val.repeat(2, 1)
        new_scaling = child_scales.repeat(2, 1)
        new_rotation = parent_rot.repeat(2, 1)

        return {
            "xyz": new_xyz,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacity": new_opacity,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

    def reset_opacity(self, gaussian_model: nn.Module) -> None:
        """
        Reset all opacity values to a low value.
        将所有不透明度重置为较低值。

        This allows the optimizer to re-discover which Gaussians
        are actually needed, effectively removing dead ones.
        这允许优化器重新发现哪些高斯实际上是需要的，有效移除无效的。
        """
        import torch.nn.functional as F

        # Set raw opacity parameter so sigmoid(output) ≈ 0.01
        # 设置原始不透明度参数使得 sigmoid(输出) ≈ 0.01
        target_opacity = 0.01
        new_val = torch.log(
            torch.tensor(target_opacity / (1.0 - target_opacity),
                        device=gaussian_model.device)
        )
        gaussian_model._opacity.data.fill_(new_val)

    def densify_and_prune(
        self,
        gaussian_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        extent: float,
    ) -> dict:
        """
        Full densification + pruning step / 完整密度控制 + 剪枝步骤。

        Returns stats dict / 返回统计字典.
        """
        stats = {"cloned": 0, "split": 0, "pruned": 0}

        clone_mask, split_mask, prune_mask = self.compute_densification_mask(
            gaussian_model, extent
        )

        # --- Prune first / 先剪枝 ---
        if prune_mask.any():
            num_before = gaussian_model._xyz.shape[0]
            gaussian_model.prune_points(~prune_mask)  # Keep non-pruned / 保留未剪枝的
            stats["pruned"] = num_before - gaussian_model._xyz.shape[0]

        # --- Clone / 克隆 ---
        clone_data = self.clone_gaussians(gaussian_model, clone_mask)
        if clone_data:
            stats["cloned"] = clone_data["xyz"].shape[0]
            gaussian_model.densification_postprocess(**clone_data)

        # --- Split / 分裂 ---
        split_data = self.split_gaussians(gaussian_model, split_mask)
        if split_data:
            stats["split"] = split_data["xyz"].shape[0]
            gaussian_model.densification_postprocess(**split_data)

        # Rebuild optimizer param groups after densification
        # 密度控制后重建优化器参数组
        if stats["cloned"] > 0 or stats["split"] > 0 or stats["pruned"] > 0:
            # Clear old param groups and re-add / 清除旧参数组并重新添加
            optimizer.param_groups.clear()
            gaussian_model.add_to_optimizer(optimizer)

        return stats
