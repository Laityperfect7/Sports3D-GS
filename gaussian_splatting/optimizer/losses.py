"""
Loss Functions / 损失函数
===========================
Implements L1 + SSIM combined loss as used in the 3DGS paper.
实现 3DGS 论文中使用的 L1 + SSIM 组合损失。

L = λ₁ * L1(pred, target) + λ₂ * (1 - SSIM(pred, target))
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from typing import Tuple


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean absolute error (L1) loss.
    平均绝对误差 (L1) 损失。
    """
    return torch.abs(pred - target).mean()


# ============================================================================
# SSIM Implementation / SSIM 实现
# ============================================================================

def _gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    """
    Create a 1D Gaussian kernel / 创建一维高斯核。
    """
    gauss = torch.tensor([
        exp(-(x - window_size // 2) ** 2 / (2.0 * sigma ** 2))
        for x in range(window_size)
    ], dtype=torch.float32)
    return gauss / gauss.sum()


def _create_window(window_size: int, channel: int) -> torch.Tensor:
    """
    Create a 2D Gaussian window for SSIM convolution.
    为 SSIM 卷积创建二维高斯窗口。
    """
    _1d_window = _gaussian_kernel(window_size, 1.5).unsqueeze(1)  # (W, 1)
    _2d_window = _1d_window @ _1d_window.T  # Outer product / 外积
    _2d_window = _2d_window.unsqueeze(0).unsqueeze(0)  # (1, 1, W, W)
    window = _2d_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> torch.Tensor:
    """
    Structural Similarity (SSIM) loss: 1.0 - SSIM score.
    结构相似性 (SSIM) 损失：1.0 - SSIM 分数。

    Args:
        pred: (B, C, H, W) predicted image / 预测图像
        target: (B, C, H, W) ground truth image / 真实图像
        window_size: Gaussian kernel size / 高斯核大小
        size_average: Average over batch if True / 是否对批次取平均
    Returns:
        SSIM loss = 1.0 - mean SSIM
    """
    C = pred.shape[1]
    device = pred.device

    window = _create_window(window_size, C).to(device)
    padding = window_size // 2

    # Compute local means / 计算局部均值
    mu1 = F.conv2d(pred, window, padding=padding, groups=C)
    mu2 = F.conv2d(target, window, padding=padding, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Compute local variances and covariance / 计算局部方差和协方差
    sigma1_sq = F.conv2d(pred * pred, window, padding=padding, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=padding, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=padding, groups=C) - mu1_mu2

    # SSIM stability constants / SSIM 稳定常数
    K1, K2 = 0.01, 0.03
    L = 1.0  # Dynamic range for normalized [0,1] images / 归一化 [0,1] 图像的动态范围
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # SSIM map / SSIM 图
    ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)

    if size_average:
        return 1.0 - ssim_map.mean()
    return 1.0 - ssim_map.mean([1, 2, 3])


# ============================================================================
# Combined Loss / 组合损失
# ============================================================================

def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_l1: float = 0.8,
    lambda_ssim: float = 0.2,
) -> Tuple[torch.Tensor, dict]:
    """
    Weighted L1 + SSIM combined loss (3DGS default).
    加权 L1 + SSIM 组合损失（3DGS 默认）。

    Args:
        pred: (B, C, H, W) rendered image / 渲染图像
        target: (B, C, H, W) ground truth image / 真实图像
        lambda_l1: Weight for L1 loss / L1 损失权重
        lambda_ssim: Weight for SSIM loss / SSIM 损失权重

    Returns:
        total_loss: Combined scalar loss / 组合标量损失
        info: Dict with individual loss components / 各损失分量的字典
    """
    l1 = l1_loss(pred, target)
    ssim = ssim_loss(pred, target)
    total = lambda_l1 * l1 + lambda_ssim * ssim

    info = {
        "l1_loss": l1.item(),
        "ssim_loss": ssim.item(),
        "total_loss": total.item(),
    }
    return total, info
