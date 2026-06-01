"""
Evaluation Metrics / 评估指标
================================
PSNR, SSIM, and LPIPS for image quality assessment.
用于图像质量评估的 PSNR、SSIM 和 LPIPS。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio / 计算峰值信噪比。

    Args:
        pred: (..., H, W) predicted image / 预测图像
        target: (..., H, W) ground truth image / 真实图像
        max_val: Maximum pixel value / 最大像素值
    Returns:
        PSNR in dB / PSNR（分贝）
    """
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return float("inf")
    return 20.0 * np.log10(max_val) - 10.0 * np.log10(mse)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    max_val: float = 1.0,
) -> float:
    """
    Compute Structural Similarity Index / 计算结构相似性指数。

    Simplified single-image version (no batch dimension needed).
    简化单图版本（无需批次维度）。

    Args:
        pred: (C, H, W) or (1, C, H, W) predicted image
        target: (C, H, W) or (1, C, H, W) target image
        window_size: Gaussian kernel size / 高斯核大小
        max_val: Maximum pixel value / 最大像素值
    Returns:
        SSIM score in [0, 1] / SSIM 分数
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)

    C = pred.shape[1]
    device = pred.device

    # Gaussian window / 高斯窗口
    sigma = 1.5
    gauss = torch.tensor([
        ((x - window_size // 2) / sigma) ** 2
        for x in range(window_size)
    ], device=device, dtype=torch.float32)
    gauss = torch.exp(-0.5 * gauss)
    gauss = gauss / gauss.sum()

    _1d = gauss.unsqueeze(1)
    _2d = (_1d @ _1d.T).unsqueeze(0).unsqueeze(0)
    window = _2d.expand(C, 1, window_size, window_size).contiguous()

    K1, K2 = 0.01, 0.03
    L = max_val
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=C)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=C) - mu1_mu2

    ssim_val = ((2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)

    return float(ssim_val.mean().item())


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = "alex",
    device: str = "cuda",
) -> float:
    """
    Compute LPIPS perceptual distance (requires lpips package).
    计算 LPIPS 感知距离（需要 lpips 包）。

    LPIPS (Learned Perceptual Image Patch Similarity) measures
    perceptual similarity using deep network features.
    LPIPS 使用深度网络特征测量感知相似度。

    Args:
        pred: (C, H, W) image tensor in [0, 1]
        target: (C, H, W) image tensor in [0, 1]
        net: Backbone network ("alex", "vgg", "squeeze") / 骨干网络
        device: Device to run on / 运行设备
    Returns:
        LPIPS distance (lower = more similar) / LPIPS 距离（越小越相似）
    """
    try:
        import lpips
    except ImportError:
        print("lpips not installed. Install: pip install lpips")
        return -1.0

    loss_fn = lpips.LPIPS(net=net).to(device)

    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)

    # LPIPS expects [-1, 1] or [0, 1] — we use [0, 1] then normalize to [-1, 1]
    pred_norm = pred * 2.0 - 1.0
    target_norm = target * 2.0 - 1.0

    with torch.no_grad():
        dist = loss_fn(pred_norm, target_norm)

    return float(dist.item())
