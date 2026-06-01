"""
Image Utility Functions / 图像工具函数
========================================
Basic image I/O, resizing, depth visualization helpers.
基本的图像读写、缩放、深度图可视化辅助函数。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Union, Optional
import cv2


def resize_image(
    image: Union[torch.Tensor, np.ndarray],
    size: tuple,  # (W, H)
    mode: str = "bilinear",
) -> Union[torch.Tensor, np.ndarray]:
    """
    Resize image to target (width, height).
    将图像调整为目标 (宽, 高)。

    Args:
        image: (C, H, W) tensor or (H, W, C) numpy array
        size: (width, height) target size
        mode: interpolation mode for tensor / 张量插值模式
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif image.dim() == 3:
            image = image.unsqueeze(0)  # (1, C, H, W)
        resized = F.interpolate(
            image,
            size=(size[1], size[0]),
            mode=mode,
            align_corners=False,
        )
        return resized.squeeze(0)
    else:
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    计算峰值信噪比。

    Args:
        pred, target: (C, H, W) tensors in [0, 1]
    Returns:
        PSNR in dB / 单位为 dB 的 PSNR
    """
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return float("inf")
    return 20.0 * np.log10(1.0) - 10.0 * np.log10(mse)


def visualize_depth(
    depth: Union[torch.Tensor, np.ndarray],
    max_depth: float = 80.0,
    colormap: int = cv2.COLORMAP_INFERNO,
) -> np.ndarray:
    """
    Colorize a depth map for display.
    将深度图着色用于显示。

    Args:
        depth: (H, W) depth map / 深度图
        max_depth: Maximum depth for normalization / 归一化的最大深度
        colormap: OpenCV colormap type / OpenCV 色彩映射类型
    Returns:
        (H, W, 3) uint8 BGR color image / BGR 彩色图
    """
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()

    depth_clamped = np.clip(depth, 0, max_depth)
    depth_norm = (depth_clamped / max_depth * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_norm, colormap)
    return colored


def save_tensor_image(
    tensor: torch.Tensor,
    path: Union[str, Path],
    denormalize: bool = False,
) -> None:
    """
    Save a (C, H, W) tensor as an image file.
    将 (C, H, W) 张量保存为图像文件。

    Args:
        tensor: (C, H, W) float tensor in [0, 1] / 值域为 [0, 1] 的浮点张量
        path: Output file path / 输出文件路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    img = tensor.detach().cpu()
    if img.dim() == 3:
        img = img.permute(1, 2, 0)  # (H, W, C)
    img = img.numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(path), img)
