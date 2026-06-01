"""
Sports Video Dataset / 体育视频数据集
=======================================
Loads preprocessed frames with optional depth maps and masks.
加载预处理帧，可选的深度图和掩码。

Supports:
  - Random frame sampling per training iteration
    每次训练迭代的随机帧采样
  - Train/val split / 训练/验证划分
  - Depth map loading for point cloud initialization
    深度图加载用于点云初始化
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np


class SportsVideoDataset(Dataset):
    """
    Dataset of frames extracted from a sports video.
    从体育视频提取的帧数据集。

    Each item is a (3, H, W) float32 tensor in [0, 1].
    每个项目是值域为 [0,1] 的 (3, H, W) float32 张量。
    """

    def __init__(
        self,
        frame_dir: str,
        depth_dir: Optional[str] = None,
        mask_dir: Optional[str] = None,
        resolution: Tuple[int, int] = (1280, 720),
        train_split: float = 0.9,
        white_background: bool = False,
    ):
        """
        Args:
            frame_dir: Directory of preprocessed frames / 预处理帧目录
            depth_dir: Optional directory of depth maps / 可选深度图目录
            mask_dir: Optional directory of foreground masks / 可选前景掩码目录
            resolution: (width, height) to resize frames / 帧调整尺寸 (宽, 高)
            train_split: Fraction for training (rest for val) / 训练集分数
            white_background: White BG instead of black / 白背景替代黑背景
        """
        self.frame_dir = Path(frame_dir)
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.resolution = resolution  # (W, H)
        self.white_background = white_background

        # Collect frame paths / 收集帧路径
        self.frame_paths: List[Path] = sorted(
            list(self.frame_dir.glob("*.png")) +
            list(self.frame_dir.glob("*.jpg")) +
            list(self.frame_dir.glob("*.jpeg"))
        )
        if not self.frame_paths:
            raise FileNotFoundError(
                f"No frames found in {frame_dir}. Run preprocess.py first. "
                f"/ 未找到帧。请先运行 preprocess.py。"
            )

        # Train/val split / 训练/验证划分
        N = len(self.frame_paths)
        self.train_indices = list(range(int(N * train_split)))
        self.val_indices = list(range(self.train_indices[-1] + 1, N))

        self.image_size = resolution  # (W, H)

        print(f"Dataset loaded: {N} frames / 数据集加载完成: {N} 帧")
        print(f"  Train: {len(self.train_indices)} | Val: {len(self.val_indices)} / 训练 | 验证")
        print(f"  Resolution: {resolution} / 分辨率")

    def __len__(self) -> int:
        return len(self.train_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return (image_tensor, frame_index) / 返回 (图像张量, 帧索引)"""
        real_idx = self.train_indices[idx]
        return self._load_frame(real_idx), real_idx

    def _load_frame(self, idx: int) -> torch.Tensor:
        """
        Load and preprocess a single frame / 加载并预处理单帧。
        """
        path = self.frame_paths[idx]
        image = cv2.imread(str(path))
        if image is None:
            raise IOError(f"Failed to load: {path} / 加载失败")

        # BGR → RGB / BGR 转 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize if needed / 如果有需要则调整大小
        W, H = self.resolution
        if image.shape[1] != W or image.shape[0] != H:
            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1] / 归一化到 [0, 1]
        image = image.astype(np.float32) / 255.0

        # Apply mask if available / 如果有掩码则应用
        if self.mask_dir is not None:
            mask_path = self.mask_dir / f"{path.stem}_mask.png"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (W, H)) / 255.0
                mask = mask[..., None]  # (H, W, 1)
                if self.white_background:
                    image = image * mask + (1 - mask) * 1.0
                else:
                    image = image * mask

        # Convert to CHW tensor / 转为 CHW 张量
        return torch.from_numpy(image).permute(2, 0, 1).float()

    def sample_random(self) -> Tuple[torch.Tensor, int]:
        """
        Random sampling for stochastic training.
        随机采样用于随机训练。
        """
        idx = np.random.randint(0, len(self.train_indices))
        return self[idx]

    def sample_val(self) -> Tuple[torch.Tensor, int]:
        """Sample a random validation frame / 随机采样验证帧"""
        if not self.val_indices:
            return self.sample_random()
        idx = np.random.choice(self.val_indices)
        return self._load_frame(idx), idx

    def get_initial_point_cloud(
        self, max_points: int = 50000
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Build initial 3D point cloud from depth maps (if available).
        从深度图构建初始三维点云（如果可用）。

        Uses the first frame + its depth map to unproject pixels to 3D.
        使用第一帧及其深度图将像素反投影到三维。

        Returns:
            (points, colors): (N, 3) tensors or (None, None) if no depth data
        """
        if self.depth_dir is None or not self.depth_dir.exists():
            return None, None

        # Load first frame / 加载第一帧
        image = self._load_frame(0)  # (3, H, W)
        _, H, W = image.shape

        # Find corresponding depth map / 查找对应的深度图
        depth_paths = sorted(list(self.depth_dir.glob("*.npy"))) + \
                      sorted(list(self.depth_dir.glob("*.png")))
        if not depth_paths:
            return None, None

        depth = np.load(str(depth_paths[0])) if depth_paths[0].suffix == ".npy" \
                else cv2.imread(str(depth_paths[0]), cv2.IMREAD_UNCHANGED)

        # Ensure matching resolution / 确保匹配分辨率
        if depth.shape[:2] != (H, W):
            depth = cv2.resize(depth, (W, H))

        depth = depth.astype(np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        # Unproject to 3D using pinhole model / 使用针孔模型反投影到三维
        # Assuming fx ~ W, fy ~ H (approximate for unknown camera)
        # 假设 fx~W, fy~H（未知相机时近似）
        fx = fy = W * 0.75  # Reasonable default / 合理默认
        cx, cy = W / 2.0, H / 2.0

        yy, xx = torch.meshgrid(
            torch.arange(H), torch.arange(W), indexing="ij"
        )

        z = torch.from_numpy(depth).float()
        x = (xx - cx) * z / fx
        y = (yy - cy) * z / fy

        # Stack to (H*W, 3) / 堆叠为 (H*W, 3)
        points = torch.stack([x, -y, z], dim=2).reshape(-1, 3)  # Flip y / 翻转 y

        # Filter invalid depths / 过滤无效深度
        valid = (z.reshape(-1) > 0) & (z.reshape(-1) < 80.0)
        points = points[valid]
        colors = image.permute(1, 2, 0).reshape(-1, 3)[valid]

        # Subsample to max_points / 下采样到 max_points
        if points.shape[0] > max_points:
            indices = torch.randperm(points.shape[0])[:max_points]
            points = points[indices]
            colors = colors[indices]

        return points.float(), colors.float()
