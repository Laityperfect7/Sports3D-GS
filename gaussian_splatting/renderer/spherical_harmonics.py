"""
Spherical Harmonics Evaluator / 球谐函数求值器
================================================
Evaluates spherical harmonics (SH) coefficients to produce
view-dependent RGB colors for each Gaussian.

根据球谐（SH）系数求值，为每个高斯生成视角相关的 RGB 颜色。

Reference / 参考:
  - Kerbl et al. 3DGS (2023) Appendix B
  - Spherical harmonics up to degree 4 / 最高 4 阶球谐函数
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


# SH coefficient normalization constants (Condon-Shortley phase)
# 球谐归一化常数（Condon-Shortley 相位约定）
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


class SphericalHarmonicsEvaluator:
    """
    Evaluate SH basis functions given view direction.
    根据观察方向计算球谐基函数值。
    """

    def __init__(self, sh_degree: int = 3):
        self.sh_degree = sh_degree
        self.num_coeffs = (sh_degree + 1) ** 2

    @staticmethod
    def evaluate_sh_basis(
        sh_degree: int, dirs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SH basis for a batch of directions.
        为一批方向计算球谐基函数。

        Args:
            sh_degree: Max SH degree / 最大球谐阶数
            dirs: (N, 3) normalized direction vectors / 归一化方向向量
        Returns:
            (N, (degree+1)²) basis values / 基函数值
        """
        x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        N = dirs.shape[0]
        device = dirs.device

        num_bases = (sh_degree + 1) ** 2
        result = torch.zeros(N, num_bases, device=device, dtype=dirs.dtype)

        # Degree 0 / 零阶
        result[:, 0] = C0

        if sh_degree < 1:
            return result

        # Degree 1 / 一阶
        result[:, 1] = C1 * y   # Y_{1,-1}
        result[:, 2] = C1 * z   # Y_{1,0}
        result[:, 3] = C1 * x   # Y_{1,1}

        if sh_degree < 2:
            return result

        # Degree 2 / 二阶
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z

        result[:, 4] = C2[0] * xy                              # Y_{2,-2}
        result[:, 5] = C2[1] * yz                              # Y_{2,-1}
        result[:, 6] = C2[2] * (2.0 * zz - xx - yy)           # Y_{2,0}
        result[:, 7] = C2[3] * xz                              # Y_{2,1}
        result[:, 8] = C2[4] * (xx - yy)                       # Y_{2,2}

        if sh_degree < 3:
            return result

        # Degree 3 / 三阶
        result[:, 9]  = C3[0] * y * (3.0 * xx - yy)            # Y_{3,-3}
        result[:, 10] = C3[1] * xy * z                          # Y_{3,-2}
        result[:, 11] = C3[2] * y * (4.0 * zz - xx - yy)       # Y_{3,-1}
        result[:, 12] = C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)  # Y_{3,0}
        result[:, 13] = C3[4] * x * (4.0 * zz - xx - yy)       # Y_{3,1}
        result[:, 14] = C3[5] * z * (xx - yy)                  # Y_{3,2}
        result[:, 15] = C3[6] * x * (xx - 3.0 * yy)            # Y_{3,3}

        if sh_degree < 4:
            return result

        # Degree 4 / 四阶
        result[:, 16] = C4[0] * xy * (xx - yy)                 # Y_{4,-4}
        result[:, 17] = C4[1] * yz * (3.0 * xx - yy)           # Y_{4,-3}
        result[:, 18] = C4[2] * xy * (7.0 * zz - 1.0)          # Y_{4,-2}
        result[:, 19] = C4[3] * yz * (7.0 * zz - 3.0)          # Y_{4,-1}
        result[:, 20] = C4[4] * (35.0 * zz * zz - 30.0 * zz + 3.0)  # Y_{4,0}
        result[:, 21] = C4[5] * xz * (7.0 * zz - 3.0)          # Y_{4,1}
        result[:, 22] = C4[6] * (xx - yy) * (7.0 * zz - 1.0)   # Y_{4,2}
        result[:, 23] = C4[7] * xz * (xx - 3.0 * yy)           # Y_{4,3}
        result[:, 24] = C4[8] * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy))  # Y_{4,4}

        return result

    @staticmethod
    def compute_view_directions(
        means3D: torch.Tensor,
        camera_center: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute normalized view directions from Gaussians to camera.
        计算从高斯到相机的归一化观察方向。

        Args:
            means3D: (N, 3) Gaussian centers / 高斯中心
            camera_center: (3,) camera position in world space / 相机在世界空间的位置
        Returns:
            (N, 3) normalized view directions / 归一化观察方向
        """
        dirs = camera_center.unsqueeze(0) - means3D  # (N, 3)
        return torch.nn.functional.normalize(dirs, dim=1)

    def evaluate_rgb(
        self,
        features: torch.Tensor,
        dirs: torch.Tensor,
        active_sh_degree: int,
    ) -> torch.Tensor:
        """
        Convert SH features to RGB colors for given view directions.
        将球谐特征转换为给定观察方向的 RGB 颜色。

        Args:
            features: (N, 3, (degree+1)²) SH coefficients per RGB channel
                     每个 RGB 通道的 SH 系数
            dirs: (N, 3) view directions / 观察方向
            active_sh_degree: current SH degree (0 → only DC) / 当前球谐阶数
        Returns:
            (N, 3) RGB colors (clamped to [0, 1]) / RGB 颜色
        """
        # Evaluate SH basis up to active degree / 计算活跃阶数内的 SH 基函数
        sh_basis = self.evaluate_sh_basis(active_sh_degree, dirs)  # (N, M)

        # features shape: (N, 3, total_coeffs)
        # Result: sum over SH coeffs weighted by basis
        # rgb = Σ_k features[:, :, k] * basis[:, k]
        rgb = torch.sum(
            features[:, :, :sh_basis.shape[1]] * sh_basis.unsqueeze(1),
            dim=2,
        )  # (N, 3)

        # Apply sigmoid to map to [0, 1] (raw SH can produce any real value)
        # 用 S 形函数映射到 [0,1]（原始 SH 可产生任意实数值）
        return torch.sigmoid(rgb)
