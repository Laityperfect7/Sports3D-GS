"""
3D Gaussian Splatting Trainer / 3D 高斯泼溅训练器
=====================================================
Core training loop that optimizes Gaussian parameters
to reconstruct a 3D scene from single-view video frames.

从单视角视频帧优化高斯参数以重建三维场景的核心训练循环。

Training pipeline / 训练流程:
  1. For each iteration, sample a camera viewpoint (frame)
     每次迭代采样一个相机视点（帧）
  2. Render Gaussians from that viewpoint / 从该视点渲染高斯
  3. Compute L1 + SSIM loss against ground truth / 计算 L1+SSIM 损失
  4. Backpropagate, update params, densify periodically
     反向传播，更新参数，周期性密度控制
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
from collections import defaultdict
import time
import sys
from datetime import datetime

# Import local modules / 导入本地模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from gaussian_splatting.scene import GaussianModel, PinholeCamera, CameraParams
from gaussian_splatting.renderer import GaussianRasterizer, SphericalHarmonicsEvaluator
from gaussian_splatting.optimizer.losses import combined_loss
from gaussian_splatting.optimizer.densification import DensificationController
from gaussian_splatting.data.dataset import SportsVideoDataset


class GaussianSplattingTrainer:
    """
    Manages the full training pipeline for 3D Gaussian Splatting.
    管理 3D 高斯泼溅的完整训练流程。

    Usage / 用法:
        trainer = GaussianSplattingTrainer(config)
        trainer.train()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        dataset: SportsVideoDataset,
    ):
        self.config = config
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training hyperparams / 训练超参数
        train_cfg = config.get("train", {})
        self.num_iterations = train_cfg.get("iterations", 30000)
        self.batch_size = train_cfg.get("batch_size", 1)
        self.save_interval = train_cfg.get("save_interval", 5000)
        self.eval_interval = train_cfg.get("eval_interval", 1000)
        self.print_interval = config.get("logging", {}).get("print_interval", 50)

        # Loss config / 损失配置
        loss_cfg = config.get("loss", {})
        self.lambda_l1 = loss_cfg.get("lambda_l1", 0.8)
        self.lambda_ssim = loss_cfg.get("lambda_ssim", 0.2)

        # Build components / 构建组件
        self._build_camera()
        self._build_model()
        self._build_renderer()
        self._build_optimizer()
        self._build_scheduler()
        self._build_densifier()

        # SH evaluator / 球谐求值器
        sh_degree = config.get("gaussian", {}).get("sh_degree", 3)
        self.sh_evaluator = SphericalHarmonicsEvaluator(sh_degree=sh_degree)

        # Logging / 日志
        self.log_dir = Path(config.get("logging", {}).get("log_dir", "output/logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_active = False

        # Try wandb / 尝试初始化 wandb
        wandb_project = config.get("logging", {}).get("wandb_project", "")
        if wandb_project:
            try:
                import wandb
                wandb.init(project=wandb_project, config=config)
                self.wandb_active = True
            except (ImportError, Exception):
                pass

        print(f"Trainer initialized on {self.device} / 训练器初始化完成，设备: {self.device}")
        print(f"  Initial Gaussians: {self.gaussian_model._xyz.shape[0]} / 初始高斯数")
        print(f"  Training frames: {len(self.dataset)} / 训练帧数")
        print(f"  Iterations: {self.num_iterations} / 迭代次数")

    def _build_camera(self) -> None:
        """Initialize camera from config or dataset / 从配置或数据集中初始化相机"""
        cam_cfg = self.config.get("camera", {})
        H, W = self.dataset.image_size

        self.camera_params = CameraParams(
            width=W,
            height=H,
            fx=cam_cfg.get("fx", 1200.0),
            fy=cam_cfg.get("fy", 1200.0),
            cx=cam_cfg.get("cx", W / 2.0),
            cy=cam_cfg.get("cy", H / 2.0),
            znear=self.config.get("render", {}).get("near_clip", 0.01),
            zfar=self.config.get("render", {}).get("far_clip", 100.0),
        )
        self.camera = PinholeCamera(self.camera_params, device=str(self.device))

    def _build_model(self) -> None:
        """Initialize Gaussian model from depth-based point cloud / 从深度点云初始化高斯模型"""
        gs_cfg = self.config.get("gaussian", {})
        num_init = gs_cfg.get("num_init", 50000)

        self.gaussian_model = GaussianModel(
            num_points=num_init,
            sh_degree=gs_cfg.get("sh_degree", 3),
            device=str(self.device),
        )

        # If we have a pre-computed point cloud from depth, load it
        # 如果有从深度图预计算的点云，加载它
        if hasattr(self.dataset, "get_initial_point_cloud"):
            pts, colors = self.dataset.get_initial_point_cloud()
            if pts is not None and len(pts) > 0:
                self._initialize_from_point_cloud(pts, colors)

    def _initialize_from_point_cloud(
        self, pts: torch.Tensor, colors: torch.Tensor
    ) -> None:
        """
        Replace random initialization with point cloud from depth.
        用深度点云替换随机初始化。

        Places Gaussians at each 3D point location with color from the image.
        在每个三维点位置放置高斯，颜色来自图像。
        """
        N_pts = min(pts.shape[0], self.gaussian_model._xyz.shape[0])
        self.gaussian_model._xyz.data[:N_pts] = pts[:N_pts].float().to(self.device)

        # Set DC color from point cloud colors / 从点云颜色设置直流分量
        if colors is not None:
            self.gaussian_model._features_dc.data[:N_pts, :, 0] = \
                colors[:N_pts].float().to(self.device)

        # Estimate initial scaling from nearest-neighbor distances
        # 从最近邻距离估算初始缩放
        if N_pts > 1:
            dists = torch.cdist(pts[:N_pts].float(), pts[:N_pts].float())
            dists[dists < 1e-6] = float("inf")
            nn_dists = dists.min(dim=1).values
            mean_dist = nn_dists.mean().clamp(min=1e-3)
            # log scale such that exp(log_scale) ≈ mean_dist
            # 对数缩放使得 exp(log_scale) ≈ mean_dist
            self.gaussian_model._scaling.data[:N_pts] = torch.log(mean_dist).to(self.device)

        print(f"Initialized {N_pts} Gaussians from point cloud / 从点云初始化了 {N_pts} 个高斯")

    def _build_renderer(self) -> None:
        """Initialize Gaussian rasterizer / 初始化高斯光栅化器"""
        render_cfg = self.config.get("render", {})
        cam = self.camera_params

        self.tanfovx = 0.5 * cam.width / cam.fx
        self.tanfovy = 0.5 * cam.height / cam.fy

        bg_color = torch.tensor(
            [1.0, 1.0, 1.0] if self.config.get("dataset", {}).get("white_background", False)
            else [0.0, 0.0, 0.0],
            device=self.device,
        )

        self.rasterizer = GaussianRasterizer(
            image_height=cam.height,
            image_width=cam.width,
            tanfovx=self.tanfovx,
            tanfovy=self.tanfovy,
            bg_color=bg_color,
            viewmatrix=self.camera.world_view_transform,
            projmatrix=self.camera.full_proj_transform,
            sh_degree=self.config.get("gaussian", {}).get("sh_degree", 3),
            near_clip=render_cfg.get("near_clip", 0.01),
            far_clip=render_cfg.get("far_clip", 100.0),
        )
        self.camera_center = self.camera.camera_center

    def _build_optimizer(self) -> None:
        """Create Adam optimizer with per-param-group learning rates / 创建分组学习率的 Adam 优化器"""
        opt_cfg = self.config.get("optim", {})
        gs_cfg = self.config.get("gaussian", {})

        self.optimizer = torch.optim.Adam(
            params=[],  # Params added via gaussian_model.add_to_optimizer
            lr=0.0,
            betas=(opt_cfg.get("adam_beta1", 0.9), opt_cfg.get("adam_beta2", 0.999)),
            eps=opt_cfg.get("adam_eps", 1e-15),
        )
        self.gaussian_model.add_to_optimizer(self.optimizer)

    def _build_scheduler(self) -> None:
        """
        Exponential LR scheduler for Gaussian positions.
        高斯位置的指数学习率调度。

        Position LR decays from initial to ~1% over training.
        位置学习率在训练过程中从初始衰减到约 1%。
        """
        gs_cfg = self.config.get("gaussian", {})

        def lr_lambda(step):
            init_lr = gs_cfg.get("position_lr", 0.00016)
            max_steps = gs_cfg.get("position_lr_max_steps", 30000)
            return max(0.01, (1.0 - step / max_steps))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=[lr_lambda] * len(self.optimizer.param_groups)
        )

    def _build_densifier(self) -> None:
        """Initialize the adaptive density controller / 初始化自适应密度控制器"""
        gs_cfg = self.config.get("gaussian", {})
        train_cfg = self.config.get("train", {})

        self.densifier = DensificationController(
            densify_from_iter=train_cfg.get("densify_from_iter", 500),
            densify_until_iter=train_cfg.get("densify_until", 15000),
            densification_interval=train_cfg.get("densify_interval", 100),
            opacity_reset_interval=train_cfg.get("opacity_reset_interval", 3000),
            densify_grad_threshold=0.0002,
            densify_opacity_threshold=0.005,
            prune_opacity_threshold=0.005,
        )

    def train(self) -> None:
        """
        Main training loop / 主训练循环。

        For single-view sports video, each "viewpoint" is a different frame/time.
        对于单视角体育视频，每个"视点"是不同的帧/时间。
        """
        print("\n" + "=" * 60)
        print("Starting 3DGS Training / 开始 3DGS 训练")
        print("=" * 60 + "\n")

        # Progress tracking / 进度追踪
        progress_bar = None
        try:
            from tqdm import tqdm
            progress_bar = tqdm(
                total=self.num_iterations,
                desc="Training / 训练中",
                unit="step",
                dynamic_ncols=True,
            )
        except ImportError:
            pass

        # Training metrics / 训练指标
        metrics = defaultdict(list)
        best_psnr = 0.0
        ema_loss = 0.0
        start_time = time.time()

        for iteration in range(1, self.num_iterations + 1):
            # ---- Sample a training frame / 采样训练帧 ----
            image, frame_idx = self.dataset.sample_random()
            image = image.to(self.device)  # (3, H, W)

            # ---- Render Gaussians / 渲染高斯 ----
            render_out = self.rasterizer(
                means3D=self.gaussian_model.get_xyz,
                means2D=self.camera.project_points(self.gaussian_model.get_xyz),
                opacity=self.gaussian_model.get_opacity,
                sh_features=self.gaussian_model.get_features,
                scales=self.gaussian_model.get_scaling,
                rotations=self.gaussian_model.get_rotation,
            )
            rendered = render_out["render"]  # (3, H, W)

            # ---- Compute loss / 计算损失 ----
            loss, loss_info = combined_loss(
                rendered.unsqueeze(0),
                image.unsqueeze(0),
                lambda_l1=self.lambda_l1,
                lambda_ssim=self.lambda_ssim,
            )

            # ---- Backprop / 反向传播 ----
            loss.backward()

            # Accumulate gradients for densification / 累积梯度用于密度控制
            with torch.no_grad():
                self.gaussian_model.xyz_gradient_accum += \
                    self.gaussian_model._xyz.grad.norm(dim=1, keepdim=True)
                self.gaussian_model.denom += 1.0

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

            # ---- Densification / 密度控制 ----
            if self.densifier.should_densify(iteration):
                extent = self.gaussian_model.get_xyz.norm(dim=1).max().item()
                stats = self.densifier.densify_and_prune(
                    self.gaussian_model, self.optimizer, iteration, max(extent, 1.0)
                )
                if any(v > 0 for v in stats.values()):
                    print(f"\n  [Iter {iteration}] Densify: {stats} / 密度控制完成")

            if self.densifier.should_reset_opacity(iteration):
                self.densifier.reset_opacity(self.gaussian_model)

            # ---- Increase SH degree / 提升球谐阶数 ----
            if iteration % 1000 == 0:
                self.gaussian_model.oneup_SH_degree()

            # ---- Logging / 日志 ----
            metrics["loss"].append(loss_info["total_loss"])
            metrics["l1"].append(loss_info["l1_loss"])
            metrics["ssim"].append(loss_info["ssim_loss"])
            ema_loss = 0.9 * ema_loss + 0.1 * loss_info["total_loss"]

            if iteration % self.print_interval == 0:
                elapsed = time.time() - start_time
                num_gaussians = self.gaussian_model._xyz.shape[0]
                print(
                    f"\r[Iter {iteration:6d}/{self.num_iterations}] "
                    f"Loss: {ema_loss:.4f} | "
                    f"#Gaussians: {num_gaussians:6d} | "
                    f"SH: {self.gaussian_model.active_sh_degree} | "
                    f"Elapsed: {elapsed:.1f}s"
                    f"  /  损失: {ema_loss:.4f} | 高斯数: {num_gaussians:6d}"
                )

            if progress_bar is not None:
                progress_bar.set_postfix(
                    loss=f"{ema_loss:.4f}",
                    gauss=f"{self.gaussian_model._xyz.shape[0]}",
                )
                progress_bar.update(1)

            # ---- Evaluation / 评估 ----
            if iteration % self.eval_interval == 0:
                psnr_val = self._evaluate()
                if psnr_val > best_psnr:
                    best_psnr = psnr_val
                    self._save_checkpoint("best.pt")
                metrics["psnr"].append(psnr_val)

            # ---- Save checkpoint / 保存检查点 ----
            if iteration % self.save_interval == 0:
                self._save_checkpoint(f"checkpoint_{iteration:06d}.pt")

        # End of training / 训练结束
        elapsed_total = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"Training complete! / 训练完成！")
        print(f"  Total time: {elapsed_total:.1f}s / 总时间")
        print(f"  Final #Gaussians: {self.gaussian_model._xyz.shape[0]} / 最终高斯数")
        print(f"  Best PSNR: {best_psnr:.2f} dB / 最佳 PSNR")
        print(f"  Output: {self.log_dir} / 输出目录")
        print(f"{'=' * 60}")

        if progress_bar is not None:
            progress_bar.close()

        # Final save / 最终保存
        self._save_checkpoint("final.pt")
        self.gaussian_model.save_ply(self.log_dir / "gaussians.ply")

    def _evaluate(self) -> float:
        """
        Evaluate PSNR on a held-out validation frame.
        在保留的验证帧上评估 PSNR。
        """
        self.gaussian_model.eval()
        with torch.no_grad():
            # Get validation frame / 获取验证帧
            if hasattr(self.dataset, "sample_val"):
                val_image, _ = self.dataset.sample_val()
            else:
                val_image, _ = self.dataset.sample_random()

            val_image = val_image.to(self.device)

            render_out = self.rasterizer(
                means3D=self.gaussian_model.get_xyz,
                means2D=self.camera.project_points(self.gaussian_model.get_xyz),
                opacity=self.gaussian_model.get_opacity,
                sh_features=self.gaussian_model.get_features,
                scales=self.gaussian_model.get_scaling,
                rotations=self.gaussian_model.get_rotation,
            )
            rendered = render_out["render"]

            # PSNR / 峰值信噪比
            mse = torch.mean((rendered - val_image) ** 2)
            psnr = 20.0 * torch.log10(1.0 / (torch.sqrt(mse) + 1e-8))

            print(f"\n  [Eval iter {self.gaussian_model.xyz_gradient_accum.shape[0]}] PSNR: {psnr.item():.2f} dB")

        self.gaussian_model.train()
        return psnr.item()

    def _save_checkpoint(self, filename: str) -> None:
        """Save model and optimizer state / 保存模型和优化器状态"""
        ckpt_dir = self.log_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "model_state_dict": self.gaussian_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
        }
        path = ckpt_dir / filename
        torch.save(ckpt, path)
        print(f"  Saved checkpoint → {path} / 保存检查点")

    def render_novel_view(
        self, camera: PinholeCamera, output_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Render the scene from a novel viewpoint.
        从新视点渲染场景。

        Args:
            camera: Novel-view camera / 新视角相机
            output_path: Optional save path / 可选保存路径
        Returns:
            (3, H, W) rendered RGB image / 渲染的 RGB 图像
        """
        self.gaussian_model.eval()

        # Compute view-dependent colors from SH
        dirs = SphericalHarmonicsEvaluator.compute_view_directions(
            self.gaussian_model.get_xyz,
            camera.camera_center,
        )
        colors = self.sh_evaluator.evaluate_rgb(
            self.gaussian_model.get_features,
            dirs,
            self.gaussian_model.active_sh_degree,
        )

        with torch.no_grad():
            render_out = self.rasterizer(
                means3D=self.gaussian_model.get_xyz,
                means2D=camera.project_points(self.gaussian_model.get_xyz),
                opacity=self.gaussian_model.get_opacity,
                sh_features=self.gaussian_model.get_features,
                scales=self.gaussian_model.get_scaling,
                rotations=self.gaussian_model.get_rotation,
                colors_precomp=colors,
            )

        rendered = render_out["render"].clamp(0, 1)

        if output_path:
            from torchvision.utils import save_image
            save_image(rendered, output_path)
            print(f"Rendered novel view → {output_path} / 渲染新视角")

        return rendered
