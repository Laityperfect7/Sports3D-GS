"""
Visualization Script / 可视化脚本
====================================
Generate novel-view renderings and comparison plots from trained model.
从训练好的模型生成新视角渲染和对比图。
"""

import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaussian_splatting.scene import GaussianModel, PinholeCamera, CameraParams
from gaussian_splatting.renderer import GaussianRasterizer, SphericalHarmonicsEvaluator
from gaussian_splatting.utils.image_utils import save_tensor_image


def load_model(checkpoint_path: str, device: str = "cuda") -> GaussianModel:
    """
    Load trained Gaussian model from checkpoint.
    从检查点加载训练好的高斯模型。
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    gs_cfg = config.get("gaussian", {})

    model = GaussianModel(
        num_points=0,  # Will be replaced / 会被替换
        sh_degree=gs_cfg.get("sh_degree", 3),
        device=device,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    print(f"Loaded model with {model._xyz.shape[0]} Gaussians / 加载了 {model._xyz.shape[0]} 个高斯")
    return model


def render_orbit(
    model: GaussianModel,
    camera_base: PinholeCamera,
    output_dir: str,
    num_views: int = 36,
    orbit_radius: float = 1.0,
    elevation: float = 0.3,
):
    """
    Render a 360-degree orbital video of the scene.
    渲染场景的 360 度轨道视频。

    Moves the camera in a circle around the scene center.
    绕场景中心旋转相机。

    Args:
        model: Trained Gaussian model / 训练好的高斯模型
        camera_base: Base camera (used for intrinsics) / 基相机（用于内参）
        output_dir: Output directory / 输出目录
        num_views: Number of views in orbit / 轨道视图数量
        orbit_radius: Radius of the circular path / 圆形路径半径
        elevation: Camera height offset / 相机高度偏移
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sh_eval = SphericalHarmonicsEvaluator(sh_degree=model.active_sh_degree)

    for i in range(num_views):
        angle = 2.0 * np.pi * i / num_views
        cam_x = orbit_radius * np.cos(angle)
        cam_y = orbit_radius * elevation
        cam_z = orbit_radius * np.sin(angle) + 2.0  # Base offset / 基础偏移

        # Build camera at new position looking at origin
        # 在新位置构建看向原点的相机
        from gaussian_splatting.scene.camera import PinholeCamera, CameraParams
        import torch

        new_params = CameraParams(
            width=camera_base.params.width,
            height=camera_base.params.height,
            fx=camera_base.params.fx,
            fy=camera_base.params.fy,
            cx=camera_base.params.cx,
            cy=camera_base.params.cy,
        )
        # Set extrinsics for new viewpoint / 设置新视点的外参
        new_params.R = torch.eye(3, device=model.device)
        new_params.T = torch.tensor([cam_x, cam_y, cam_z], device=model.device)

        novel_cam = PinholeCamera(new_params, device=model.device)

        # Render / 渲染
        render_out = render_view(model, novel_cam, sh_eval)
        save_path = output_dir / f"orbit_{i:03d}.png"
        save_tensor_image(render_out, str(save_path))

    print(f"Orbit renderings saved → {output_dir} / 轨道渲染已保存")
    print(f"  To create video: ffmpeg -framerate 30 -i {output_dir}/orbit_%03d.png -c:v libx264 orbit.mp4")


def render_view(
    model: GaussianModel,
    camera: PinholeCamera,
    sh_eval: Optional[SphericalHarmonicsEvaluator] = None,
) -> torch.Tensor:
    """
    Render a single view / 渲染单视图。
    """
    if sh_eval is None:
        sh_eval = SphericalHarmonicsEvaluator(sh_degree=model.active_sh_degree)

    # Create rasterizer for this camera / 为此相机构建光栅化器
    rasterizer = GaussianRasterizer(
        image_height=camera.params.height,
        image_width=camera.params.width,
        tanfovx=0.5 * camera.params.width / camera.params.fx,
        tanfovy=0.5 * camera.params.height / camera.params.fy,
        bg_color=torch.zeros(3, device=model.device),
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
    )

    with torch.no_grad():
        render_out = rasterizer(
            means3D=model.get_xyz,
            means2D=camera.project_points(model.get_xyz),
            opacity=model.get_opacity,
            sh_features=model.get_features,
            scales=model.get_scaling,
            rotations=model.get_rotation,
        )

    return render_out["render"].clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize trained Sports3D-GS model / 可视化训练好的 Sports3D-GS 模型"
    )
    parser.add_argument(
        "--checkpoint", "-c", type=str, required=True,
        help="Path to .pt checkpoint / .pt 检查点路径"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="./output/renders",
        help="Output directory for renderings / 渲染输出目录"
    )
    parser.add_argument(
        "--mode", type=str, default="orbit",
        choices=["orbit", "single", "compare"],
        help="Visualization mode / 可视化模式"
    )
    parser.add_argument(
        "--views", type=int, default=36,
        help="Number of orbit views / 轨道视图数量"
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Render width / 渲染宽度"
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="Render height / 渲染高度"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: cuda or cpu"
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} / 使用设备: {device}")

    # Load model / 加载模型
    model = load_model(args.checkpoint, device)

    # Build camera / 构建相机
    from gaussian_splatting.scene.camera import PinholeCamera, CameraParams

    cam_params = CameraParams(
        width=args.width,
        height=args.height,
        fx=args.width * 0.75,  # Approximate / 近似
        fy=args.width * 0.75,
        cx=args.width / 2.0,
        cy=args.height / 2.0,
    )
    camera = PinholeCamera(cam_params, device=device)

    if args.mode == "orbit":
        render_orbit(model, camera, args.output, num_views=args.views)
    elif args.mode == "single":
        sh_eval = SphericalHarmonicsEvaluator(sh_degree=model.active_sh_degree)
        rendered = render_view(model, camera, sh_eval)
        save_tensor_image(rendered, str(Path(args.output) / "single_view.png"))
        print(f"Saved → {args.output}/single_view.png / 已保存")
    elif args.mode == "compare":
        print("Comparison mode: place reference image in output dir as 'reference.png'")
        sh_eval = SphericalHarmonicsEvaluator(sh_degree=model.active_sh_degree)
        rendered = render_view(model, camera, sh_eval)
        save_tensor_image(rendered, str(Path(args.output) / "rendered.png"))


if __name__ == "__main__":
    main()
