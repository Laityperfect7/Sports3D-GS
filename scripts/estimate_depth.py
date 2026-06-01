"""
Depth Estimation Script / 深度估计脚本
========================================
Estimates monocular depth maps from extracted frames
using Depth Anything V2 (via HuggingFace Transformers).

使用 Depth Anything V2（通过 HuggingFace Transformers）
从提取的帧中估计单目深度图。
"""

import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from tqdm import tqdm


def load_depth_anything(model_size: str = "vitl", device: str = "cuda"):
    """
    Load Depth Anything V2 model from HuggingFace Hub.
    从 HuggingFace Hub 加载 Depth Anything V2 模型。

    Args:
        model_size: Model variant: vits | vitb | vitl | vitg
                    模型变体
        device: "cuda" or "cpu"
    Returns:
        depth_model, transform function
    """
    print(f"Loading Depth Anything V2 ({model_size})... / 加载深度模型...")

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    import torch

    model_name = f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}-hf"

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)
    model.eval()

    def predict(image_bgr: np.ndarray) -> np.ndarray:
        """
        Run depth estimation on a single BGR image.
        对单张 BGR 图像运行深度估计。
        """
        # BGR → RGB / BGR 转 RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        inputs = processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs.predicted_depth.squeeze().cpu().numpy()

        return pred

    return model, predict


def main():
    parser = argparse.ArgumentParser(
        description="Estimate depth maps for extracted frames / 为提取的帧估计深度图"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Directory of extracted frames / 提取帧目录"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="./output/depth",
        help="Output directory for depth maps / 深度图输出目录"
    )
    parser.add_argument(
        "--model", type=str, default="vitl",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="Depth Anything model size / 模型尺寸 (default: vitl)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: cuda or cpu"
    )
    parser.add_argument(
        "--max-size", type=int, default=1280,
        help="Max image dimension for processing / 处理时的最大图像尺寸"
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} / 使用设备: {device}")

    # Load model / 加载模型
    model, predict_fn = load_depth_anything(args.model, device)

    # Collect frame paths / 收集帧路径
    input_dir = Path(args.input)
    frame_paths = sorted(
        list(input_dir.glob("*.png")) +
        list(input_dir.glob("*.jpg")) +
        list(input_dir.glob("*.jpeg"))
    )
    if not frame_paths:
        print(f"No frames found in {input_dir} / 未找到帧文件")
        return

    # Create output dir / 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(frame_paths)} frames... / 正在处理 {len(frame_paths)} 帧...")

    for frame_path in tqdm(frame_paths, desc="Depth estimation / 深度估计"):
        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"  Warning: cannot read {frame_path.name} / 无法读取")
            continue

        # Resize for efficiency / 调整大小提高效率
        h, w = image.shape[:2]
        scale = 1.0
        if max(h, w) > args.max_size:
            scale = args.max_size / max(h, w)
            image = cv2.resize(
                image,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        # Predict depth / 预测深度
        depth = predict_fn(image)

        # Restore original resolution / 恢复原始分辨率
        if scale != 1.0:
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        # Save as .npy (float32 meters) / 保存为 .npy（float32，米）
        out_path = output_dir / f"{frame_path.stem}_depth.npy"
        np.save(str(out_path), depth.astype(np.float32))

        # Also save a visualization / 同时保存可视化图
        depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_vis = (depth_vis * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        vis_path = output_dir / f"{frame_path.stem}_depth_viz.png"
        cv2.imwrite(str(vis_path), depth_vis)

    print(f"\n✔ Depth maps saved → {output_dir} / 深度图已保存")


if __name__ == "__main__":
    main()
