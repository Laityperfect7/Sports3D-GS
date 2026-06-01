"""
Training Entry Point / 训练入口脚本
=====================================
Main entry point to launch 3DGS training on preprocessed sports video data.
启动 3DGS 训练的主入口，读取预处理后的体育视频数据。
"""

import argparse
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaussian_splatting.data.dataset import SportsVideoDataset
from gaussian_splatting.optimizer.trainer import GaussianSplattingTrainer


def load_config(config_path: str) -> dict:
    """Load YAML config / 加载 YAML 配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Train 3D Gaussian Splatting on sports video / 在体育视频上训练 3D 高斯泼溅"
    )
    parser.add_argument(
        "--config", "-c", type=str, default="configs/default.yaml",
        help="Path to YAML config file / YAML 配置文件路径"
    )
    parser.add_argument(
        "--frames", "-f", type=str, default=None,
        help="Override frame directory / 覆盖帧目录"
    )
    parser.add_argument(
        "--depth", "-d", type=str, default=None,
        help="Override depth directory / 覆盖深度目录"
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=None,
        help="Override number of training iterations / 覆盖训练迭代次数"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Override output/log directory / 覆盖输出/日志目录"
    )
    args = parser.parse_args()

    # Load config / 加载配置
    config = load_config(args.config)

    # Override from CLI / 从命令行覆盖
    if args.frames:
        config["dataset"]["frame_dir"] = args.frames
    if args.depth:
        config["dataset"]["depth_dir"] = args.depth
    if args.iterations:
        config["train"]["iterations"] = args.iterations
    if args.output:
        config["logging"]["log_dir"] = args.output

    # Load dataset / 加载数据集
    dataset_cfg = config.get("dataset", {})
    dataset = SportsVideoDataset(
        frame_dir=dataset_cfg.get(
            "frame_dir",
            str(Path(args.config).parent.parent / "output" / "frames")
        ),
        depth_dir=dataset_cfg.get("depth_dir"),
        mask_dir=dataset_cfg.get("mask_dir"),
        resolution=(
            dataset_cfg.get("resolution", [1280, 720])[0],
            dataset_cfg.get("resolution", [1280, 720])[1],
        ),
        train_split=dataset_cfg.get("train_split", 0.9),
        white_background=dataset_cfg.get("white_background", False),
    )

    # Train / 训练
    trainer = GaussianSplattingTrainer(config=config, dataset=dataset)
    trainer.train()


if __name__ == "__main__":
    main()
