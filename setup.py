"""
Sports3D-GS Installation Script / 安装脚本
============================================
Single-view sports video 3D reconstruction via 3D Gaussian Splatting.
基于 3D 高斯泼溅的单视角体育视频三维重建。
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description / 读取 README 作为长描述
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements / 读取依赖
req_path = Path(__file__).parent / "requirements.txt"
requirements = []
if req_path.exists():
    requirements = [
        line.strip()
        for line in req_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

setup(
    name="sports3d-gs",
    version="0.1.0",
    author="Sports3D-GS Team",
    description="3D Gaussian Splatting for Single-View Sports Video Reconstruction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/Sports3D-GS",
    packages=find_packages(include=["gaussian_splatting", "gaussian_splatting.*"]),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "pre-commit>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sports3d-preprocess=scripts.preprocess:main",
            "sports3d-train=scripts.train:main",
            "sports3d-visualize=scripts.visualize:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
