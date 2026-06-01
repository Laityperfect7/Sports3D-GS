"""
Video Preprocessing Script / 视频预处理脚本
============================================
Automatically extract high-quality frames from sports videos.
自动从体育运动视频中提取高质量图像帧。

Key features / 核心功能:
  1. Sharpness-based frame scoring / 基于清晰度的帧评分
  2. Adaptive threshold selection / 自适应阈值选取
  3. Duplicate / near-duplicate removal / 重复帧去重
  4. Scene boundary detection / 场景切换检测
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from collections import deque
from typing import List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Import shared utilities (written later) is optional at this stage


# ============================================================================
# Sharpness Evaluation / 清晰度评估
# ============================================================================

def compute_laplacian_variance(image: np.ndarray) -> float:
    """
    Compute sharpness score via Laplacian variance.
    通过拉普拉斯方差计算图像清晰度分数。

    Blurry frames → low variance; sharp frames → high variance.
    模糊帧 → 低方差；清晰帧 → 高方差。

    Args:
        image: BGR image array / BGR 图像数组
    Returns:
        Sharpness score (float) / 清晰度分数
    """
    if image is None:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def compute_brenner_gradient(image: np.ndarray) -> float:
    """
    Brenner gradient focus measure — emphasis on horizontal edges.
    Brenner 梯度聚焦度量 — 强调水平边缘。

    Good for sports: action is usually horizontal motion.
    适合体育视频：运动通常沿水平方向。
    """
    if image is None:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    diff = np.abs(gray[:, 2:] - gray[:, :-2])
    return float(np.mean(diff * diff))


def compute_combined_score(image: np.ndarray) -> float:
    """
    Weighted combination of Laplacian and Brenner scores.
    拉普拉斯和 Brenner 分数的加权组合。
    """
    lap_score = compute_laplacian_variance(image)
    brenner_score = compute_brenner_gradient(image)
    # Normalize roughly so both contribute / 归一化让两者贡献相当
    return 0.6 * lap_score + 0.4 * brenner_score * 100


# ============================================================================
# Duplicate Detection / 重复帧检测
# ============================================================================

def compute_histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute correlation between two HSV histograms.
    计算两张图 HSV 直方图的相关性。

    Returns:
        Correlation value in [0, 1]; 1 = identical.
        相关性值 [0, 1]；1 = 完全相同。
    """
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))


def is_near_duplicate(
    img1: np.ndarray,
    img2: np.ndarray,
    similarity_threshold: float = 0.95
) -> bool:
    """
    Check if two frames are near-duplicates.
    检查两帧是否几乎重复。
    """
    sim = compute_histogram_similarity(img1, img2)
    return sim > similarity_threshold


# ============================================================================
# Scene Boundary Detection / 场景切换检测
# ============================================================================

def detect_scene_cut(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    threshold: float = 0.30
) -> bool:
    """
    Detect hard scene cuts via histogram distance.
    通过直方图距离检测场景硬切换。

    When correlation drops below threshold → scene change.
    相关性低于阈值 → 场景切换。
    """
    sim = compute_histogram_similarity(prev_frame, curr_frame)
    return sim < threshold


# ============================================================================
# Adaptive Frame Extractor / 自适应帧提取器
# ============================================================================

class AdaptiveFrameExtractor:
    """
    Extracts keyframes from video with adaptive quality thresholds.
    自适应质量阈值的视频关键帧提取器。

    Algorithm / 算法:
      1. Scan all frames, compute sharpness scores / 扫描所有帧，计算清晰度分数
      2. Fit a bimodal distribution; separate sharp from blurry / 拟合双峰分布，分离清晰与模糊
      3. Select frames above threshold + deduplicate / 选取高于阈值的帧并去重
    """

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        target_frames: int = 200,
        min_interval: int = 2,
        similarity_threshold: float = 0.92,
        resize_max: int = 1280,
    ):
        """
        Args:
            video_path: Path to input sports video / 输入视频路径
            output_dir: Directory to save extracted frames / 输出帧保存目录
            target_frames: Desired number of output frames / 期望输出帧数
            min_interval: Minimum frame gap to prevent clusters / 最小帧间隔，避免簇集
            similarity_threshold: Max histogram correlation before duplicate flag
                                  直方图相关性阈值，超过视为重复
            resize_max: Maximum dimension for saved frames / 保存帧的最大尺寸
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.target_frames = target_frames
        self.min_interval = min_interval
        self.similarity_threshold = similarity_threshold
        self.resize_max = resize_max

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def read_all_scores(self) -> List[Tuple[int, float]]:
        """
        Scan entire video, compute combined sharpness for every frame.
        遍历整个视频，为每一帧计算综合清晰度分数。

        Returns:
            List of (frame_index, combined_score) tuples.
            包含 (帧索引, 综合分数) 的列表。
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path} / 无法打开视频")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        scores: List[Tuple[int, float]] = []

        print(f"Scanning {total_frames} frames for sharpness... / 正在扫描 {total_frames} 帧的清晰度...")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 5 == 0:  # Sample every 5th for speed / 每5帧采样一次提速
                scaled = self._resize_frame(frame)
                score = compute_combined_score(scaled)
                scores.append((frame_idx, score))

            frame_idx += 1

        cap.release()
        print(f"Sampled {len(scores)} frames. / 采样了 {len(scores)} 帧。")
        return scores

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame so max dimension ≤ resize_max / 将帧的最大尺寸限制为 resize_max"""
        h, w = frame.shape[:2]
        if max(h, w) <= self.resize_max:
            return frame
        scale = self.resize_max / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h))

    def _estimate_sharpness_threshold(self, scores: List[float]) -> float:
        """
        Estimate a threshold separating sharp from blurry frames.
        估算分离清晰帧和模糊帧的阈值。

        Uses Otsu's method on score histogram — finds the bimodal split.
        对分数直方图使用大津法 — 找到双峰分割点。
        """
        arr = np.array(scores, dtype=np.float32)
        if len(arr) < 10:
            return float(np.median(arr))

        # Histogram binning / 直方图分箱
        hist, bin_edges = np.histogram(arr, bins=min(50, len(arr) // 5))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # Otsu thresholding / 大津阈值法
        total = hist.sum()
        current_max, threshold = 0.0, float(np.median(arr))
        sum_all = (hist * bin_centers).sum()
        sum_b, w_b = 0.0, 0.0

        for i in range(len(hist)):
            w_b += hist[i]
            if w_b == 0 or w_b == total:
                continue
            w_f = total - w_b
            sum_b += hist[i] * bin_centers[i]
            m_b = sum_b / w_b
            m_f = (sum_all - sum_b) / w_f
            between = w_b * w_f * (m_b - m_f) ** 2
            if between > current_max:
                current_max = between
                threshold = bin_centers[i]

        return float(threshold + 0.2 * np.std(arr))  # Slightly stricter / 略微更严格

    def extract(self) -> List[Path]:
        """
        Main extraction pipeline / 主提取流程。

        1. Read frame scores / 读取帧分数
        2. Compute adaptive threshold / 计算自适应阈值
        3. Select best frames with dedup / 选择最佳帧并去重
        4. Save selected frames to disk / 保存选中的帧

        Returns:
            List of saved frame paths / 已保存帧的路径列表
        """
        # Step 1: Score all frames / 对所有帧评分
        all_scores = self.read_all_scores()
        raw_scores = [s for _, s in all_scores]

        # Step 2: Adaptive threshold / 自适应阈值
        threshold = self._estimate_sharpness_threshold(raw_scores)
        print(f"Adaptive sharpness threshold: {threshold:.2f} / 自适应清晰度阈值: {threshold:.2f}")

        # Step 3: Filter & deduplicate / 过滤并去重
        cap = cv2.VideoCapture(str(self.video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Build set of candidate indices above threshold / 构建高于阈值的候选帧索引集合
        candidates = sorted([
            idx for idx, score in all_scores if score >= threshold
        ])

        if len(candidates) < self.target_frames:
            # Not enough sharp frames; fill with median-scored frames / 清晰帧不够，用中等分数帧补足
            sorted_indices = sorted(all_scores, key=lambda x: x[1], reverse=True)
            candidates = sorted([idx for idx, _ in sorted_indices[:self.target_frames + 50]])

        print(f"Candidates after threshold: {len(candidates)} / 阈值过滤后的候选帧: {len(candidates)}")

        # Deduplicate with a rolling buffer / 使用滑动缓冲区去重
        selected: List[int] = []
        recent_frames: deque = deque(maxlen=5)

        for frame_idx in candidates:
            # Enforce min interval / 强制最小间隔
            if selected and frame_idx - selected[-1] < self.min_interval:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame = self._resize_frame(frame)

            # Check against recent selected frames / 与近期选中帧比较
            is_dup = any(
                is_near_duplicate(frame, rf, self.similarity_threshold)
                for rf in recent_frames
            )
            if is_dup:
                continue

            selected.append(frame_idx)
            recent_frames.append(frame)

            if len(selected) >= self.target_frames:
                break

        cap.release()

        # Step 4: Save / 保存
        saved_paths: List[Path] = []
        cap = cv2.VideoCapture(str(self.video_path))

        for i, frame_idx in enumerate(selected):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame = self._resize_frame(frame)
            out_path = self.output_dir / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(out_path), frame)
            saved_paths.append(out_path)

        cap.release()

        print(f"\n✔ Extracted {len(saved_paths)} keyframes → {self.output_dir}")
        print(f"  / 提取了 {len(saved_paths)} 个关键帧 → {self.output_dir}")
        return saved_paths


# ============================================================================
# CLI Entry Point / 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract high-quality keyframes from sports video / 从体育视频提取高质量关键帧"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to input video / 输入视频路径"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="./output/frames",
        help="Directory to save frames / 保存帧的输出目录 (default: ./output/frames)"
    )
    parser.add_argument(
        "--target", "-t", type=int, default=200,
        help="Target number of keyframes / 目标关键帧数量 (default: 200)"
    )
    parser.add_argument(
        "--min-interval", type=int, default=2,
        help="Minimum frame index gap / 最小帧索引间隔 (default: 2)"
    )
    parser.add_argument(
        "--sim-threshold", type=float, default=0.92,
        help="Histogram similarity threshold for dedup / 去重相似度阈值 (default: 0.92)"
    )
    parser.add_argument(
        "--resize", type=int, default=1280,
        help="Max dimension for output / 输出图像最大尺寸 (default: 1280)"
    )
    args = parser.parse_args()

    extractor = AdaptiveFrameExtractor(
        video_path=args.input,
        output_dir=args.output,
        target_frames=args.target,
        min_interval=args.min_interval,
        similarity_threshold=args.sim_threshold,
        resize_max=args.resize,
    )
    extractor.extract()


if __name__ == "__main__":
    main()
