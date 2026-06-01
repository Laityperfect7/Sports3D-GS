"""
Unit tests for video preprocessing / 视频预处理单元测试
"""

import sys
import os
import unittest
import cv2
import numpy as np
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.preprocess import (
    compute_laplacian_variance,
    compute_brenner_gradient,
    compute_combined_score,
    compute_histogram_similarity,
    is_near_duplicate,
    detect_scene_cut,
)


class TestSharpnessMetrics(unittest.TestCase):
    """Test sharpness scoring functions / 测试清晰度评分函数"""

    def setUp(self):
        # Create a synthetic sharp image (edge) / 创建合成的清晰图像
        self.sharp = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(self.sharp, (25, 25), (75, 75), (255, 255, 255), -1)

        # Create a synthetic blurry image / 创建合成的模糊图像
        self.blurry = cv2.GaussianBlur(self.sharp, (31, 31), 10)

    def test_laplacian_sharp_gt_blurry(self):
        """Sharp image should have higher Laplacian variance than blurry / 清晰图应有更高的拉普拉斯方差"""
        sharp_score = compute_laplacian_variance(self.sharp)
        blurry_score = compute_laplacian_variance(self.blurry)
        self.assertGreater(sharp_score, blurry_score,
                           "Sharp image should score higher than blurry / 清晰图分数应高于模糊图")

    def test_brenner_sharp_gt_blurry(self):
        """Brenner gradient should also prefer sharp images / Brenner 梯度也应偏好清晰图像"""
        sharp_score = compute_brenner_gradient(self.sharp)
        blurry_score = compute_brenner_gradient(self.blurry)
        self.assertGreater(sharp_score, blurry_score)

    def test_combined_score_range(self):
        """Combined score should be non-negative / 综合分数应非负"""
        score = compute_combined_score(self.sharp)
        self.assertGreaterEqual(score, 0.0)

    def test_none_input(self):
        """None input should return 0.0 / None 输入应返回 0.0"""
        self.assertEqual(compute_laplacian_variance(None), 0.0)
        self.assertEqual(compute_brenner_gradient(None), 0.0)
        self.assertEqual(compute_combined_score(None), 0.0)


class TestHistogramSimilarity(unittest.TestCase):
    """Test histogram-based frame comparison / 测试直方图帧比较"""

    def setUp(self):
        self.img1 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        self.img2 = self.img1.copy()
        self.img3 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    def test_identical_images(self):
        """Identical images should have correlation ~1.0 / 相同图像相关性应接近 1.0"""
        sim = compute_histogram_similarity(self.img1, self.img2)
        self.assertGreater(sim, 0.99)

    def test_different_images(self):
        """Random different images should have lower correlation / 随机不同图像相关性应较低"""
        sim = compute_histogram_similarity(self.img1, self.img3)
        self.assertLess(sim, 0.99)

    def test_near_duplicate_detection(self):
        """is_near_duplicate should work / is_near_duplicate 应正常工作"""
        self.assertTrue(is_near_duplicate(self.img1, self.img2, 0.90))
        self.assertFalse(is_near_duplicate(self.img1, self.img3, 0.99))


class TestSceneCutDetection(unittest.TestCase):
    """Test scene boundary detection / 测试场景切换检测"""

    def setUp(self):
        self.scene_a = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        self.scene_b = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        # Make them very different / 让它们非常不同
        self.scene_a[:, :, :] = [100, 50, 200]  # Purple-ish
        self.scene_b[:, :, :] = [200, 200, 50]  # Yellow-ish

    def test_scene_cut_detected(self):
        """Scene cut between very different images should be detected / 差异大的图应检测为场景切换"""
        result = detect_scene_cut(self.scene_a, self.scene_b, threshold=0.5)
        self.assertTrue(result)

    def test_no_scene_cut_same_image(self):
        """Same image should not trigger scene cut / 相同图像不应触发场景切换"""
        result = detect_scene_cut(self.scene_a, self.scene_a, threshold=0.5)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
