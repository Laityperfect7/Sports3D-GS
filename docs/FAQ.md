# FAQ / 常见问题

## General / 通用

### Q: What's the difference between Sports3D-GS and the original 3DGS?

The key difference: **we replace COLMAP (SfM) with monocular depth estimation**. Original 3DGS requires multi-view images of a static scene processed through COLMAP. Sports3D-GS works with a **single video** of a dynamic scene, using Depth Anything V2 to estimate per-frame depth maps.

### Q: Does this work with moving cameras?

Partially. Slow camera motion (e.g., panning) is OK — depth estimation still works per-frame. Fast camera motion or rapid cuts will degrade results. Best results come from **tripod or steady handheld** footage.

### Q: How long does training take?

| GPU | 30k iterations | FPS (render) |
|-----|---------------|--------------|
| RTX 3060 12GB | ~45 min | 90+ |
| RTX 4090 24GB | ~20 min | 200+ |
| A100 80GB | ~12 min | 400+ |

---

## Technical / 技术

### Q: Can I use a different depth estimation model?

Yes. Edit `scripts/estimate_depth.py` to swap the model. Any model that outputs per-pixel metric or relative depth can work:

- **Metric3D** (metric depth, best accuracy)
- **ZoeDepth** (relative depth, fast)
- **MiDaS** (relative depth, legacy)

### Q: How do I export to mesh?

```bash
# Option 1: Open3D Poisson reconstruction
python -c "
from gaussian_splatting.scene import GaussianModel
import open3d as o3d
import numpy as np

model = GaussianModel(num_points=0, device='cpu')
# Load checkpoint and extract points...

# Option 2: Export PLY and use external tool
# model.save_ply('output.ply')
# Then use MeshLab or CloudCompare
"
```

### Q: Training loss is NaN. What to do?

1. Reduce learning rates by 10x in config
2. Check that input images are properly normalized to [0, 1]
3. Verify CUDA toolchain versions are compatible
4. Try with `--device cpu` to rule out CUDA issues

---

## Contribution / 贡献

### Q: I want to add a new feature. Where do I start?

Check the `gaussian_splatting/` module structure:
- New loss → `optimizer/losses.py`
- New renderer → `renderer/rasterizer.py`
- New data format → `data/dataset.py`
- New config → add a YAML file in `configs/`

### Q: How do I run the tests?

```bash
pip install pytest
python -m pytest tests/ -v
```
