# Training Configuration
The training configuration used for the Improved YOLOv5s model is summarized below.

| Setting | Value |
|---|---|
| Random seed | 1 |
| Optimizer | SGD |
| Epochs | 200 |
| Batch size | 4 |
| Input resolution | 640 × 640 |
| Initial weights | YOLOv5s pretrained weights |
| Initial learning rate (lr0) | 0.01 |
| Final learning rate factor (lrf) | 0.2 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Warm‑up epochs | 3.0 |
| Warm‑up momentum | 0.8 |
| Warm‑up bias learning rate | 0.1 |
| Box loss gain | 0.05 |
| Classification loss gain | 0.5 |
| Objectness loss gain | 1.0 |
| IoU training threshold | 0.20 |
| Anchor threshold | 4.0 |
| HSV augmentation (h/s/v) | 0.015 / 0.7 / 0.4 |
| Translation | 0.1 |
| Scale | 0.5 |
| Horizontal flip | 0.5 |
| Mosaic | 1.0 |
