# Model Weights

The trained checkpoint `best.pt` is provided in the repository at:

```text
weights/best.pt
```

This checkpoint corresponds to the main Improved-YOLOv5s model with M1 (Coordinate-Attention) and M3 (P2 multi-scale detection head).

The repository also provides the source code for M4-M7. The fully processed M1-M7 compressed checkpoint is not provided; users can obtain it by following the training, M6 pruning, and M7 knowledge-distillation steps described in the root `README.md`.
