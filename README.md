# Lightweight‑YOLOv5s for Nectar‑Plant Flower Detection
```
## Training
Main config for M1+M3 full‑model training: `models/yolov5‑ghost.yaml`
Dataset config: `data/clo.yaml`
```bash
python train.py \
  --cfg models/yolov5‑ghost.yaml \
  --data data/clo.yaml \
  --hyp data/hyp.scratch.yaml \
  --epochs 200 \
  --batch‑size 4 \
  --img‑size 640 \
  --device 0 \
  --weights yolov5s.pt
```

### Training Configuration

表格

| Parameter | Value |
| --- | --- |
| Random seed | 1 |
| Optimizer | SGD |
| Epochs | 200 |
| Batch size | 4 |
| Initial weights | YOLOv5s pretrained |
| Input size | 640 × 640 |
| Initial learning rate | 0.01 |
| Final learning rate factor | 0.2 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Warm‑up epochs | 3.0 |
| Box loss gain | 0.05 |
| Classification loss gain | 0.5 |
| Objectness loss gain | 1.0 |

### M5 K‑means++ anchor usage (off‑line before training)

Run this script **before starting training** to generate optimized anchor boxes for your dataset labels:

```
python utils/kmeans_plus_plus_anchor.py --label‑dir ./data/wubeizi/train/labels
```

Copy the printed anchor list and paste into `anchors:` block inside your model yaml file.
The original `autoanchor.py` will automatically validate anchor quality when train.py starts.

## Inference

Custom drone large‑image crop & count inference script:

```
python scripts/detect.py --weights ./weights/best.pt --source ./images --conf‑thres 0.3 --iou‑thres 0.45
```

## Model export for deployment

Export trained model to ONNX format for deployment:

```
python scripts/export.py --weights ./weights/best.pt --img 640 --simplify
```

### Deployment Environment

#### Setting Configuration

表格

| Item | Configuration |
| --- | --- |
| Device | NVIDIA Jetson Xavier NX |
| Operating system | Ubuntu 18.04 |
| Python version | 3.8 |
| Inference framework | TensorRT |
| Inference precision | FP16 |
| Input resolution | 640 × 640 |
| Batch size | 1 |
| Power mode | Maximum performance mode |
| Acceleration strategies | TensorRT optimization, FP16 precision calibration, operator fusion, and multi‑threaded CPU/GPU pipeline scheduling |

## M6 Pruning & M7 Knowledge Distillation

These two are post‑training offline operations (run after you finish normal training and obtain `best.pt`):

1. Finish normal training and get `best.pt`;
2. Run `utils/fpgm_apoz_prune.py` for APoZ‑FPGM filter pruning;
3. Run `utils/knowledge_distill.py` for distillation fine‑tuning on pruned model.

## Model weights note

*.pt checkpoint files are excluded from git repository via `.gitignore` to avoid repository bloat.
Original trained `best.pt` and pruned‑distilled lightweight weights are available in **GitHub Releases assets** of this repository.
See `weights/README.txt` inside repository for more details.

## Dataset limitation

The nectar‑plant flower dataset cannot be publicly released due to field‑collection constraints.
Researchers can build similar small‑object datasets with comparable statistics for reproduction, using our provided training configurations.

## Reference

- Official YOLOv5: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- EfficientDet BiFPN reference: [https://github.com/zylo117/Yet](https://github.com/zylo117/Yet)‑Another‑EfficientDet‑Pytorch
- Tan M, Pang R, Le Q V. EfficientDet: Scalable and efficient object detection[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
