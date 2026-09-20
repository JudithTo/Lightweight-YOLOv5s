# Lightweight-YOLOv5s for Nectar-Plant Flower Detection

Lightweight YOLOv5s for UAV real-time flower detection

This repository is modified based on the official YOLOv5 repository (https://github.com/ultralytics/yolov5).
This code implements the seven improvement strategies (M1-M7) described in the manuscript.

## Summary of improvements (M1-M7)
| ID | Improvement | Status | Description |
|---|---|---|---|
| M1 | Coordinate-Attention (CA) module | ✅ Fully implemented | CoordAtt inserted at the end of the backbone, defined in `models/common.py`. Configured in `models/yolov5-ghost.yaml`. |
| M2 | Bidirectional Feature Pyramid Networks | ⚠️ Partial implementation | Weighted feature fusion and shallow cross-layer skip connections are used; a full repeated BiFPN topology is not adopted in the final network. |
| M3 | Multi-scale P2 detection head | ✅ Fully implemented | Added a 4× down-sampling P2 detection branch. Output feature maps: 160×160×255, 80×80×255, and 40×40×255. |
| M4 | Focal-αEIoU loss function | ✅ Integrated | Replaces the original CIoU box-regression loss in `utils/loss.py`. |
| M5 | K-means++ anchor clustering | ✅ Offline | `utils/kmeans_plus_plus_anchor.py` generates dataset-specific anchor priors before training; the generated anchors are copied into the model YAML. |
| M6 | Activation-based FPGM filter pruning | ✅ Implemented | `utils/fpgm_apoz_prune.py` performs APoZ-based filter pruning followed by FPGM structural pruning. |
| M7 | Knowledge distillation | ✅ Integrated | `utils/knowledge_distill.py` and the modified `train.py` support regression-based KD fine-tuning of the pruned M6 student model using the Improved YOLOv5s teacher. |

## Environment setup
Create conda environment:
```bash
conda activate yolov5-nectar
```

Or use pip:
```bash
pip install -r requirements.txt
pip install torch-pruning
```

## Training
Main config for Improved YOLOv5s training: `models/yolov5-ghost.yaml`  
Dataset config: `data/clo.yaml`

```bash
python train.py \
  --cfg models/yolov5-ghost.yaml \
  --data data/clo.yaml \
  --hyp data/hyp.scratch.yaml \
  --epochs 200 \
  --batch-size 4 \
  --img-size 640 \
  --device 0 \
  --weights yolov5s.pt
```

### M5 K-means++ anchor usage (offline before training)
Run this script before training:

```bash
python utils/kmeans_plus_plus_anchor.py \
  --label-dir ./data/wubeizi/train/labels \
  --n-anchor 9
```

Copy the generated anchor list into the `anchors:` block of the model YAML. The original `autoanchor.py` will validate the anchors when training starts.

## M6 Pruning & M7 Knowledge Distillation

These are post-training stages applied after obtaining the Improved YOLOv5s checkpoint.

M6 APoZ-FPGM pruning:
```bash
python utils/fpgm_apoz_prune.py \
  --weights ./weights/Improved_YOLOv5s_teacher.pt \
  --data data/clo.yaml \
  --split val \
  --img-size 640 \
  --batch-size 4 \
  --apoz-threshold 0.95 \
  --prune-ratio 0.40 \
  --output ./weights/M6_pruned.pt \
  --device 0
```

M7 KD fine-tuning:
```bash
python train.py \
  --weights ./weights/M6_pruned.pt \
  --teacher-weights ./weights/Improved_YOLOv5s_teacher.pt \
  --data data/clo.yaml \
  --hyp data/hyp.scratch.yaml \
  --epochs <M7_EPOCHS> \
  --batch-size 4 \
  --img-size 640 \
  --device 0 \
  --kd-v 0.5 \
  --name M7_KD
```

For M6, the APoZ threshold is 95%. For M7, the regression distillation weighting coefficient is `v=0.5`, as described in Supplementary Information Section S3.

## Inference
Custom drone large-image crop & count inference script:

```bash
python scripts/detect.py --weights ./weights/best.pt --source ./images --conf-thres 0.3 --iou-thres 0.45
```

## Model export for deployment
Export the trained model to ONNX format:

```bash
python scripts/export.py --weights ./weights/best.pt --img 640 --simplify
```

## Reproducibility
Key experimental settings: input resolution `640×640`, training epochs `200`, batch size `4`, and SGD optimizer.

Detailed training and deployment configurations are stored in separate documents:

- [Training Configuration](configs/training_configuration.md)
- [Deployment Configuration](configs/deployment_configuration.md)

## Model weights
The trained checkpoint `best.pt` is provided in the repository under `weights/best.pt`.

This checkpoint corresponds to the main Improved-YOLOv5s model with M1 (Coordinate-Attention) and M3 (P2 multi-scale detection head). It is provided as a starting checkpoint; it is not the final M6-pruned or M7-distilled model.

## Dataset limitation
The nectar-plant flower dataset cannot be publicly released due to field-collection constraints. Researchers can adapt the provided code and training configurations to similar small-object detection datasets.

## Reference
- Official YOLOv5: https://github.com/ultralytics/yolov5
- EfficientDet BiFPN reference: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
- Tan M, Pang R, Le Q V. EfficientDet: Scalable and efficient object detection[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
