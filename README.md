# Lightweight‑YOLOv5s for Nectar‑Plant Flower Detection
Lightweight YOLOv5s for UAV real‑time system

This repository is modified based on the official YOLOv5 repository (https://github.com/ultralytics/yolov5).
This code is for nectar‑plant small‑object detection with seven improvement strategies (M1‑M7).

## Summary of improvements (M1‑M7)
| ID | Improvement | Status | Description |
|---|---|---|---|
| M1 | Coordinate‑Attention (CA) module | ✅ Fully implemented | CoordAtt inserted at the end of backbone, defined in `models/common.py`. Configured in `models/yolov5‑ghost.yaml`. |
| M2 | Bidirectional Feature Pyramid Networks | ⚠️ Partial implementation | Weighted Concat operator is implemented in `models/common.py`. **Full repeated BiFPN block topology is NOT adopted in final network**. Our practical feature‑fusion scheme uses multi‑scale shallow cross‑layer skip connections. |
| M3 | Multi‑scale P2 detection head | ✅ Fully implemented | Added 4× down‑sampling P2 detection branch (output 160×160 feature map). Configured in `models/yolov5‑ghost.yaml`. Output feature maps: 160×160×255, 80×80×255, 40×40×255. |
| M4 | Focal‑αEIOU loss function | ⚠️ Experimental prototype | Implemented class inside `utils/loss.py`. **NOT enabled by default**. Manual code modification is required to replace original CIoU loss in `ComputeLoss`. |
| M5 | K‑means++ anchor clustering | ⚠️ Experimental prototype | Offline pre‑processing script `utils/kmeans_plus_plus_anchor.py`. Run before training to generate improved anchor priors, then copy output anchor numbers into model yaml. Original `utils/autoanchor.py` is kept for runtime anchor validation during training. |
| M6 | Activation‑based FPGM filter pruning | ⚠️ Experimental prototype | Offline pruning prototype script: `utils/fpgm_apoz_prune.py`. Workflow: prune trained *.pt checkpoint after training, not integrated into train.py main loop. |
| M7 | Knowledge distillation | ⚠️ Experimental prototype | Distillation loss prototype in `utils/knowledge_distill.py`. Used for accuracy recovery after M6 pruning, executed as separate offline script. |

> Note for experimental prototypes(M4‑M7): These scripts strictly follow the algorithm descriptions in our manuscript, but manual parameter tuning and code adaption are required, they are not enabled by default.
> Important note for model configuration files:
There is NO single yaml file enabling all M1‑M7 improvements at once.
1. M1‑M3 are defined in model yaml files and can be directly loaded by train.py.
2. M4(Focal‑αEIOU loss) needs manual modification inside `utils/loss.py`.
3. M5(K‑means++) runs as an offline script; generated anchors need manually copied into your yaml.
4. M6‑M7 are post‑training offline prototype scripts, executed after model training, not part of network yaml definition.


## Environment setup
Create conda environment:
```bash
conda activate yolov5‑nectar
```

Or use pip:

```
pip install -r requirements.txt
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

## Reproducibility

Key experimental hyper‑parameters: input resolution `640×640`, training epochs `200`, batch‑size `4`, SGD optimizer.

Full detailed training and deployment configurations are stored in separate documents:

- [Training Configuration](configs/training_configuration.md)
- [Deployment Configuration](configs/deployment_configuration.md)

> Note: Actual hyper‑parameters used for training are also available in `data/hyp/hyp.scratch.yaml`.


## M6 Pruning & M7 Knowledge Distillation

These two are post‑training offline operations (run after you finish normal training and obtain `best.pt`):

1. Finish normal training and get `best.pt`;
2. Run `utils/fpgm_apoz_prune.py` for APoZ‑FPGM filter pruning;
3. Run `utils/knowledge_distill.py` for distillation fine‑tuning on pruned model.

## Model weights note
*.pt checkpoint files are excluded from git repository via `.gitignore` to avoid repository bloat.
The trained checkpoint `best.pt` available in GitHub Releases corresponds to our main model with M1 (Coordinate‑Attention) and M3 (P2 multi‑scale detection head).

M4‑M7 are experimental prototype scripts provided in the source repository. To obtain the final compressed model incorporating all M1‑M7 improvements, users need to:
1. Re‑generate anchors using the M5 K‑means++ script and re‑train or fine‑tune;
2. Manually enable M4 Focal‑αEIOU loss by modifying `utils/loss.py`;
3. Perform M6 APoZ‑FPGM pruning on the obtained checkpoint;
4. Run M7 knowledge distillation for accuracy recovery.

These steps require hyper‑parameter tuning. Therefore, the fully‑processed M1‑M7 compressed weight is not provided, while all algorithm implementation scripts are available for reproduction.

See `weights/README.txt` inside repository for more details.


## Dataset limitation

The nectar‑plant flower dataset cannot be publicly released due to field‑collection constraints.
Researchers can build similar small‑object datasets with comparable statistics for reproduction, using our provided training configurations.

## Reference

- Official YOLOv5: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- EfficientDet BiFPN reference: [https://github.com/zylo117/Yet](https://github.com/zylo117/Yet)‑Another‑EfficientDet‑Pytorch
- Tan M, Pang R, Le Q V. EfficientDet: Scalable and efficient object detection[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
