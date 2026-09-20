"""
M6 Activation-based FPGM filter pruning with APoZ.

This script implements the M6 workflow described in the manuscript:
    1. Load a trained Improved YOLOv5s checkpoint.
    2. Collect APoZ (Average Percentage of Zeros) statistics from
       validation images.
    3. Directly prune filters whose APoZ is greater than the fixed 95%
       threshold.
    4. Apply FPGM-style geometric-median pruning to the remaining filters
       until the requested per-layer pruning ratio is reached.
    5. Apply graph-aware structural pruning so dependent layers are updated
       consistently.
    6. Save a complete PyTorch checkpoint that can be loaded directly as the
       M6 student model in M7 knowledge-distillation fine-tuning.

The pruning stage is performed after training the Improved YOLOv5s model
and before M7 knowledge-distillation fine-tuning.

Dependency:
    pip install torch-pruning

Important:
    The APoZ threshold is fixed at 0.95 to match the manuscript. The default
    The default FPGM target pruning ratio is 0.40; pass --prune-ratio
    explicitly when using a different experimental setting.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

try:
    import torch_pruning as tp
except ImportError as exc:  # pragma: no cover - import-time environment check
    raise ImportError(
        "M6 structural pruning requires the 'torch-pruning' package. "
        "Install it with: pip install torch-pruning"
    ) from exc

from models.experimental import attempt_load
from models.yolo import Detect

LOGGER = logging.getLogger("M6_APOZ_FPGM")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M6 APoZ + FPGM structural pruning")
    parser.add_argument("--weights", type=str, required=True,
                        help="trained Improved YOLOv5s checkpoint before pruning")
    parser.add_argument("--data", type=str, required=True,
                        help="dataset YAML, e.g. data/clo.yaml")
    parser.add_argument("--split", type=str, default="val", choices=("train", "val"),
                        help="dataset split used for APoZ calibration (default: val)")
    parser.add_argument("--img-size", type=int, default=640,
                        help="square model input size (default: 640)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="APoZ calibration batch size (default: 4)")
    parser.add_argument("--apoz-images", type=int, default=0,
                        help="maximum number of calibration images; 0 uses all split images")
    parser.add_argument("--apoz-threshold", type=float, default=0.95,
                        help="fixed APoZ threshold; manuscript uses 0.95")
    parser.add_argument("--prune-ratio", type=float, default=0.40,
                        help="target fraction of filters pruned per eligible layer (default: 0.40)")
    parser.add_argument("--min-channels", type=int, default=8,
                        help="minimum output channels retained in an eligible Conv2d")
    parser.add_argument("--fpgm-iters", type=int, default=20,
                        help="Weiszfeld iterations for geometric median calculation")
    parser.add_argument("--output", type=str, default="weights/M6_pruned.pt",
                        help="output pruned checkpoint path")
    parser.add_argument("--report", type=str, default="",
                        help="optional CSV report path; default: <output>_pruning_report.csv")
    parser.add_argument("--seed", type=int, default=42,
                        help="numpy / torch seed used for deterministic calibration ordering")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device index or 'cpu'")
    parser.add_argument("--verbose", action="store_true",
                        help="enable verbose logging")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_module_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    if not name:
        return model
    current: nn.Module = model
    for part in name.split("."):
        if part.isdigit():
            current = current[int(part)]  # type: ignore[index]
        else:
            current = getattr(current, part)
    return current


def get_parent_module(model: nn.Module, name: str) -> Tuple[Optional[nn.Module], str]:
    if "." not in name:
        return None, name
    parent_name, attr = name.rsplit(".", 1)
    return get_module_by_name(model, parent_name), attr


def is_detect_conv(model: nn.Module, module: nn.Module) -> bool:
    """Return True if module is one of Detect's final prediction convolutions."""
    try:
        detect = model.model[-1]
        if isinstance(detect, Detect):
            return any(module is m for m in detect.m)
    except (AttributeError, IndexError):
        pass
    return False


def module_path_contains_attention(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("coordatt", "coord_att", "coordattn", "ca."))


def eligible_conv_modules(model: nn.Module) -> List[Tuple[str, nn.Conv2d]]:
    """
    Return Conv2d modules suitable as structural pruning roots.

    Final Detect convolutions are excluded because their output dimension is
    fixed at anchors * (5 + nc). Grouped/depthwise convolutions are excluded
    for conservative structural compatibility. Internal CoordAtt convolutions
    are also excluded; the surrounding feature channels can still be pruned
    through graph dependencies.
    """
    result: List[Tuple[str, nn.Conv2d]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if is_detect_conv(model, module):
            continue
        if module.groups != 1:
            continue
        if module.out_channels < 2:
            continue
        if module_path_contains_attention(name):
            continue
        result.append((name, module))
    return result


def activation_observer_for_conv(model: nn.Module, name: str, conv: nn.Conv2d) -> nn.Module:
    """
    Select the module whose output is used for APoZ counting.

    For the standard YOLOv5 Conv/DWConv wrapper, APoZ should be measured after
    BN/activation rather than on the raw Conv2d output. For a bare Conv2d, the
    Conv2d output itself is used.
    """
    parent, attr = get_parent_module(model, name)
    if parent is not None and getattr(parent, attr, None) is conv:
        if hasattr(parent, "conv") and getattr(parent, "conv", None) is conv and hasattr(parent, "act"):
            return parent
    return conv


class APoZCollector:
    def __init__(self, model: nn.Module, targets: Sequence[Tuple[str, nn.Conv2d]]) -> None:
        self.counts: Dict[str, torch.Tensor] = {}
        self.totals: Dict[str, int] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

        for name, conv in targets:
            observer = activation_observer_for_conv(model, name, conv)
            self.counts[name] = torch.zeros(conv.out_channels, dtype=torch.float64)
            self.totals[name] = 0

            def hook(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output: object,
                     layer_name: str = name) -> None:
                if not isinstance(output, torch.Tensor) or output.ndim != 4:
                    return
                if output.shape[1] != self.counts[layer_name].numel():
                    # A dependency-pruning operation has changed this module before
                    # calibration finishes. Calibration is performed before pruning,
                    # so this should normally never happen.
                    return
                zeros = torch.count_nonzero(output == 0, dim=(0, 2, 3)).detach().cpu().double()
                self.counts[layer_name] += zeros
                self.totals[layer_name] += int(output.shape[0] * output.shape[2] * output.shape[3])

            self.handles.append(observer.register_forward_hook(hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def apoz(self, name: str) -> torch.Tensor:
        total = max(self.totals.get(name, 0), 1)
        return self.counts[name] / float(total)


def letterbox_image(image: np.ndarray, size: int = 640) -> np.ndarray:
    """YOLO-style letterbox to a square image."""
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    new_w = int(round(w * ratio))
    new_h = int(round(h * ratio))
    if (new_w, new_h) != (w, h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw = size - new_w
    dh = size - new_h
    left = int(round(dw / 2 - 0.1))
    right = int(round(dw / 2 + 0.1))
    top = int(round(dh / 2 - 0.1))
    bottom = int(round(dh / 2 + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return image


def image_paths_from_entry(entry: object, root: Path) -> List[Path]:
    """Resolve a YOLO dataset YAML train/val entry to image paths."""
    if isinstance(entry, (list, tuple)):
        paths: List[Path] = []
        for item in entry:
            paths.extend(image_paths_from_entry(item, root))
        return paths

    raw = str(entry)
    path = Path(raw)
    if not path.is_absolute():
        path = (root / path).resolve()

    if path.is_file() and path.suffix.lower() in {".txt", ".list"}:
        paths: List[Path] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if not p.is_absolute():
                p = (path.parent / p).resolve()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(p)
        return paths

    if path.is_dir():
        return sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    # Support glob patterns in the YAML entry.
    return sorted(
        p.resolve() for p in root.glob(raw)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_calibration_paths(data_yaml: str, split: str, max_images: int, seed: int) -> List[Path]:
    yaml_path = Path(data_yaml).resolve()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if split not in data:
        raise KeyError(f"Dataset YAML does not contain '{split}': {data_yaml}")

    paths = image_paths_from_entry(data[split], yaml_path.parent)
    if not paths:
        raise FileNotFoundError(f"No images found for split '{split}' in {data_yaml}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    paths = [paths[int(i)] for i in order]
    if max_images > 0:
        paths = paths[:max_images]
    return paths


def batch_images(paths: Sequence[Path], batch_size: int, size: int, device: torch.device) -> Iterable[torch.Tensor]:
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        tensors: List[np.ndarray] = []
        for path in batch_paths:
            image = cv2.imread(str(path))
            if image is None:
                LOGGER.warning("Unable to read image: %s", path)
                continue
            image = letterbox_image(image, size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose(2, 0, 1)
            tensors.append(image)
        if not tensors:
            continue
        batch = torch.from_numpy(np.stack(tensors)).float() / 255.0
        yield batch.to(device, non_blocking=True)


def geometric_median(points: torch.Tensor, iterations: int = 20, eps: float = 1e-6) -> torch.Tensor:
    """Compute a geometric median with the Weiszfeld algorithm."""
    if points.ndim != 2:
        points = points.reshape(points.shape[0], -1)
    if points.shape[0] == 1:
        return points[0]

    median = points.mean(dim=0)
    for _ in range(iterations):
        distances = torch.linalg.norm(points - median, dim=1).clamp_min(eps)
        weights = 1.0 / distances
        new_median = (points * weights[:, None]).sum(dim=0) / weights.sum()
        if torch.linalg.norm(new_median - median) < eps:
            median = new_median
            break
        median = new_median
    return median


def fpgm_indices(weight: torch.Tensor, candidate_indices: Sequence[int], num_remove: int,
                 iterations: int = 20) -> List[int]:
    """
    Select filters nearest to the geometric median, iteratively.

    The selection is restricted to filters not already selected by APoZ.
    """
    if num_remove <= 0 or len(candidate_indices) <= 1:
        return []

    remaining = list(int(i) for i in candidate_indices)
    selected: List[int] = []
    for _ in range(min(num_remove, len(remaining) - 1)):
        subset = weight[remaining].detach().float().reshape(len(remaining), -1)
        median = geometric_median(subset, iterations=iterations)
        distances = torch.linalg.norm(subset - median, dim=1)
        local_index = int(torch.argmin(distances).item())
        selected.append(remaining.pop(local_index))
    return selected


def select_filters_to_prune(conv: nn.Conv2d, apoz: torch.Tensor, apoz_threshold: float,
                            prune_ratio: float, min_channels: int,
                            geometric_median_iterations: int) -> Tuple[List[int], int, int]:
    """Return final local output-channel indices to prune."""
    out_channels = conv.out_channels
    max_remove = max(0, out_channels - min_channels)
    target_remove = min(int(round(out_channels * prune_ratio)), max_remove)

    apoz_candidates = [i for i, value in enumerate(apoz.tolist()) if value > apoz_threshold]
    apoz_remove = apoz_candidates[:]

    # APoZ pruning has priority. If all high-APoZ filters would exceed the safe
    # channel floor, retain the lowest-APoZ subset needed to keep min_channels.
    if len(apoz_remove) > max_remove:
        apoz_remove = sorted(apoz_remove, key=lambda i: float(apoz[i]), reverse=True)[:max_remove]
        LOGGER.warning(
            "Layer has %d filters above APoZ %.3f, but only %d can be pruned while "
            "retaining min_channels=%d; APoZ pruning was capped for safety.",
            len(apoz_candidates), apoz_threshold, max_remove, min_channels,
        )

    remaining = [i for i in range(out_channels) if i not in set(apoz_remove)]
    extra = max(0, target_remove - len(apoz_remove))
    extra = min(extra, max(0, len(remaining) - min_channels))

    fpgm_remove = fpgm_indices(
        conv.weight.data,
        remaining,
        extra,
        iterations=geometric_median_iterations,
    )
    final = sorted(set(apoz_remove + fpgm_remove))
    return final, len(apoz_remove), len(fpgm_remove)


def build_pruning_group(model: nn.Module, conv: nn.Conv2d, indices: Sequence[int], example_inputs: torch.Tensor):
    dg = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
    group = dg.get_pruning_group(conv, tp.prune_conv_out_channels, idxs=list(indices))
    return dg, group


def model_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(model: nn.Module, source_weights: str, output: str, metadata: Dict[str, object]) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the complete model object so that attempt_load() can reconstruct the
    # structurally-pruned architecture without consulting the original YAML.
    model_to_save = deepcopy(model).float().cpu()
    checkpoint = {
        "model": model_to_save,
        "ema": None,
        "optimizer": None,
        "updates": None,
        "training_results": None,
        "best_fitness": None,
        "epoch": -1,
        "source_weights": str(source_weights),
        "m6_pruning": metadata,
    }
    torch.save(checkpoint, output_path)


def write_report(path: str, records: Sequence[Dict[str, object]]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    fields = list(records[0].keys())
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    set_seed(args.seed)

    if not (0.0 <= args.apoz_threshold <= 1.0):
        raise ValueError("--apoz-threshold must be between 0 and 1")
    if not (0.0 < args.prune_ratio < 1.0):
        raise ValueError("--prune-ratio must be between 0 and 1")
    if args.img_size <= 0 or args.batch_size <= 0:
        raise ValueError("--img-size and --batch-size must be positive")

    device = torch.device(
        f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    )
    LOGGER.info("Device: %s", device)
    LOGGER.info("Loading checkpoint: %s", args.weights)
    model = attempt_load(args.weights, map_location=device)
    if isinstance(model, (list, tuple)):
        raise TypeError("M6 expects one PyTorch model checkpoint, not an ensemble.")
    model = model.to(device).float().eval()

    try:
        model.nc = int(getattr(model, "nc", 1))
    except Exception:
        pass

    before_params = model_parameter_count(model)
    candidates = eligible_conv_modules(model)
    LOGGER.info("Eligible Conv2d pruning roots: %d", len(candidates))
    LOGGER.info("APoZ threshold: %.3f", args.apoz_threshold)
    LOGGER.info("Target pruning ratio per layer: %.3f", args.prune_ratio)

    calibration_paths = load_calibration_paths(args.data, args.split, args.apoz_images, args.seed)
    LOGGER.info("APoZ calibration images: %d", len(calibration_paths))

    # Build the first dependency-safe example before hooks/pruning.
    example_inputs = next(batch_images(calibration_paths[:1], 1, args.img_size, device))

    # ------------------------------------------------------------------
    # Step 1: APoZ calibration on the unpruned model.
    # ------------------------------------------------------------------
    collector = APoZCollector(model, candidates)
    try:
        with torch.no_grad():
            for batch in batch_images(calibration_paths, args.batch_size, args.img_size, device):
                _ = model(batch)
    finally:
        collector.close()

    apoz_stats: Dict[str, torch.Tensor] = {
        name: collector.apoz(name).float() for name, _ in candidates
    }

    # ------------------------------------------------------------------
    # Step 2: Graph-aware structural pruning.
    # ------------------------------------------------------------------
    records: List[Dict[str, object]] = []
    pruned_layers = 0
    skipped_layers = 0

    for name, original_conv in candidates:
        conv = get_module_by_name(model, name)
        if not isinstance(conv, nn.Conv2d):
            skipped_layers += 1
            continue

        # A previous graph-aware pruning operation may have changed this layer's
        # output dimensionality as a dependency. In that case, its original APoZ
        # vector no longer indexes the current channels, so do not apply stale
        # indices a second time.
        apoz = apoz_stats[name]
        if conv.out_channels != apoz.numel():
            LOGGER.info(
                "Skip %s: output channels changed by an earlier dependency pruning "
                "operation (%d -> %d).",
                name, int(apoz.numel()), conv.out_channels,
            )
            skipped_layers += 1
            continue

        remove_indices, apoz_count, fpgm_count = select_filters_to_prune(
            conv,
            apoz,
            args.apoz_threshold,
            args.prune_ratio,
            args.min_channels,
            args.fpgm_iters,
        )

        if not remove_indices:
            records.append({
                "layer": name,
                "original_out_channels": int(conv.out_channels),
                "apoz_over_threshold": int(sum(float(x) > args.apoz_threshold for x in apoz)),
                "apoz_pruned": 0,
                "fpgm_pruned": 0,
                "total_pruned": 0,
                "remaining_out_channels": int(conv.out_channels),
                "apoz_threshold": args.apoz_threshold,
                "target_prune_ratio": args.prune_ratio,
                "status": "no_filters_selected",
            })
            continue

        # The graph is rebuilt after each structural pruning step because channel
        # dimensions in dependent modules have changed.
        example_inputs = next(batch_images(calibration_paths[:1], 1, args.img_size, device))
        dg = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
        group = dg.get_pruning_group(
            conv,
            tp.prune_conv_out_channels,
            idxs=remove_indices,
        )

        if not dg.check_pruning_group(group):
            LOGGER.warning("Skip %s: torch-pruning rejected the pruning group.", name)
            records.append({
                "layer": name,
                "original_out_channels": int(conv.out_channels),
                "apoz_over_threshold": int(sum(float(x) > args.apoz_threshold for x in apoz)),
                "apoz_pruned": apoz_count,
                "fpgm_pruned": fpgm_count,
                "total_pruned": len(remove_indices),
                "remaining_out_channels": int(conv.out_channels),
                "apoz_threshold": args.apoz_threshold,
                "target_prune_ratio": args.prune_ratio,
                "status": "dependency_group_rejected",
            })
            skipped_layers += 1
            continue

        original_out = conv.out_channels
        group.prune()
        new_conv = get_module_by_name(model, name)
        remaining_out = int(new_conv.out_channels) if isinstance(new_conv, nn.Conv2d) else -1
        pruned_layers += 1

        records.append({
            "layer": name,
            "original_out_channels": int(original_out),
            "apoz_over_threshold": int(sum(float(x) > args.apoz_threshold for x in apoz)),
            "apoz_pruned": apoz_count,
            "fpgm_pruned": fpgm_count,
            "total_pruned": len(remove_indices),
            "remaining_out_channels": remaining_out,
            "apoz_threshold": args.apoz_threshold,
            "target_prune_ratio": args.prune_ratio,
            "status": "pruned",
        })
        LOGGER.info(
            "Pruned %-35s: %d -> %d | APoZ=%d, FPGM=%d",
            name, original_out, remaining_out, apoz_count, fpgm_count,
        )

    # ------------------------------------------------------------------
    # Step 3: Validate forward pass after graph-aware pruning.
    # ------------------------------------------------------------------
    model = model.to(device).float().eval()
    validation_batch = next(batch_images(calibration_paths[:1], 1, args.img_size, device))
    with torch.no_grad():
        output = model(validation_batch)

    if isinstance(output, tuple):
        raw = output[1]
    else:
        raw = output
    if not isinstance(raw, (list, tuple)):
        raise RuntimeError("Unexpected model output after M6 pruning; expected YOLO training outputs.")

    detect = model.model[-1]
    LOGGER.info("Post-pruning detection layers: %d", len(raw))
    LOGGER.info("Post-pruning parameter count: %d", model_parameter_count(model))

    # Check that the detection head still has the expected fixed output dimension.
    if isinstance(detect, Detect):
        expected_no = int(detect.nc + 5)
        for head in detect.m:
            expected_channels = int(detect.na * expected_no)
            if head.out_channels != expected_channels:
                raise RuntimeError(
                    "The M6 pruning operation changed the final Detect output channel count, "
                    "which must remain anchors * (5 + nc)."
                )

    after_params = model_parameter_count(model)
    reduction = 1.0 - after_params / max(before_params, 1)

    metadata = {
        "method": "Activation-based FPGM pruning with APoZ",
        "apoz_threshold": args.apoz_threshold,
        "prune_ratio": args.prune_ratio,
        "min_channels": args.min_channels,
        "geometric_median_iterations": args.fpgm_iters,
        "calibration_split": args.split,
        "calibration_images": len(calibration_paths),
        "input_size": args.img_size,
        "seed": args.seed,
        "device": str(device),
        "parameters_before": before_params,
        "parameters_after": after_params,
        "parameter_reduction_fraction": reduction,
        "pruned_layers": pruned_layers,
        "skipped_layers": skipped_layers,
    }

    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else output_path.with_name(output_path.stem + "_pruning_report.csv")
    save_checkpoint(model, args.weights, str(output_path), metadata)
    write_report(str(report_path), records)

    json_path = output_path.with_name(output_path.stem + "_metadata.json")
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    LOGGER.info("M6 pruning finished.")
    LOGGER.info("Parameters: %d -> %d (%.2f%% reduction)", before_params, after_params, 100.0 * reduction)
    LOGGER.info("M6 checkpoint: %s", output_path)
    LOGGER.info("Layer report: %s", report_path)
    LOGGER.info("Metadata: %s", json_path)


if __name__ == "__main__":
    main()
