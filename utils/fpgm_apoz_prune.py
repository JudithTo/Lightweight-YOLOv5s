"""
M6 Activation‑based FPGM filter pruning with APoZ
Experimental prototype script, not integrated into train.py.
Workflow: 1.Load trained pt model; 2.Compute APoZ for conv layers; 3.Filter high‑zero‑activation filters; 4.FPGM geometric‑median pruning.
Note: Requires sample validation images to run forward pass for APoZ statistics.
"""
import torch
import numpy as np
from models.experimental import attempt_load


def compute_apoz(feature_map: torch.Tensor):
    """
    Compute APoZ(Average Percentage of Zeros) per filter: [B,C,H,W] feature map
    return shape: (C,) APoZ value for each convolution output channel
    """
    b, c, h, w = feature_map.shape
    zero_cnt = torch.count_nonzero((feature_map == 0), dim=[0, 2, 3])
    total_elem = b * h * w
    apoz_ratio = zero_cnt.float() / float(total_elem)
    return apoz_ratio


def fpgm_calc_redundant_kernel(kernel_weight: torch.Tensor):
    """
    Input kernel_weight: [out_channels, in_channels, k, k]
    FPGM: compute pairwise distance, return index of most replaceable filter
    """
    out_c = kernel_weight.shape[0]
    dist_list = []
    for i in range(out_c):
        other_kernels = torch.cat([kernel_weight[j:j+1] for j in range(out_c) if j != i], dim=0)
        dist = torch.sum(torch.sqrt(torch.sum(torch.pow(kernel_weight[i] - other_kernels, 2), dim=[1,2,3])))
        dist_list.append(dist.item())
    return int(np.argmin(np.array(dist_list)))


def prune_conv_layer(conv_module: torch.nn.Conv2d, keep_mask: torch.Tensor):
    """
    Prune conv layer according to boolean keep_mask
    :param conv_module: original Conv2d
    :param keep_mask: boolean tensor shape [out_channels] True=keep
    :return pruned conv module, keep index list
    """
    keep_idx = torch.where(keep_mask)[0]
    new_conv = torch.nn.Conv2d(
        in_channels=conv_module.in_channels,
        out_channels=int(len(keep_idx)),
        kernel_size=conv_module.kernel_size,
        stride=conv_module.stride,
        padding=conv_module.padding,
        dilation=conv_module.dilation,
        groups=conv_module.groups,
        bias=conv_module.bias is not None
    )
    new_conv.weight.data.copy_(conv_module.weight.data[keep_idx].clone())
    if conv_module.bias is not None:
        new_conv.bias.data.copy_(conv_module.bias.data[keep_idx].clone())
    return new_conv, keep_idx


def get_conv_modules(model):
    conv_list = []
    name_list = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            conv_list.append(m)
            name_list.append(name)
    return name_list, conv_list


def prune_pipeline(model, sample_batch_img, apoz_threshold=0.95, prune_ratio=0.3, device="cuda:0"):
    """
    Full M6 pruning pipeline
    Args:
        model: loaded trained yolov5 model
        sample_batch_img: sample image tensor [B,3,H,W] for APoZ statistics
        apoz_threshold: filter with APoZ > this value will be directly pruned
        prune_ratio: FPGM target pruning ratio
        device: device
    """
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        _ = model(sample_batch_img.to(device))  # trigger forward to collect feature maps

    print("[INFO] M6 APoZ + FPGM pruning pipeline start.")
    name_list, conv_list = get_conv_modules(model)
    for layer_name, conv in zip(name_list, conv_list):
        print(f"Process conv layer: {layer_name}, out_channels={conv.out_channels}")
        # step1 APoZ statistics (need intermediate hook implementation, prototype placeholder)
        # Note: This prototype only shows logic; to get real APoZ you need register forward hook to capture output feature maps.
        # step2 FPGM filter select
        remove_idx = fpgm_calc_redundant_kernel(conv.weight.data)
        keep_mask = torch.ones(conv.out_channels, dtype=torch.bool)
        keep_mask[remove_idx] = False
        new_conv, _ = prune_conv_layer(conv, keep_mask)
        conv = new_conv
    print("[INFO] Pruning pipeline finished. Save pruned model manually.")
    return model


if __name__ == "__main__":
    """Example usage"""
    weights_path = "best.pt"
    device = "cuda:0"
    model = attempt_load(weights_path, map_location=device)
    dummy_img = torch.randn((2, 3, 640, 640), device=device)
    pruned_model = prune_pipeline(model, dummy_img, apoz_threshold=0.95, prune_ratio=0.3, device=device)
    torch.save({"model": pruned_model.half()}, "pruned_model.pt")
