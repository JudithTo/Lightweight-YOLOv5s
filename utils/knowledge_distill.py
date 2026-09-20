"""M7 knowledge distillation for the single-class nectar-flower detector.

This module implements Supplementary Information Section S3:

    L_reg = L_sL1(R_s, y_reg) + v L_b(R_s, R_t)       (Eq. S1)
    L_b   = ||R_t - R_s||^2                            (Eq. S2)

where v = 0.5.

The current task has nc=1, so the classification branch of the YOLOv5 loss is
inactive. M7 therefore uses the regression objective in S1 together with the
original YOLOv5 objectness loss; classification soft-target KD, temperature,
and alpha_cls are not used.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.loss import FocalLoss, bbox_iou_xywh, smooth_BCE
from utils.torch_utils import is_parallel


M7_V_REG = 0.5


def get_raw_predictions(model_output):
    """Extract YOLOv5 raw multi-scale predictions.

    In this repository's models/yolo.py:
      * train mode returns the raw prediction list;
      * eval mode returns (decoded_prediction, raw_prediction_list).
    """
    if isinstance(model_output, tuple):
        if len(model_output) != 2:
            raise ValueError(
                f'Unexpected model output tuple length: {len(model_output)}'
            )
        raw = model_output[1]
    else:
        raw = model_output

    if not isinstance(raw, (list, tuple)):
        raise TypeError(
            'Expected raw YOLOv5 predictions as a list/tuple, '
            f'got {type(raw).__name__}'
        )
    return list(raw)


def check_teacher_student_compatibility(student_model, teacher_model, atol=1e-6):
    """Validate that teacher and student have compatible detector outputs."""
    student_det = student_model.module.model[-1] if is_parallel(student_model) else student_model.model[-1]
    teacher_det = teacher_model.module.model[-1] if is_parallel(teacher_model) else teacher_model.model[-1]

    if student_det.nc != teacher_det.nc:
        raise ValueError(f'nc mismatch: student={student_det.nc}, teacher={teacher_det.nc}')
    if student_det.nl != teacher_det.nl:
        raise ValueError(f'detection-layer mismatch: student={student_det.nl}, teacher={teacher_det.nl}')
    if student_det.na != teacher_det.na:
        raise ValueError(f'anchor-count mismatch: student={student_det.na}, teacher={teacher_det.na}')
    if not torch.allclose(student_det.stride.float(), teacher_det.stride.float(), atol=atol):
        raise ValueError(
            f'stride mismatch: student={student_det.stride.tolist()}, '
            f'teacher={teacher_det.stride.tolist()}'
        )
    if not torch.allclose(student_det.anchors.float(), teacher_det.anchors.float(), atol=atol):
        raise ValueError('Teacher and student anchors differ. M7 requires the student to be pruned from the same trained teacher model.')


def decode_yolo_box(raw_box, anchors):
    """Decode YOLOv5 raw box values to [cx, cy, w, h] grid-space boxes."""
    pxy = raw_box[:, :2].sigmoid() * 2.0 - 0.5
    pwh = (raw_box[:, 2:4].sigmoid() * 2.0).pow(2) * anchors
    return torch.cat((pxy, pwh), dim=1)


class RegressionDistillationLoss(nn.Module):
    """Implementation of Supplementary Equations S1-S2."""

    def __init__(self, v=M7_V_REG):
        super().__init__()
        if v < 0:
            raise ValueError(f'v must be non-negative, got {v}')
        self.v = float(v)
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean')

    def forward(self, student_reg, target_reg, teacher_reg):
        """Return L_reg, L_sL1 and L_b for matched positive boxes."""
        if student_reg.numel() == 0:
            zero = student_reg.sum() * 0.0
            return zero, zero, zero

        if (student_reg.shape != target_reg.shape or
                student_reg.shape != teacher_reg.shape or
                student_reg.ndim != 2 or student_reg.shape[1] != 4):
            raise ValueError(
                'All regression tensors must have shape [N, 4]; '
                f'got student={tuple(student_reg.shape)}, '
                f'target={tuple(target_reg.shape)}, '
                f'teacher={tuple(teacher_reg.shape)}'
            )

        # Equation S1: L_sL1(R_s, y_reg).
        loss_s_l1 = self.smooth_l1(student_reg, target_reg)

        # Equation S2: L_b(R_s, R_t) = ||R_t - R_s||^2.
        # Teacher is detached so gradients are propagated only through the student.
        diff = teacher_reg.detach() - student_reg
        loss_b = torch.sum(diff.pow(2), dim=1).mean()

        # Equation S1.
        loss_reg = loss_s_l1 + self.v * loss_b
        return loss_reg, loss_s_l1, loss_b


class ComputeLossKD:
    """M7 YOLOv5 training loss.

    Training:
        total = objectness_loss + box_gain * (L_sL1 + v * L_b)

    Validation:
        When no teacher prediction is supplied, the two-argument call remains
        compatible with the existing YOLOv5 validation interface.
    """

    def __init__(self, model, v=M7_V_REG, autobalance=False):
        device = next(model.parameters()).device
        h = model.hyp

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]
        if det.nc != 1:
            raise ValueError(
                f'ComputeLossKD is configured for the single-class task (nc=1), got nc={det.nc}'
            )

        BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['obj_pw']], device=device)
        )
        BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['cls_pw']], device=device)
        )

        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))
        g = h['fl_gamma']
        if g > 0:
            BCEobj = FocalLoss(BCEobj, g)
            BCEcls = FocalLoss(BCEcls, g)

        self.balance = {3: [4.0, 1.0, 0.4]}.get(
            det.nl, [4.0, 1.0, 0.25, 0.06, 0.02]
        )
        self.ssi = list(det.stride).index(16) if autobalance else 0
        self.BCEobj = BCEobj
        self.BCEcls = BCEcls
        self.gr = model.gr
        self.hyp = h
        self.autobalance = autobalance

        for k in ('na', 'nc', 'nl', 'anchors'):
            setattr(self, k, getattr(det, k))

        self.regression_distill = RegressionDistillationLoss(v=v).to(device)
        self.last_kd_items = torch.zeros(2, device=device)

    def __call__(self, p, teacher_p_or_targets, targets=None):
        """Compute M7 loss.

        Training call:
            compute_loss(student_p, teacher_p, targets)

        Validation-compatible call used by the existing YOLOv5 test.py:
            compute_loss(student_p, targets)
        In validation mode, no teacher term is supplied.
        """
        if targets is None:
            teacher_p = None
            targets = teacher_p_or_targets
        else:
            teacher_p = teacher_p_or_targets

        device = targets.device
        if teacher_p is not None:
            teacher_p = list(teacher_p)
            if len(p) != len(teacher_p):
                raise ValueError(
                    f'Teacher/student detection layers differ: {len(teacher_p)} vs {len(p)}'
                )
            for i, (ps, pt) in enumerate(zip(p, teacher_p)):
                if ps.shape != pt.shape:
                    raise ValueError(
                        f'Teacher/student prediction shape mismatch at layer {i}: '
                        f'student={tuple(ps.shape)}, teacher={tuple(pt.shape)}'
                    )

        lreg = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        l_s_l1_total = torch.zeros(1, device=device)
        l_b_total = torch.zeros(1, device=device)

        # nc=1, so classification is inactive by construction.
        lcls = torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)

        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)
            n = b.shape[0]

            if n:
                ps = pi[b, a, gj, gi]
                student_box = decode_yolo_box(ps, anchors[i])

                if teacher_p is not None:
                    teacher_ps = teacher_p[i][b, a, gj, gi]
                    teacher_box = decode_yolo_box(teacher_ps, anchors[i])
                    lreg_i, l_s_l1_i, l_b_i = self.regression_distill(
                        student_box,
                        tbox[i],
                        teacher_box,
                    )
                    l_b_total += l_b_i
                else:
                    # Used only by test.py during validation.
                    l_s_l1_i = self.regression_distill.smooth_l1(student_box, tbox[i])
                    lreg_i = l_s_l1_i
                    l_b_i = student_box.sum() * 0.0

                lreg += lreg_i
                l_s_l1_total += l_s_l1_i

                # Preserve the original YOLOv5 objectness target construction.
                iou = bbox_iou_xywh(student_box, tbox[i])
                tobj[b, a, gj, gi] = (
                    (1.0 - self.gr)
                    + self.gr * iou.detach().clamp(0).type(tobj.dtype)
                )

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]
            if self.autobalance:
                self.balance[i] = (
                    self.balance[i] * 0.9999
                    + 0.0001 / obji.detach().item()
                )

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        lreg *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = p[0].shape[0]

        loss = lreg + lobj + lcls
        self.last_kd_items = torch.cat((l_s_l1_total, l_b_total)).detach()
        loss_items = torch.cat((lreg, lobj, lcls, loss)).detach()
        return loss * bs, loss_items

    def build_targets(self, p, targets):
        """Build YOLOv5 anchor/grid targets (same matching scheme as ComputeLoss)."""
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

        g = 0.5
        off = torch.tensor(
            [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
            device=targets.device,
            dtype=torch.float32,
        ) * g

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(
                p[i].shape,
                device=targets.device,
                dtype=torch.float32,
            )[[3, 2, 3, 2]]

            t = targets * gain
            if nt:
                r = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1.0 / r).max(2)[0] < self.hyp['anchor_t']
                t = t[j]

                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1.0 < g) & (gxy > 1.0)).T
                l, m = ((gxi % 1.0 < g) & (gxi > 1.0)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = (gxy - offsets).long()
            gi, gj = gij.T

            a = t[:, 6].long()
            indices.append((
                b,
                a,
                gj.clamp_(0, gain[3] - 1),
                gi.clamp_(0, gain[2] - 1),
            ))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)

        return tcls, tbox, indices, anch
