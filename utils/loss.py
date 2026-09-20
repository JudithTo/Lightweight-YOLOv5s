# Loss functions
import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    """Return positive and negative targets for label smoothing."""
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss with reduced missing-label effects."""

    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)
        dx = pred - true
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    """Wrap focal loss around an existing BCE loss."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class QFocalLoss(nn.Module):
    """Quality focal loss wrapper retained for compatibility with YOLOv5."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


# =============================================================================
# M4: Focal-alphaEIoU loss
# =============================================================================
# The implementation below follows the equations supplied for the manuscript:
#
# Eq. (1):
# L_EIOU = 1 - IOU + rho^2(b,b_gt)/c^2
#                  + rho^2(w,w_gt)/c_w^2
#                  + rho^2(h,h_gt)/c_h^2
#
# Eq. (2):
# L_Focal-EIOU = IOU^gamma * L_EIOU
#
# Eq. (3):
# L_Focal-alphaEIOU = IOU^gamma * (
#       1 - IOU
#       + rho^(2 alpha)(b,b_gt)/c^2
#       + rho^(2 beta)(w,w_gt)/c_w^2
#       + rho^(2 beta)(h,h_gt)/c_h^2 )
#
# The manuscript specifies alpha=3 and beta=2. Gamma is kept configurable
# because its numerical value is not stated in the supplied manuscript text.
# Set focal_gamma to the value used for the reported experiment.
# =============================================================================


def bbox_iou_xywh(pred_box: torch.Tensor,
                  target_box: torch.Tensor,
                  eps: float = 1e-7) -> torch.Tensor:
    """IoU for aligned boxes represented as [cx, cy, w, h]."""
    if (pred_box.shape != target_box.shape or pred_box.ndim != 2
            or pred_box.shape[1] != 4):
        raise ValueError(
            'pred_box and target_box must both have shape [N, 4]; '
            f'got {tuple(pred_box.shape)} and {tuple(target_box.shape)}'
        )

    px, py, pw, ph = pred_box.unbind(dim=1)
    tx, ty, tw, th = target_box.unbind(dim=1)

    p_x1 = px - pw / 2.0
    p_y1 = py - ph / 2.0
    p_x2 = px + pw / 2.0
    p_y2 = py + ph / 2.0

    t_x1 = tx - tw / 2.0
    t_y1 = ty - th / 2.0
    t_x2 = tx + tw / 2.0
    t_y2 = ty + th / 2.0

    inter_w = (torch.min(p_x2, t_x2) - torch.max(p_x1, t_x1)).clamp(min=0.0)
    inter_h = (torch.min(p_y2, t_y2) - torch.max(p_y1, t_y1)).clamp(min=0.0)
    inter = inter_w * inter_h

    p_area = pw.clamp(min=0.0) * ph.clamp(min=0.0)
    t_area = tw.clamp(min=0.0) * th.clamp(min=0.0)
    union = p_area + t_area - inter

    return (inter / (union + eps)).clamp(min=0.0, max=1.0)


class FocalAlphaEIOULoss(nn.Module):
    """Focal-alphaEIoU loss corresponding to manuscript Equation (3)."""

    def __init__(self, alpha=3.0, beta=2.0, gamma=2.0, eps=1e-7):
        super(FocalAlphaEIOULoss, self).__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.eps = float(eps)

    def forward(self, pred_boxes, target_boxes, return_iou=False):
        if pred_boxes.numel() == 0:
            zero = pred_boxes.sum() * 0.0
            if return_iou:
                return zero, pred_boxes.new_zeros((0,))
            return zero

        if (pred_boxes.shape != target_boxes.shape or pred_boxes.ndim != 2
                or pred_boxes.shape[1] != 4):
            raise ValueError(
                'pred_boxes and target_boxes must have the same shape [N, 4]; '
                f'got {tuple(pred_boxes.shape)} and {tuple(target_boxes.shape)}'
            )

        px, py, pw, ph = pred_boxes.unbind(dim=1)
        tx, ty, tw, th = target_boxes.unbind(dim=1)

        # IoU term used in Eqs. (1)-(3).
        iou = bbox_iou_xywh(pred_boxes, target_boxes, eps=self.eps)

        # rho^2(b, b_gt): squared Euclidean distance between centers.
        rho2_center = (px - tx).pow(2) + (py - ty).pow(2)

        # c_w and c_h: width and height of the smallest enclosing box.
        p_x1 = px - pw / 2.0
        p_y1 = py - ph / 2.0
        p_x2 = px + pw / 2.0
        p_y2 = py + ph / 2.0

        t_x1 = tx - tw / 2.0
        t_y1 = ty - th / 2.0
        t_x2 = tx + tw / 2.0
        t_y2 = ty + th / 2.0

        cw = (torch.max(p_x2, t_x2) - torch.min(p_x1, t_x1)).clamp(min=self.eps)
        ch = (torch.max(p_y2, t_y2) - torch.min(p_y1, t_y1)).clamp(min=self.eps)
        c2 = cw.pow(2) + ch.pow(2) + self.eps

        # Equation (3): rho^(2 alpha) = (rho^2)^alpha.
        center_term = rho2_center.pow(self.alpha) / c2

        # Equation (3): |w-w_gt|^(2 beta) and |h-h_gt|^(2 beta).
        width_term = torch.abs(pw - tw).pow(2.0 * self.beta) / (cw.pow(2) + self.eps)
        height_term = torch.abs(ph - th).pow(2.0 * self.beta) / (ch.pow(2) + self.eps)

        l_eiou = 1.0 - iou + center_term + width_term + height_term
        loss = iou.pow(self.gamma) * l_eiou
        loss = loss.mean()

        if return_iou:
            return loss, iou
        return loss


class ComputeLoss:
    """YOLOv5 detection loss with M4 Focal-alphaEIoU box regression."""

    def __init__(self, model, autobalance=False, focal_alpha=3.0,
                 focal_beta=2.0, focal_gamma=2.0):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device
        h = model.hyp

        BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['cls_pw']], device=device)
        )
        BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['obj_pw']], device=device)
        )

        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))

        g = h['fl_gamma']
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]
        self.balance = {3: [4.0, 1.0, 0.4]}.get(
            det.nl, [4.0, 1.0, 0.25, 0.06, 0.02]
        )
        self.ssi = list(det.stride).index(16) if autobalance else 0
        self.BCEcls = BCEcls
        self.BCEobj = BCEobj
        self.gr = model.gr
        self.hyp = h
        self.autobalance = autobalance

        for k in ('na', 'nc', 'nl', 'anchors'):
            setattr(self, k, getattr(det, k))

        self.focal_alpha_eiou = FocalAlphaEIOULoss(
            alpha=focal_alpha,
            beta=focal_beta,
            gamma=focal_gamma,
        ).to(device)

    def __call__(self, p, targets):
        device = targets.device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)

        tcls, tbox, indices, anchors = self.build_targets(p, targets)

        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)
            n = b.shape[0]

            if n:
                ps = pi[b, a, gj, gi]

                # Decode YOLOv5 regression outputs to [cx, cy, w, h]
                # in grid units, matching the target representation.
                pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2.0).pow(2) * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)

                # M4: replace CIoU with manuscript Eq. (3).
                lbox_i, iou = self.focal_alpha_eiou(
                    pbox, tbox[i], return_iou=True
                )
                lbox += lbox_i

                # Keep original YOLOv5 objectness-target construction.
                tobj[b, a, gj, gi] = (
                    (1.0 - self.gr)
                    + self.gr * iou.detach().clamp(0).type(tobj.dtype)
                )

                # Kept for multi-class compatibility. For the current nc=1 task,
                # this branch is inactive.
                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]
            if self.autobalance:
                self.balance[i] = (
                    self.balance[i] * 0.9999
                    + 0.0001 / obji.detach().item()
                )

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = p[0].shape[0]

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        """Build YOLOv5 anchor/grid targets."""
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
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)

        return tcls, tbox, indices, anch
