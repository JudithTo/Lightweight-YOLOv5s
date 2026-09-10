"""
M7 Knowledge distillation prototype
Experimental prototype.
Teacher: full‑size improved YOLOv5s
Student: pruned lightweight model from M6(ApoZ‑FPGM pruning)
Not integrated into train.py main loop.
"""
import torch
import torch.nn as nn


class DistillLoss(nn.Module):
    def __init__(self, temperature: float = 2.0, alpha_cls: float = 0.5):
        """
        :param temperature: distillation temperature T
        :param alpha_cls: weight between soft‑target distillation loss and hard‑label loss
        """
        super().__init__()
        self.T = temperature
        self.alpha_cls = alpha_cls
        self.ce_hard = nn.CrossEntropyLoss()
        self.mse_reg = nn.MSELoss()

    def forward(self, student_raw_pred, teacher_raw_pred, hard_label_target):
        """
        Args:
            student_raw_pred: student model raw output logits
            teacher_raw_pred: teacher model raw output logits
            hard_label_target: ground‑truth hard label
        Returns: total distillation loss
        """
        # classification distillation loss
        soft_teacher = torch.softmax(teacher_raw_pred / self.T, dim=-1)
        soft_student = torch.log_softmax(student_raw_pred / self.T, dim=-1)
        loss_soft_cls = -torch.mean(torch.sum(soft_teacher * soft_student, dim=-1))

        loss_hard_cls = self.ce_hard(student_raw_pred, hard_label_target)
        cls_total_loss = self.alpha_cls * loss_soft_cls + (1.0 - self.alpha_cls) * loss_hard_cls

        # box regression distillation loss
        loss_reg_box = self.mse_reg(student_raw_pred[..., :4], teacher_raw_pred[..., :4])

        total_loss = cls_total_loss + loss_reg_box
        return total_loss


def distill_one_epoch(teacher_model, student_model, dataloader, optimizer, distill_criterion, device):
    """
    One epoch distill training loop prototype
    :param teacher_model: frozen teacher model (eval mode)
    :param student_model: student model to update
    :param dataloader: training dataloader
    :param optimizer: optimizer for student
    :param distill_criterion: DistillLoss instance
    :param device: device
    """
    teacher_model.eval()
    student_model.train()
    for imgs, targets, *_ in dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            teacher_out = teacher_model(imgs)
        student_out = student_model(imgs)
        loss = distill_criterion(student_out, teacher_out, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return


if __name__ == "__main__":
    """Simple usage demo"""
    pass
