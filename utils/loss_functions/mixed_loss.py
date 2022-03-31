__all__ = ["DiceLoss", "FocalLoss", "MixedLoss", "LogCoshLoss", "DicePerimeterLoss"]

import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    def __init__(self, merge="mean"):
        super().__init__()
        self.smooth = 1
        self.p = 1
        self.merge = merge

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(pred.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = 2 * torch.sum(torch.mul(pred, target), dim=1) + self.smooth
        den = torch.sum(pred.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den
        if self.merge == "sum":
            return loss.sum()
        elif self.merge == "mean":
            return loss.mean()
        else:
            raise Exception("Merging mode not implemented")


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, merge="mean"):
        super().__init__()
        self.gamma = gamma
        self.merge = merge

    def forward(self, pred, target):
        if len(target.shape) == 1:
            pred = pred.view(-1)
        if not (target.size() == pred.size()):
            raise ValueError(
                "Target size ({}) must be the same as input size ({})".format(
                    target.size(), pred.size()
                )
            )
        max_val = (-pred).clamp(min=0)
        loss = (
            pred
            - pred * target
            + max_val
            + ((-max_val).exp() + (-pred - max_val).exp()).log()
        )
        invprobs = F.logsigmoid(-pred * (target * 2.0 - 1.0))

        loss = (invprobs * self.gamma).exp() * loss
        if self.merge == "sum":
            return loss.sum()
        elif self.merge == "mean":
            return loss.mean()
        else:
            raise Exception("Merging mode not implemented")


class MixedLoss(nn.Module):
    def __init__(self, alpha=10.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        loss = self.alpha * self.focal(pred, target) + self.dice(pred, target)
        return loss


class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        loss = torch.log(torch.cosh(self.dice(pred, target)))
        return loss


class DicePerimeterLoss(nn.Module):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.dice = DiceLoss()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def _contour(self, mask):
        min_pool = -F.max_pool2d(-mask, (3, 3), 1, 1)
        max_pool = F.max_pool2d(mask, (3, 3), 1, 1)
        contour = F.relu(min_pool - max_pool)
        return contour

    def forward(self, pred, target):
        cont_pred = self._contour(pred)
        cont_target = self._contour(target)
        perim_loss = self.mse(cont_pred, cont_target)
        dice_loss = self.dice(pred, target)
        return (1 - self.alpha) * dice_loss + self.alpha * perim_loss
