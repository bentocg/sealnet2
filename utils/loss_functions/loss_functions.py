__all__ = ["DiceLoss", "SoftDiceLoss", "FocalLoss", "MixedLoss"]

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha=0.25,
        gamma=2,
        reduction="mean",
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, label):
        """
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLoss()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        """
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(
            logits >= 0, F.softplus(logits, -1, 50), logits - F.softplus(logits, 1, 50)
        )
        log_1_probs = torch.where(
            logits >= 0,
            -logits + F.softplus(logits, -1, 50),
            -F.softplus(logits, 1, 50),
        )
        loss = (
            label * self.alpha * log_probs
            + (1.0 - label) * (1.0 - self.alpha) * log_1_probs
        )
        loss = loss * coeff

        if self.reduction == "mean":
            loss = loss.mean()
        if self.reduction == "sum":
            loss = loss.sum()
        return loss


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


class MixedLoss(nn.Module):
    """
    Mix between FocalLoss and DiceLoss
    """
    def __init__(self, alpha=10.0):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        loss = self.alpha * self.focal(pred, target) + self.dice(pred, target)
        return loss


class SoftDiceLoss(nn.Module):
    """
    soft-dice loss, useful in binary segmentation
    """

    def __init__(self, p=1, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        """
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        """
        probs = torch.sigmoid(logits)
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        loss = 1.0 - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss

