__all__ = ["predict", "metric", "Meter", "epoch_log", "seed_all"]

import numpy as np
import torch
import os
import random
import pandas as pd


def predict(X, threshold):
    X_p = np.copy(X)
    preds = (X_p > threshold).astype("uint8")
    return preds


def compute_ious(pred, label, classes, only_present=True):
    """computes iou for one ground truth mask and predicted mask"""
    if np.max(label) == 255:
        label = np.divide(label, 255, casting="unsafe")

    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = (pred_c * label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(outputs, labels, classes=None):
    """computes mean iou for a batch of ground truth masks and predicted masks"""
    ious = []
    preds = np.copy(outputs)
    labels = np.array(labels)
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou


def metric(probability, truth, threshold=0.5):
    """Calculates dice of positive and negative images seperately
    probability and truth must be torch tensors"""
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert probability.shape == truth.shape

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice.numpy(), dice_neg.numpy(), dice_pos.numpy(), num_neg, num_pos


class Meter:
    """A meter to keep track of iou and dice scores throughout an epoch"""

    def __init__(self, segmentation):
        self.segmentation = segmentation
        self.base_threshold = 0.5
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        probs = probs.detach().cpu()
        targets = targets.detach().cpu()
        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)
        self.base_dice_scores.extend(dice)
        self.dice_pos_scores.extend(dice_pos)
        self.dice_neg_scores.extend(dice_neg)
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.nanmean(self.base_dice_scores)
        dice_neg = np.nanmean(self.dice_neg_scores)
        dice_pos = np.nanmean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou


def epoch_log(epoch_loss, meter):
    """logging the metrics at the end of an epoch"""
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print(
        "Loss: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f | IoU: %0.4f"
        % (epoch_loss, dice, dice_neg, dice_pos, iou)
    )
    return dice, iou


def seed_all(seed: int):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    random.seed(seed)  # set fixed value for python built-in pseudo-random generator
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_model_stats():
    model_stats = [
        ele
        for ele in os.listdir("checkpoints")
        if "dice" in ele and "scratch" in ele and "Unet" in ele
    ]
    stats_df = pd.DataFrame()

    tsets = []
    model_names = []
    for idx, ele in enumerate(model_stats):
        model_names.append(ele)
        tset = []
        if "hand" in ele:
            tset.append("hand")
        if "watershed" in ele:
            tset.append("watershed")
        if "synthetic" in ele:
            tset.append("synthetic")
        tset = "_".join(tset)
        model_stats[idx] = model_stats[idx].replace(f"tsets_{tset}", "")
        tsets.append(tset)

    for idx, ele in enumerate(model_stats):
        (
            model,
            patch_size,
            lr,
            batch_size,
            finetuned,
            _,
            _,
            aug,
            _,
            ratio,
            _,
            loss,
            dice,
            iou,
            epoch,
        ) = ele.split("_")
        stats_df = stats_df.append(
            {
                "patch_size": int(patch_size),
                "lr": float(lr),
                "tset": tsets[idx],
                "aug": aug,
                "ratio": float(ratio),
                "finetuned": finetuned,
                "loss": loss,
                "batch_size": batch_size,
                "dice": round(float("".join(dice.split("-")[1:])), 3),
                "iou": round(float("".join(iou.split("-")[1:])), 3),
                "epochs": int(epoch.split("-")[-1].split(".")[0]),
                "model_name": model_names[idx],
            },
            ignore_index=True,
        )

    stats_df["dice_iou"] = stats_df.dice + stats_df.iou
    stats_df["dice_iou"] = stats_df.dice_iou.values / 2

    stats_df = stats_df.sort_values(by=["dice_iou"], ascending=False)
    return stats_df
