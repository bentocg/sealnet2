import os

import cv2
import numpy as np
import rasterio
import torch
import wandb
from shapely.geometry import Polygon
from torch import nn
from tqdm import tqdm

from utils.data_processing import TestDataset, write_output, merge_output
from utils.evaluation.unet_instance_f1_score import unet_instance_f1_score_thresh
from utils.evaluation.dice_score import dice_coeff


def match_polygons(true_mask, pred_mask):

    # Find elements in true and pred masks
    true_contours, _ = cv2.findContours(true_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

    # Reformat contours
    true_contours = [np.array([ele[0] for ele in coords]) for coords in true_contours if
                     len(coords) > 2]
    pred_contours = [np.array([ele[0] for ele in coords]) for coords in pred_contours if
                     len(coords) > 2]

    # Convert into polygons
    true_polygons = [Polygon(coords).buffer(0) for coords in true_contours]
    pred_polygons = [Polygon(coords).buffer(0) for coords in pred_contours]

    # Match true polygons
    matched = set([])
    for idx_true, true_pol in enumerate(true_polygons):
        for idx_pred, pred_pol in enumerate(pred_polygons):
            if idx_pred in matched:
                continue
            if true_pol.intersects(pred_pol):
                matched.add(idx_pred)
                break

    tp = len(matched)
    fp = len(pred_polygons) - len(matched)
    fn = len(true_polygons) - len(matched)

    return tp, fp, fn


def validate_unet(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    f1_score = 0
    precision = 0
    recall = 0
    count_mae = 0
    dice_score = 0

    # Iterate over the validation set
    for images, true_counts, _, true_masks in tqdm(
        dataloader,
        total=num_val_batches,
        desc="Validation round",
        unit="batch",
        leave=False,
    ):
        # Move images and labels to correct device and type
        images = images.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # Predict the mask and count
            pred_masks, pred_counts = net(images)

            # Calculate instance level f1 score after thresholding
            batch_f1, batch_precision, batch_recall, batch_mae = unet_instance_f1_score_thresh(
                true_masks=true_masks,
                true_counts=true_counts,
                pred_masks=pred_masks,
                pred_counts=pred_counts,
                threshold=0.5
            )
            f1_score += batch_f1
            precision += batch_precision
            recall += batch_recall
            count_mae += batch_mae

            # Calculate dice coefficient
            pred_masks = (torch.sigmoid(pred_masks) > 0.5).float()
            true_masks = true_masks.to(device=device, dtype=torch.float32)
            dice_score += dice_coeff(pred_masks, true_masks, reduce_batch_first=False)

    # Revert network to training
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return f1_score, precision, recall, dice_score, count_mae
    return (
        f1_score / num_val_batches,
        precision / num_val_batches,
        recall / num_val_batches,
        dice_score / num_val_batches,
        count_mae / num_val_batches,
    )


def test_unet(net: nn.Module, test_dir: str, experiment_id: str, device: str, test_scenes_dir: str):
    # Resume experiment
    experiment = wandb.init(
        project="SealNet2.0", resume="allow", anonymous="must", id=experiment_id
    )

    # Store statistics
    tp = 0
    fp = 0
    fn = 0

    # Create Dataloader for test scenes
    test_loader = TestDataset(test_dir)

    # Put model in eval mode
    net = net.eval()

    # Loop through dataloader
    out_folder = f"{experiment_id}_temp_output"
    for images, image_names in test_loader:

        images = images.to(device)
        outputs = net(images)

        # Save output
        write_output(outputs, image_names, out_folder)

    # Merge output for each scene and match with GT mask
    for scene in os.listdir(out_folder):
        with rasterio.open(f"{test_scenes_dir}/{scene}") as src:
            mask_shape = (src.height, src.width)
        pred_mask = merge_output(f"{out_folder}/{scene}", mask_shape)
        true_mask = cv2.imread(f"{test_dir}/y]{scene}", 0)
        tp_scene, fp_scene, fn_scene = match_polygons(true_mask=true_mask, pred_mask=pred_mask)

        tp += tp_scene
        fp += fp_scene
        fn += fn_scene

    # Store statistics
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall / (precision + recall + eps))

    experiment.log(
        {
            "test instance f1": f1,
            "test instance precision": precision,
            "test instance recall": recall,
        }
    )

