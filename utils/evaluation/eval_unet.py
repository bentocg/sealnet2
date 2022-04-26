import os
import shutil

import cv2
import pandas as pd
import torch
import wandb
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.data_processing import TestDataset, write_output, merge_output
from utils.evaluation.polygon_matching import match_polygons
from utils.evaluation.unet_instance_f1_score import unet_instance_f1_score_thresh
from utils.evaluation.dice_score import dice_coeff


def validate_unet(net, dataloader, device, amp=False):
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
        true_counts = true_counts.to(device=device, dtype=torch.float32)

        with torch.cuda.amp.autocast(enabled=amp):
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


def test_unet(device, net: nn.Module, test_dir: str, experiment_id: str, num_workers: int,
              batch_size: int, threshold: float = 0.5, amp: bool = False):
    # Resume experiment
    experiment = wandb.init(
        project="SealNet2.0", resume="allow", anonymous="must", id=experiment_id
    )

    # Store statistics
    tp = 0
    fp = 0
    fn = 0

    # Create Dataloader for test scenes
    test_loader = DataLoader(dataset=TestDataset(f"{test_dir}/x"),
                             num_workers=num_workers,
                             batch_size=batch_size,
                             shuffle=False)

    # Put model in eval mode
    net = net.eval()

    # Read scene stats csv
    scene_stats = pd.read_csv(f"{test_dir}/scene_stats.csv")

    # Loop through dataloader
    out_folder = f"{experiment_id}_temp_output"
    for images, image_names in test_loader:

        images = images.to(device)
        with torch.cuda.amp.autocast(enabled=amp):
            with torch.no_grad():
                outputs, _ = net(images)

                # Threshold output
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > threshold).squeeze(1).detach().float() * 255

                # Save output
                write_output(outputs, image_names, out_folder)

    # Merge output for each scene and match with GT mask
    for scene in os.listdir(out_folder):
        stats_scene = scene_stats.loc[scene_stats.scene == scene]
        mask_shape = (stats_scene.height, stats_scene.width)
        pred_mask = merge_output(f"{out_folder}/{scene}", mask_shape)
        true_mask = cv2.imread(f"{test_dir}/y/{scene}", 0)
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

    # Remove temp folders
    shutil.rmtree(out_folder)

