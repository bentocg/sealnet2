import os
import shutil
from typing import Union

import affine
import cv2
import numpy as np
import pandas as pd
import torch
import wandb
from fiona.crs import from_epsg
from shapely.geometry import Point
from tqdm import tqdm
from torch.utils.data import DataLoader
from segmentation_models_pytorch import Unet
import geopandas as gpd

from utils.data_processing import TestDataset
from utils.evaluation.extract_centroids import extract_centroids
from utils.evaluation.polygon_matching import match_points
from utils.evaluation.unet_instance_f1_score import unet_instance_f1_score_thresh
from utils.evaluation.dice_score import dice_coeff
from utils.models.transunet import TransUnet


def validate_unet(
    net: Union[Unet, TransUnet], dataloader: DataLoader, device: torch.device, amp: bool = False
) -> (float, float, float, float, float):
    """
    Validation loop for SealNet2. Matches predicted polygons to ground truth polygons to get
    instance level f1-score.

    :param net: pytorch model that predicts masks and counts
    :param dataloader: dataloader with validation set
    :param device: torch device for running validation
    :param amp: use auto-mixed precision?

    :return: f-1 score, precision, recall, pixel-level dice, and count MAE, respectively
    """

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
                (
                    batch_f1,
                    batch_precision,
                    batch_recall,
                    batch_mae,
                ) = unet_instance_f1_score_thresh(
                    true_masks=true_masks,
                    true_counts=true_counts,
                    pred_masks=pred_masks,
                    pred_counts=pred_counts,
                    threshold=0.5,
                )
                f1_score += batch_f1
                precision += batch_precision
                recall += batch_recall
                count_mae += batch_mae

                # Calculate dice coefficient
                pred_masks = (torch.sigmoid(pred_masks) > 0.5).float()
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                dice_score += dice_coeff(
                    pred_masks, true_masks, reduce_batch_first=False
                )

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


def test_unet(
    device: torch.device,
    net: Union[Unet, TransUnet],
    test_dir: str,
    experiment_id: str,
    num_workers: int,
    batch_size: int,
    ground_truth_gdf: str,
    threshold: float = 0.5,
    nms_distance: float = 1.0,
    match_distance: float = 1.5,
    amp: bool = False,
) -> None:
    """
    Test loop for SealNet2. Compares mosaics for entire testing scenes to scene-level groundtruth
    masks to get a real-world estimate for instance f-1 score. Statistics are saved directly to
    wandb project.

    :param device: device for running test
    :param net: pytorch model predicting masks and counts
    :param test_dir: directory with test set
    :param experiment_id: experiment id for saving statistics
    :param num_workers: number of workers for dataloader
    :param batch_size: batch size for dataloader
    :param threshold: threshold for binarizing output masks (applied after sigmoid transform)
    :param amp: use auto-mixed precision?
    """
    # Resume experiment
    experiment = wandb.init(
        project="SealNet2.0", resume="allow", anonymous="must", id=experiment_id
    )

    # Create Dataloader for test scenes
    test_loader = DataLoader(
        dataset=TestDataset(f"{test_dir}/x"),
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
    )

    # Store Statistics
    tp = 0
    fp = 0
    fn = 0
    eps = 1e-8

    # Put model in eval mode
    net = net.eval()

    # Read scene stats csv
    scene_stats = pd.read_csv(f"{test_dir}/scene_stats.csv")

    # Start shapefile for predictions
    pred_points_support = []
    pred_points = []
    pred_point_scenes = []

    # Loop through dataloader

    for images, image_names in test_loader:

        images = images.to(device)
        with torch.cuda.amp.autocast(enabled=amp):
            with torch.no_grad():
                outputs, _ = net(images)

                # Threshold output
                outputs = (
                    torch.sigmoid(outputs).squeeze(1).detach().float().cpu().numpy()
                    * 255
                )
                outputs_bin = (outputs > threshold).astype(np.uint8)

                # Extract predicted centroids and scene names
                centroids = [extract_centroids(pred_mask) for pred_mask in outputs_bin]
                scenes = [
                    f"{'_'.join(os.path.basename(img_name).split('_')[:-4])}.tif"
                    for img_name in image_names
                ]

                # Project centroids and store support level for each centroid for NMS
                for idx, scene in enumerate(scenes):
                    if centroids[idx]:
                        transform_scene = affine.Affine(
                            *scene_stats.loc[scene_stats.scene == scene].values[0, 3:]
                        )
                        for centroid in centroids[idx]:
                            x, y = centroid
                            pred_points.append(Point(*((x, y) * transform_scene)))
                            x, y = int(round(x)), int(round(y))  # Convert to integer for indexing
                            pred_points_support.append(
                                outputs[
                                    idx,
                                    max(0, x - 1) : x + 2,
                                    y - 1 : min(y + 2, outputs[idx].shape[-1]),
                                ].sum()
                            )
                            pred_point_scenes.append(scene)

    # Add points to shapefile
    preds_gdf = gpd.GeoDataFrame(
        {
            "geometry": pred_points,
            "scene": pred_point_scenes,
            "support": pred_points_support,
            "ids": list(range(len(pred_points))),
        },
        crs=from_epsg(3031),
    )
    preds_gdf.to_file("preds_shape.shp")

    # Read groundtruth gdf
    gt_gdf = gpd.read_file(ground_truth_gdf)

    # Run non-maximum suppression
    for scene in gt_gdf.scene.unique():
        gt_points_scene = gt_gdf.loc[gt_gdf.scene == scene]
        if scene not in preds_gdf.scene.unique():
            fn += len(gt_points_scene)
            continue
        points_scene = preds_gdf.loc[preds_gdf.scene == scene]
        points_scene = points_scene.sort_values(
            by="support", ascending=False
        ).reset_index()
        to_keep = set([])
        while True:
            if len(points_scene) < 2:
                break
            curr = points_scene.iloc[0]["geometry"]
            to_keep.add(points_scene.iloc[0]["ids"])
            curr_pol = curr.buffer(nms_distance)
            points_scene = points_scene.loc[~(points_scene.geometry.intersects(curr_pol))]
        points_scene = preds_gdf.loc[preds_gdf.ids.isin(to_keep)]

        # Compare with groundtruth
        scene_tp, scene_fp, scene_fn = match_points(
            gt_points_scene.geometry.values,
            points_scene.geometry.values,
            match_distance=match_distance,
        )
        tp += scene_tp
        fp += scene_fp
        fn += scene_fn

        # Calculate scene statistics and store to wandb
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall / (precision + recall + eps))
        experiment.log(
            {
                f"test instance f1 {scene}": f1,
                f"test instance precision {scene}": precision,
                f"test instance recall {scene}": recall,
            }
        )

    # Calculate global test statistics and store to wandb
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
