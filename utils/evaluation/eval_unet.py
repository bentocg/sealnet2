import os
from collections import defaultdict
from typing import Union, Tuple

import affine
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
    net: Union[Unet, TransUnet],
    dataloader: DataLoader,
    device: torch.device,
    amp: bool = False,
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
    cutoffs: Tuple[float] = (-50.0, 0.0, 0.25, 0.5, 0.75, 1.0),
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
    :param ground_truth_gdf: path to shapefile with groundtruth points
    :param threshold: threshold for binarizing output masks (applied after sigmoid transform)
    :param nms_distance: distance for non-maximum supression removal of redundant poitnts
    :param cutoffs: cutoffs for using regression predictions to remove false positives
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

    # Store statistics using regression to filter out false-positives
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    eps = 1e-7  # To prevent division by zero when calculating precision / recall / f-1

    # Put model in eval mode
    net.eval()

    # Read scene stats csv
    scene_stats = pd.read_csv(f"{test_dir}/scene_stats.csv")

    # Start shapefile for predictions
    pred_points_support = []
    pred_points = []
    pred_point_scenes = []
    pred_points_distance_from_center = []
    pred_points_tile_ids = []
    pred_counts = []

    # Loop through dataloader

    for batch_idx, images, image_names in enumerate(test_loader):

        images = images.to(device)
        with torch.cuda.amp.autocast(enabled=amp):
            with torch.no_grad():

                outputs, counts = net(images)

                # Threshold output
                outputs = (
                    torch.sigmoid(outputs).squeeze(1).detach().float().cpu().numpy()
                    * 255
                )
                outputs_bin = (outputs > threshold).astype(np.uint8)

                # Convert counts to numpy
                counts = counts.squeeze(1).detach().float().cpu().numpy()

                # Extract predicted centroids and scene names
                centroids = [extract_centroids(pred_mask) for pred_mask in outputs_bin]
                scenes = [
                    f"{'_'.join(os.path.basename(img_name).split('_')[:-4])}.tif"
                    for img_name in image_names
                ]
                corners = [
                    os.path.basename(img_name).split("_")[-4:-2]
                    for img_name in image_names
                ]

                # Project centroids and store support level for each centroid for NMS
                for idx, scene in enumerate(scenes):
                    if centroids[idx]:
                        transform_scene = affine.Affine(
                            *scene_stats.loc[scene_stats.scene == scene].values[0, 3:]
                        )
                        left, down = corners[idx]
                        for centroid in centroids[idx]:
                            x, y = centroid

                            # Add support for point
                            pred_points_support.append(
                                outputs[
                                    idx,
                                    max(0, round(y) - 1): min(round(y) + 2, outputs[idx].shape[-1]),
                                    max(0, round(x) - 1): min(round(x) + 2, outputs[idx].shape[-1]),
                                ].sum()
                            )

                            # Get distance from center of the tile
                            distance_from_center = np.linalg.norm(
                                np.array([x, y]) -
                                np.array(
                                    [outputs[idx].shape[-1] // 2,
                                     outputs[idx].shape[-1] // 2],
                                ),
                            )
                            pred_points_distance_from_center.append(distance_from_center)

                            # Add scene coordinates from patch filename and project with transforms
                            x, y = x + int(down), y + int(left)
                            pred_points.append(Point(*((x, y) * transform_scene)))
                            pred_counts.append(counts[idx])
                            pred_point_scenes.append(scene)
                            pred_points_tile_ids.append(f"{batch_idx}_{idx}")

    # Add points to shapefile
    preds_gdf = gpd.GeoDataFrame(
        {
            "geometry": pred_points,
            "scene": pred_point_scenes,
            "support": pred_points_support,
            "pred_count": pred_counts,
            "distance_from_center": pred_points_distance_from_center,
            "tile_id": pred_points_tile_ids,
            "ids": list(range(len(pred_points))),
        },
        crs=from_epsg(3031),
    )

    # Read groundtruth gdf
    gt_gdf = gpd.read_file(ground_truth_gdf)

    # Run non-maximum suppression
    scenes_to_process = set(gt_gdf.scene.unique()).union(preds_gdf.scene.unique())
    to_keep = set([])
    for scene in scenes_to_process:
        gt_points_scene = gt_gdf.loc[gt_gdf.scene == scene]

        # Add empty geometry for scenes with no GT points
        if len(gt_points_scene) == 0:
            gt_points_scene["geometry"] = []

        # Add false-negatives for scenes present in GT but missing in preds
        if scene not in preds_gdf.scene.unique():
            for cutoff in cutoffs:
                fn[cutoff] += len(gt_points_scene)
            continue

        # Subset predictions and run NMS
        points_scene = preds_gdf.loc[preds_gdf.scene == scene]
        points_scene = points_scene.sort_values(
            by="support", ascending=False
        ).reset_index()
        while True:
            if len(points_scene) < 2:
                break
            curr = points_scene.iloc[0]["geometry"]
            to_keep.add(points_scene.iloc[0]["ids"])
            curr_pol = curr.buffer(nms_distance)
            points_scene = points_scene.loc[
                ~(points_scene.geometry.intersects(curr_pol))
            ]
        points_scene = preds_gdf.loc[
            (preds_gdf.ids.isin(to_keep)) & (preds_gdf.scene == scene)
        ]

        # Compare with groundtruth
        scene_tp, scene_fp, scene_fn = match_points(
            true_points=gt_points_scene.geometry.values,
            pred_points=points_scene.geometry.values,
            pred_counts=points_scene.pred_count.values,
            match_distance=match_distance,
            cutoffs=cutoffs,
        )

        # Add test metrics for every cutoff
        for cutoff in cutoffs:
            tp[cutoff] += scene_tp[cutoff]
            fp[cutoff] += scene_fp[cutoff]
            fn[cutoff] += scene_fn[cutoff]

            # Calculate scene statistics and store to wandb
            if cutoff == -50.0:
                precision = scene_tp[cutoff] / (
                    scene_tp[cutoff] + scene_fp[cutoff] + eps
                )
                recall = scene_tp[cutoff] / (scene_tp[cutoff] + scene_fn[cutoff] + eps)
                f1 = 2 * (precision * recall / (precision + recall + eps))
                experiment.log(
                    {
                        f"test instance f1 {scene}": f1,
                        f"test instance precision {scene}": precision,
                        f"test instance recall {scene}": recall,
                    }
                )

    # Store predictions
    os.makedirs("predicted_shapefiles", exist_ok=True)
    preds_gdf = preds_gdf.loc[preds_gdf.ids.isin(to_keep)]
    preds_gdf.to_file(f"predicted_shapefiles/{experiment_id}.shp")

    # Calculate global test statistics and store to wandb
    best_f1_test = 0
    best_f1_test_recall = 0
    best_f1_test_precision = 0
    for cutoff in cutoffs:
        precision = tp[cutoff] / (tp[cutoff] + fp[cutoff] + eps)
        recall = tp[cutoff] / (tp[cutoff] + fn[cutoff] + eps)
        f1 = 2 * (precision * recall / (precision + recall + eps))
        experiment.log(
            {
                f"test instance f1{(' ' + str(cutoff) if cutoff != -50.0 else '')}": f1,
                f"test instance precision{(' ' + str(cutoff) if cutoff != -50.0 else '')}": precision,
                f"test instance recall{(' ' + str(cutoff) if cutoff != -50.0 else '')}": recall,
            }
        )
        if f1 > best_f1_test:
            best_f1_test = f1
            best_f1_test_precision = precision
            best_f1_test_recall = recall

    # Store best test instance f-1
    experiment.log(
        {
            f"best test instance f1": best_f1_test,
            f"best test instance f1 precision": best_f1_test_precision,
            f"best test instance f1 recall": best_f1_test_recall,
        }
    )
