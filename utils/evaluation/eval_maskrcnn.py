import os
from typing import Union

import affine
import pandas as pd
import torch

import wandb
from fiona.crs import from_epsg
from shapely.geometry import Point
from torch.utils.data import DataLoader
from torchvision.models.detection import MaskRCNN, FasterRCNN
import geopandas as gpd


from utils.data_processing import TestDataset
from utils.evaluation.polygon_matching import bbox_to_pol, match_bbox_pols, match_points
from utils.evaluation.visualize_maskrcnn_boxes import create_vizualization_grid


def validate_maskrcnn(net, val_loader, device):
    tp = 0
    fp = 0
    fn = 0
    eps = 1e-7
    net.eval()

    with torch.no_grad():
        for epoch in range(1):
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [
                    {
                        k: v.to(device)
                        for k, v in t.items()
                        if k in ["masks", "boxes", "labels"]
                    }
                    for t in targets
                ]
                gt_boxes = [
                    v.detach().cpu().numpy()
                    for t in targets
                    for k, v in t.items()
                    if k == "boxes"
                ]
                pred_boxes = [
                    v.detach().cpu().numpy()
                    for p in net(images)
                    for k, v in p.items()
                    if k == "boxes"
                ]
                gt_pols = [bbox_to_pol(*bbox) for ele in gt_boxes for bbox in ele]
                pred_pols = [bbox_to_pol(*bbox) for ele in pred_boxes for bbox in ele]

                tp_batch, fp_batch, fn_batch = match_bbox_pols(gt_pols, pred_pols)
                tp += tp_batch
                fp += fp_batch
                fn += fn_batch

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall / (precision + recall + eps))

    output = net(images)
    output_grid = create_vizualization_grid(images=images, outputs=output, targets=targets)

    net.train()
    return f1, precision, recall, output_grid


def test_maskrcnn(
    device: torch.device,
    net: Union[MaskRCNN, FasterRCNN],
    test_dir: str,
    experiment_id: str,
    num_workers: int,
    batch_size: int,
    ground_truth_gdf: str,
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

    tp = 0
    fp = 0
    fn = 0
    eps = 1e-7  # To prevent division by zero when calculating precision / recall / f-1

    # Put model in eval mode
    net.eval()

    # Read scene stats csv
    scene_stats = pd.read_csv(f"{test_dir}/scene_stats.csv")

    # Start shapefile for predictions
    pred_points_support_box = []
    pred_points_support_mask = []
    pred_points = []
    pred_point_scenes = []
    pred_counts = []

    # Loop through dataloader

    for images, image_names in test_loader:

        images = list(image.to(device) for image in images)
        with torch.cuda.amp.autocast(enabled=amp):
            with torch.no_grad():

                outputs = net(images)
                pred_box_scores = [
                    v.detach().cpu().numpy()
                    for p in outputs
                    for k, v in p.items()
                    if k == "scores"
                ]

                pred_boxes = [
                    v.detach().cpu().numpy()
                    for p in outputs
                    for k, v in p.items()
                    if k == "boxes"
                ]
                if "masks" in outputs[0]:
                    pred_masks = [
                        v.detach().cpu().numpy()
                        for p in outputs
                        for k, v in p.items()
                        if k == "masks"
                    ]

                    support_mask = {
                        idx: [mask.sum() for mask in ele]
                        for idx, ele in enumerate(pred_masks)
                    }

                support_box = {
                    idx: ele
                    for idx, ele in enumerate(pred_box_scores)
                }

                centroids = {
                    idx: [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in ele]
                    for idx, ele in enumerate(pred_boxes)
                }

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
                        for idx2, centroid in enumerate(centroids[idx]):
                            x, y = centroid

                            # Add support for point from bbox
                            pred_points_support_box.append(
                                support_box[idx][idx2]
                            )

                            # Add support for point from mask, if available
                            if "masks" in outputs[0]:
                                pred_points_support_mask.append(
                                    support_mask[idx][idx2]
                                )

                            pred_counts.append(len(centroids[idx]))

                            # Add scene coordinates from patch filename and project with transforms
                            x, y = x + int(down), y + int(left)
                            pred_points.append(Point(*((x, y) * transform_scene)))
                            pred_point_scenes.append(scene)

    # Add points to shapefile
    preds_gdf = gpd.GeoDataFrame(
        {
            "geometry": pred_points,
            "scene": pred_point_scenes,
            "support_box": pred_points_support_box,
            "support_mask": pred_points_support_mask,
            "pred_count": pred_counts,
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
            fn += len(gt_points_scene)
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
            cutoffs=(-50.0,),
        )

        # Add test metrics for every cutoff
        for cutoff in (-50.0,):
            tp += scene_tp[cutoff]
            fp += scene_fp[cutoff]
            fn += scene_fn[cutoff]

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
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1_test = 2 * (precision * recall / (precision + recall + eps))

    # Store best test instance f-1
    experiment.log(
        {
            f"best test instance f1": f1_test,
            f"best test instance f1 precision": precision,
            f"best test instance f1 recall": recall,
        }
    )
