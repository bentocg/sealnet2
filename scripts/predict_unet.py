import logging
import os
import shutil
import sys
from collections import OrderedDict

import affine
import numpy as np
import rasterio
import torch
import ttach
from fiona.crs import from_epsg
from shapely.geometry import Point
from torch.utils.data import DataLoader
import geopandas as gpd
from segmentation_models_pytorch import Unet

sys.path.insert(0, "../")

from utils.data_processing import TestDataset, tile_image
from utils.models.tta_wrapper import SegmentationRegTTAWrapper

from argparse import ArgumentParser

from utils.evaluation.extract_centroids import extract_centroids
from utils.models.model_factory import get_semantic_segmentation_model


def parse_args():
    parser = ArgumentParser("Script to predict seals on new scenes")
    parser.add_argument(
        "--input-raster",
        "-i",
        dest="input_raster",
        type=str,
        help="Path to input raster",
    )
    parser.add_argument(
        "--patch-size",
        "-ps",
        dest="patch_size",
        type=int,
        default=512,
        help="Patch size for tiling input raster",
    )
    parser.add_argument(
        "--stride",
        "-s",
        dest="stride",
        type=float,
        default=0.5,
        help="Amount of overlap between neighboring patches",
    )
    parser.add_argument(
        "--tta", "-t", type=int, default=1, help="Use test-time augmentation?"
    )
    parser.add_argument(
        "--model-checkpoint",
        "-mc",
        dest="model_checkpoint",
        type=str,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model-architecture",
        "-ma",
        dest="model_architecture",
        type=str,
        help="Model architecture for checkpoint",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=str,
        default="prediction_shapefiles",
        help="Path to folder where prediction shapefile will be saved",
    )
    parser.add_argument(
        "--batch-size",
        "-bs",
        dest="batch_size",
        type=int,
        default=20,
        help="Batch size",
    )
    return parser.parse_args()


def predict_unet(
    device: torch.device,
    net: Unet,
    input_dir: str,
    num_workers: int,
    batch_size: int,
    transforms: affine.Affine,
    threshold_mask: float = 0.5,
    nms_distance: float = 1.0,
    amp: bool = False,
) -> gpd.GeoDataFrame:
    """
    Prediction loop for SealNet2.

    :param device: device for running test
    :param net: pytorch model predicting masks and counts
    :param input_dir: directory with tiles to process
    :param num_workers: number of workers for dataloader
    :param batch_size: batch size for dataloader
    :param transforms: affine matrix for input raster
    :param threshold_mask: threshold for binarizing output masks (applied after sigmoid transform)
    :param nms_distance: distance threshold for NMS (removing sub-optimal overlapping predictions)
    :param amp: use auto-mixed precision?
    """

    # Create Dataloader for test scenes
    test_loader = DataLoader(
        dataset=TestDataset(input_dir),
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
    )

    # Put model in eval mode
    net = net.eval()

    # Start shapefile for predictions
    pred_points_support = []
    pred_points = []
    pred_counts = []
    tile_image_names = []
    pred_points_distance_from_center = []

    for images, image_names in test_loader:

        images = images.to(device)
        with torch.cuda.amp.autocast(enabled=amp):
            with torch.no_grad():

                outputs, counts = net(images)

                # Threshold output
                outputs = (
                    torch.sigmoid(outputs).squeeze(1).detach().float().cpu().numpy()
                    * 255
                )
                outputs_bin = (outputs > threshold_mask).astype(np.uint8)

                # Convert counts to numpy
                counts = counts.squeeze(1).detach().float().cpu().numpy()

                # Extract predicted centroids and scene names
                centroids = [extract_centroids(pred_mask) for pred_mask in outputs_bin]
                corners = [
                    os.path.basename(img_name).split("_")[-4:-2]
                    for img_name in image_names
                ]

                # Project centroids and store support level for each centroid for NMS
                for idx, corner in enumerate(corners):
                    if centroids[idx]:
                        left, down = corner
                        for centroid in centroids[idx]:
                            x, y = centroid

                            # Get distance from center of the tile
                            distance_from_center = np.linalg.norm(
                                np.array([x, y])
                                - np.array(
                                    [
                                        outputs[idx].shape[-1] // 2,
                                        outputs[idx].shape[-1] // 2,
                                    ],
                                ),
                            )
                            pred_points_distance_from_center.append(
                                distance_from_center
                            )

                            # Add support for point
                            pred_points_support.append(
                                outputs[
                                    idx,
                                    max(0, round(y) - 1) : min(
                                        round(y) + 2, outputs[idx].shape[-1]
                                    ),
                                    max(0, round(x) - 1) : min(
                                        round(x) + 2, outputs[idx].shape[-1]
                                    ),
                                ].sum()
                            )

                            # Add scene coordinates from patch filename and project with transforms
                            x, y = x + int(down), y + int(left)
                            pred_points.append(Point(*((x, y) * transforms)))
                            pred_counts.append(counts[idx])
                            tile_image_names.append(image_names[idx])

    # Add points to shapefile
    preds_gdf = gpd.GeoDataFrame(
        {
            "geometry": pred_points,
            "support": pred_points_support,
            "pred_count": pred_counts,
            "distance_from_center": pred_points_distance_from_center,
            "ids": list(range(len(pred_points))),
        },
        crs=from_epsg(3031),
    )

    # Apply NMS
    to_keep = set([])
    points_scene = preds_gdf.copy()
    points_scene = points_scene.sort_values(by="support", ascending=False).reset_index()

    # Remove overlapping points with less support
    while True:
        if len(points_scene) < 2:
            break
        curr = points_scene.iloc[0]["geometry"]
        to_keep.add(points_scene.iloc[0]["ids"])
        curr_pol = curr.buffer(nms_distance)
        points_scene = points_scene.loc[~(points_scene.geometry.intersects(curr_pol))]

    preds_gdf = preds_gdf.loc[(preds_gdf.ids.isin(to_keep))]
    return preds_gdf


if __name__ == "__main__":
    args = parse_args()

    # Start logging and get device
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tile image
    scene = os.path.basename(args.input_raster)
    temp_dir = f"temp_output_{scene}"
    with rasterio.open(args.input_raster) as src:
        transforms = src.transform
        tile_image(
            img=src.read([1]),
            patch_size=args.patch_size,
            stride=args.stride,
            out_dir=temp_dir,
            scene=scene,
        )

    # Instantiate model and load checkpoint
    net = get_semantic_segmentation_model(
        model_architecture=args.model_architecture,
        dropout_regression=0.0,
        patch_size=args.patch_size,
    )
    state_dict = torch.load(args.model_checkpoint, map_location=torch.device("cpu"))
    state_dict = OrderedDict(
        {key.replace("model.", ""): val for key, val in state_dict.items()}
    )
    net.load_state_dict(state_dict)
    net.to(device=device)
    net.eval()

    # Apply tta
    if args.tta:
        net = SegmentationRegTTAWrapper(
            model=net, transforms=ttach.aliases.d4_transform()
        )

    # Predict on tiles
    preds_gdf = predict_unet(
        device=device,
        net=net,
        input_dir=temp_dir,
        num_workers=1,
        transforms=transforms,
        batch_size=args.batch_size,
    )
    preds_gdf["scene"] = scene

    # Remove temporary files
    shutil.rmtree(temp_dir)

    # Store output
    os.makedirs(args.out_dir, exist_ok=True)

    preds_gdf.to_file(f"{args.out_dir}/{scene.split('.')[0]}.shp")
