import os

import cv2
import pandas as pd
import rasterio
import geopandas as gpd
import numpy as np
import sys
from argparse import ArgumentParser
from rasterio.windows import Window
from shapely.geometry import Polygon

sys.path.insert(0, "..")
from utils.data_processing import Tiff


def parse_args():
    parser = ArgumentParser("Script to create binary masks from an esri shapefile")
    parser.add_argument(
        "-i",
        "--input-shapefile",
        dest="input_shapefile",
        type=str,
        help="Path to input shapefile with " "a 'geometry' and a 'scene' column",
    )
    parser.add_argument(
        "-s", "--scenes-dir", dest="scenes_dir", type=str, help="Path to folder with scenes as .tif files."
    )
    parser.add_argument(
        "-p",
        "--patch-size",
        dest="patch_size",
        type=int,
        default=768,
        help="Patch size for creating tiles.",
    )
    parser.add_argument(
        "-n",
        "--negatives-per-scene",
        dest="negatives_per_scene",
        type=int,
        default=250,
        help="Number of negative patches per scene.",
    )
    parser.add_argument(
        "-v",
        "--percent-val-negative",
        dest="percent_val_negative",
        type=float,
        default=0.2,
        help="Percent validation for negative patches",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=str,
        default="training_set",
        help="Path to output dir to save centroids binary mask.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Get half patch size for cropping rasters and masks
    half_patch = args.patch_size // 2

    # Read annotations and keep only images with seals
    annotations = gpd.read_file(args.input_shapefile)

    # Store filenames
    annotations_df = pd.DataFrame()

    # Assert annotations are valid
    for col in ["scene", "label", "geometry"]:
        assert (
            col in annotations
        ), f"Invalid annotations file {args.input_shapefile}, missing '{col}' column."
    annotations = annotations.loc[annotations.label.isin({"crabeater", "weddell"})]
    scenes = annotations.scene.unique()

    # Make output dir
    for split in annotations.dataset.unique():
        for subdir in ["x", "y"]:
            os.makedirs(f"{args.output_dir}/{split}/{subdir}", exist_ok=True)

    # Loop over scenes
    existing_negatives = []
    curr_catalog_id = ""
    tiff = Tiff()
    for scene in scenes:
        print(scene)
        scn_annotations = annotations.loc[annotations.scene == scene]

        # Get dataset split, catalog id and label for naming
        catalog_id = scene.split("_")[2]
        strip_number = scene.split("_")[5]

        if catalog_id != curr_catalog_id:
            existing_negatives = []
            curr_catalog_id = catalog_id
        catalog_annotations = annotations.loc[annotations.catalog_id == catalog_id]
        labels = scn_annotations.label.values
        splits = scn_annotations.dataset.values
        label_counters = {label: 0 for label in scn_annotations.label.unique()}

        try:
            img, width, height, transform, meta = tiff.process_raster(
                f"{args.scenes_dir}/{scene}"
            )
        except ValueError:
            print(f"Scene {scene} not present in directory {args.scenes_dir}.")
            continue

        # Fill in scene mask with seal centroids
        scene_mask = np.zeros((height, width), dtype=np.uint8)
        xs = [point.xy[0][0] for point in scn_annotations["geometry"]]
        ys = [point.xy[1][0] for point in scn_annotations["geometry"]]

        xs, ys = rasterio.transform.rowcol(transform=transform, xs=xs, ys=ys)
        if type(xs) == int:
            xs, ys = [xs], [ys]
        for point in zip(xs, ys):
            scene_mask[point] = 255

        scene_mask = cv2.dilate(
            scene_mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        )

        # Save output crops and crop masks
        with rasterio.open(f"{args.scenes_dir}/{scene}") as src:
            for idx, point in enumerate(zip(xs, ys)):

                # Read from scene mask and raster window
                x, y = point
                mask = scene_mask[
                    x - half_patch : x + half_patch, y - half_patch : y + half_patch
                ]

                window = Window(
                    row_off=x - half_patch,
                    col_off=y - half_patch,
                    width=args.patch_size,
                    height=args.patch_size,
                )
                crop = src.read(1, window=window)

                assert crop.shape == (args.patch_size, args.patch_size), "invalid crop"

                # Get filename and save
                label = labels[idx]
                split = splits[idx]
                filename = (
                    f"{catalog_id}-{strip_number}_{label}_{label_counters[label]}.tif"
                )

                cv2.imwrite(f"{args.output_dir}/{split}/x/{filename}", crop)
                cv2.imwrite(f"{args.output_dir}/{split}/y/{filename}", mask)
                label_counters[label] += 1

                # Add entry to annotation
                annotations_df = annotations_df.append(
                    {
                        "label": label,
                        "split": split,
                        "img_name": filename,
                        "scene": scene,
                        "catalog_id": catalog_id,
                    },
                    ignore_index=True,
                )

            # Add negative patches
            prev_len = len(existing_negatives)
            while len(existing_negatives) - prev_len < args.negatives_per_scene:

                # Sample x, y at random
                x, y = np.random.randint(0, height), np.random.randint(0, width)
                curr_point = np.array([x, y])

                # Make sure the crop center is not too close to an existing negative crop
                for point in existing_negatives:
                    if np.linalg.norm(point - curr_point) < half_patch:
                        continue

                # Get window and ake sure the crop has non-zero content
                window = Window(
                    row_off=x - half_patch,
                    col_off=y - half_patch,
                    width=args.patch_size,
                    height=args.patch_size,
                )

                crop = src.read(1, window=window)

                # Make sure the crop has the correct shape
                if crop.shape != (args.patch_size, args.patch_size):
                    continue

                # Make sure the crop has non-missing data
                if np.sum(crop) == 0:
                    continue

                # Make sure the crop does not contain any annotated positives
                crop_polygon = Polygon(
                    [
                        transform * ele
                        for ele in [
                            [y - half_patch, x - half_patch],
                            [y + half_patch, x - half_patch],
                            [y + half_patch, x + half_patch],
                            [y - half_patch, x + half_patch],
                        ]
                    ]
                )

                if catalog_annotations.geometry.intersects(crop_polygon).max():
                    continue

                # Save file and add point to existing negatives
                split = (
                    "training"
                    if np.random.rand() > args.percent_val_negative
                    else "validation"
                )
                filename = (
                    f"{catalog_id}-{strip_number}_negative_"
                    f"{len(existing_negatives) - prev_len}.tif"
                )
                cv2.imwrite(f"{args.output_dir}/{split}/x/{filename}", crop)
                existing_negatives.append(curr_point)

                # Add entry to annotation df
                annotations_df = annotations_df.append(
                    {
                        "label": "negative",
                        "split": split,
                        "img_name": filename,
                        "scene": scene,
                        "catalog_id": catalog_id,
                    },
                    ignore_index=True,
                )

    # Save annotations df
    annotations_df.to_csv(f"{args.output_dir}/annotations_df.csv", index=False)


if __name__ == "__main__":
    main()
