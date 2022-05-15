import json
import os
from collections import defaultdict

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


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


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
        "-s",
        "--scenes-dir",
        dest="scenes_dir",
        type=str,
        help="Path to folder with scenes as .tif files.",
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
        default="../training_set",
        help="Path to output dir to save centroids binary mask.",
    )
    parser.add_argument(
        "-b",
        "--bbox-size",
        dest="bbox_size",
        type=int,
        default=11,
        help="Bounding box size for COCO annotations"
    )

    return parser.parse_args()


def main():
    """
    Generates training set from shapefile points and scene rasters.
    """
    args = parse_args()

    # Max tries for negative tiles generation
    MAX_TRIES = 2500

    # Get half patch size for cropping rasters and masks
    half_patch = args.patch_size // 2

    # Read annotations and keep only images with seals
    annotations = gpd.read_file(args.input_shapefile)

    # Store filenames
    annotations_df = pd.DataFrame()

    # Store annotations json for COCO format
    annotations_coco = {
        split: {
            "images": defaultdict(dict),
            "annotations": defaultdict(list),
            "info": {
                "contributor": None,
                "date_created": "2022-5-01T16:11:15.258399+00:00",
                "description": "sealnet_binary",
                "url": None,
                "version": None,
                "year": 2022,
            },
            "licenses": None,
            "categories": [{"name": "seal", "category": "seal", "id": 1}],
        }
        for split in annotations.dataset.unique()
    }

    annotation_id = {split: 0 for split in annotations.dataset.unique()}
    img_id = {split: 0 for split in annotations.dataset.unique()}

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

                # Add entry to annotation df
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

                # Add coco format img
                annotations_coco[split]["images"][img_id[split]] = {
                        "file_name": filename,
                        "height": args.patch_size,
                        "width": args.patch_size,
                        "id": img_id[split],
                }

                # Find all centroids in mask
                contours, _ = cv2.findContours(
                    cv2.dilate(mask, np.ones((3, 3))), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:

                    # Get bounding box for contour
                    top = max(0, min(ele[0][0] for ele in contour) - args.bbox_size // 2)
                    bottom = max(0, min(ele[0][1] for ele in contour) - args.bbox_size // 2)
                    bbox = [
                        top,
                        bottom,
                        min(args.bbox_size, args.patch_size - top),
                        min(args.bbox_size, args.patch_size - bottom)
                    ]

                    # Get segmentation mask for contour
                    contour = (
                        contour.flatten().tolist()
                    )  # Flatten and convert to list for json

                    area = bbox[2] * bbox[3]
                    annotations_coco[split]["annotations"][img_id[split]].append(
                        {
                            "image_id": img_id[split],
                            "area": area,
                            "bbox": bbox,
                            "id": annotation_id[split],
                            "category_id": 1,
                            "iscrowd": 0,
                            "segmentation": [contour],
                        }
                    )
                    annotation_id[split] += 1

                img_id[split] += 1

            # Add negative patches
            prev_len = len(existing_negatives)
            tries = 0
            while len(existing_negatives) - prev_len < args.negatives_per_scene:

                # Sample x, y at random
                x, y = np.random.randint(0, height), np.random.randint(0, width)
                curr_point = np.array([x, y])

                # Make sure the crop center is not too close to an existing negative crop
                for point in existing_negatives:
                    if type(point) != np.ndarray:  # Ignore fake points
                        continue
                    if np.linalg.norm(point - curr_point) < half_patch:
                        tries += 1
                        if tries > MAX_TRIES:

                            # Add fake points so negative patch creation breaks out of while loop
                            existing_negatives = existing_negatives + [False] * (
                                args.negatives_per_scene
                            )
                            print(
                                "Exceeded max tries, skipping negative tile generation."
                            )
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
                    tries += 1
                    if tries > MAX_TRIES:

                        # Add fake points so negative patch creation breaks out of while loop
                        existing_negatives = existing_negatives + [False] * (
                            args.negatives_per_scene
                        )
                        print("Exceeded max tries, skipping negative tile generation.")
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
        print(f"Finished processing {scene}")

    # Save annotations df
    annotations_df.to_csv(f"{args.output_dir}/annotations_df.csv", index=False)

    # Save JSON dataset
    for split in annotations_coco:
        with open(f"{args.output_dir}/{split}/annotations_coco.json", "w") as fout:
            json.dump(annotations_coco[split], fout, cls=NpEncoder)


if __name__ == "__main__":
    main()
