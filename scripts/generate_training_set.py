import os

import cv2
import rasterio
import geopandas as gpd
import numpy as np
import sys
from argparse import ArgumentParser
from rasterio.windows import Window

sys.path.insert(0, "..")
from utils.data_processing import Tiff


def parse_args():
    parser = ArgumentParser("Script to create binary masks from an esri shapefile")
    parser.add_argument(
        "-i",
        "--input_shapefile",
        type=str,
        help="Path to input shapefile with " "a 'geometry' and a 'scene' column",
    )
    parser.add_argument(
        "-s", "--scenes_dir", type=str, help="Path to folder with scenes as .tif files."
    )
    parser.add_argument(
        "-p",
        "--patch_size",
        type=int,
        default=768,
        help="Patch size for creating tiles",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
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
    tiff = Tiff()
    for scene in scenes:
        scn_annotations = annotations.loc[annotations.scene == scene]

        # Get dataset split, catalog id and label for naming
        catalog_id = f"{scene.split('_')[2]}-{scene.split('_')[5]}"
        labels = annotations.label.values
        splits = annotations.dataset.values
        label_counters = {label: 0 for label in scn_annotations.label}

        try:
            img, width, height, transform, meta = tiff.process_raster(
                f"{args.scenes_dir}/{scene}"
            )
        except ValueError:
            print(f"Scene {scene} not present in directory {args.scenes_dir}.")
            continue

        # Fill in scene mask with seal centroids
        scene_mask = np.zeros((width, height), dtype=np.uint8)
        xs = [point.xy[0][0] for point in scn_annotations["geometry"]]
        ys = [point.xy[1][0] for point in scn_annotations["geometry"]]

        ys, xs = rasterio.transform.rowcol(
            transform=transform, xs=xs, ys=ys
        )  # Indexing is inverted for rasterio.
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
                    x - half_patch: x + half_patch, y - half_patch: y + half_patch
                ][::-1]
                mask = np.rot90(mask)
                window = Window(
                    col_off=x - half_patch,
                    row_off=y - half_patch,
                    width=args.patch_size,
                    height=args.patch_size,
                )
                crop = src.read(1, window=window)

                assert crop.shape == (args.patch_size, args.patch_size), "invalid crop"

                # Get filename and save
                label = labels[idx]
                split = splits[idx]
                filename = f"{catalog_id}_{label}_{label_counters[label]}.tif"

                cv2.imwrite(f"{args.output_dir}/{split}/x/{filename}", crop)
                cv2.imwrite(f"{args.output_dir}/{split}/y/{filename}", mask)
                label_counters[label] += 1


if __name__ == "__main__":
    main()
