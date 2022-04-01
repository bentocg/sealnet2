import os

import cv2
import rasterio
import geopandas as gpd
import numpy as np
import sys
from argparse import ArgumentParser

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
        "-o",
        "--output_dir",
        type=str,
        default="scenes_centroid_masks",
        help="Path to output dir to save centroids binary mask.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

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
    os.makedirs(args.output_dir, exist_ok=True)

    # Loop over scenes
    tiff = Tiff()
    for scene in scenes:
        scn_annotations = annotations.loc[annotations.scene == scene]

        try:
            img, width, height, transform, meta = tiff.process_raster(
                f"{args.scenes_dir}/{scene}"
            )
        except ValueError:
            print(f"Scene {scene} not present in directory {args.scenes_dir}.")
            continue

        # Fill in scene mask with seal centroids
        mask = np.zeros((width, height), dtype=np.uint8)
        xs = [point.xy[0][0] for point in scn_annotations["geometry"]]
        ys = [point.xy[1][0] for point in scn_annotations["geometry"]]
        ys, xs = rasterio.transform.rowcol(
            transform=transform, xs=xs, ys=ys
        )  # Indexing is inverted for rasterio.
        if type(xs) == int:
            xs, ys = [xs], [ys]
        for point in zip(xs, ys):
            mask[point] = 255

        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))

        # Save output
        cv2.imwrite(f"{args.output_dir}/{scene}", mask)


if __name__ == "__main__":
    main()
