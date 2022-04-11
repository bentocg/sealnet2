import os
from itertools import product

import cv2
import numpy as np
import rasterio
import geopandas as gpd
from argparse import ArgumentParser

from utils.data_processing import Tiff


def parse_args():
    parser = ArgumentParser("Test set creation script.")
    parser.add_argument(
        "--scenes-dir",
        "-s",
        dest="scenes_dir",
        type=str,
        help="Path to directory with input rasters",
    )
    parser.add_argument(
        "--test-neg-scenes-dir",
        "-n",
        dest="test_neg_scenes_dir",
        type=str,
        help="Path to directory with negative scenes",
    )
    parser.add_argument(
        "--test-annotations-shp",
        "-a",
        dest="test_annotations_shp",
        type=str,
        help="Path to test annotations shapefile",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        type=str,
        default="../test_set",
        help="Directory to save test set",
    )
    parser.add_argument(
        "--stride",
        "-s",
        type=float,
        default=0.33,
        help="Stride to define scene overlap (factor of patch-size)",
    )
    parser.add_argument(
        "--patch-size",
        "-p",
        dest="patch_size",
        type=int,
        default=512,
        help="Patch-size for test tiles",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Read annotations
    annotations = gpd.read_file(args.test_annotations_shp)

    # Create output dir
    for subdir in ["x", "y"]:
        os.makedirs(f"{args.output_dir}/{subdir}")

    # Process scenes
    tiff = Tiff()
    for scene in annotations.scene.unique():
        # Get annotations for scene
        scn_annotations = annotations.loc[annotations.scene == scene]

        # Get scene identifier
        scene_id = f"{scene.split('_')[2]}-{scene.split('_')[5]}"

        # Read input scene
        try:
            img, width, height, transform, meta = tiff.process_raster(
                f"{args.scenes_dir}/{scene}"
            )
        except ValueError:
            print(f"Scene {scene} not present in directory {args.scenes_dir}.")
            continue

        # Create test mask for input scene
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

        # Save scene mask
        cv2.imwrite(f"{args.output_dir}/y/{scene_id}.tif", scene_mask)

        # Tile input scene
        for idx, row_col in enumerate(
            product(
                range(0, height, int(args.patch_size * args.stride)),
                range(0, width, int(args.patch_size * args.stride)),
            )
        ):
            row, col = row_col
            filename = f"{args.out_dir}/x/{scene_id}_{idx}.tif"
            crop = img[row : row + args.patch_size, col : col + args.patch_size]

            # Check if crop has non-missing data
            if crop.sum() == 0:
                continue

            # Save tile
            cv2.imwrite(filename, crop)


if __name__ == "__main__":
    main()
