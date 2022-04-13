import os
import sys
from itertools import product

import cv2
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from argparse import ArgumentParser

sys.path.insert(0, "..")
from utils.data_processing import Tiff


def parse_args():
    parser = ArgumentParser("Test set creation script.")
    parser.add_argument(
        "--scenes-dir",
        "-sd",
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
        default="../test",
        help="Directory to save test set",
    )
    parser.add_argument(
        "--stride",
        "-s",
        type=float,
        default=0.5,
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
        os.makedirs(f"{args.output_dir}/{subdir}", exist_ok=True)

    # Process scenes
    tiff = Tiff()
    for scene in annotations.scene.unique():
        # Get annotations for scene
        scn_annotations = annotations.loc[annotations.scene == scene]

        # Read input scene
        try:
            img, width, height, transform, meta = tiff.process_raster(
                f"{args.scenes_dir}/{scene}"
            )
        except ValueError:
            print(f"Scene {scene} not present in directory {args.scenes_dir}.")
            continue

        # Save scene transform, width and height to test set
        transform_coords = [
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        ]
        if os.path.exists(f"{args.output_dir}/scene_stats.csv"):
            with open(f"{args.output_dir}/scene_stats.csv", "a") as file:
                file.write(
                    ",".join([str(ele) for ele in [scene, width, height, transform_coords]]) + "\n"
                )
        else:
            stats = pd.DataFrame(
                {"scene": scene, "height": height, "width": width, "transform": transform_coords}
            )
            stats.to_csv(f"{args.output_dir}/scene_stats.csv", index=False)

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
        cv2.imwrite(f"{args.output_dir}/y/{scene}", scene_mask)
        del scene_mask  # Free up memory

        # Tile input scene
        for left, down in product(
            range(0, height, int(args.patch_size * args.stride)),
            range(0, width, int(args.patch_size * args.stride)),
        ):
            right, top = left + args.patch_size, down + args.patch_size
            filename = (
                f"{args.output_dir}/x/{scene[:-4]}_{left}_{down}_{right}_{top}.tif"
            )
            crop = img[0, left : left + args.patch_size, down : down + args.patch_size]

            # Check if crop has non-missing data
            if crop.sum() == 0:
                continue

            # Save tile
            cv2.imwrite(filename, crop)

        # Loop through negative scenes
        neg_scenes = [
            ele for ele in os.listdir(args.test_neg_scenes_dir) if ele.endswith(".tif")
        ]
        for scene in neg_scenes:
            img, width, height, transform, meta = tiff.process_raster(
                f"{args.test_neg_scenes_dir}/{scene}"
            )

            for left, down in product(
                range(0, height, int(args.patch_size * args.stride)),
                range(0, width, int(args.patch_size * args.stride)),
            ):
                right, top = left + args.patch_size, down + args.patch_size
                filename = (
                    f"{args.output_dir}/x/{scene[:-4]}_{left}_{down}_{right}_{top}.tif"
                )
                crop = img[
                    0, left : left + args.patch_size, down : down + args.patch_size
                ]
                cv2.imwrite(filename, crop)


if __name__ == "__main__":
    main()
