import gc
import os
import sys

import cv2
import numpy as np
import rasterio
import geopandas as gpd
from argparse import ArgumentParser

sys.path.insert(0, "..")
from utils.data_processing import Tiff, tile_image, store_scene_stats


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
        default="../training_set/test",
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

        store_scene_stats(
            scene=scene,
            width=width,
            height=height,
            transform=transform,
            out_path=f"{args.output_dir}/scene_stats.csv",
        )

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
        gc.collect()

        # Tile input scene
        tile_image(
            img=img,
            patch_size=args.patch_size,
            stride=args.stride,
            scene=scene,
            out_dir=f"{args.output_dir}/x",
        )

    # Loop through negative scenes
    neg_scenes = [
        ele for ele in os.listdir(args.test_neg_scenes_dir) if ele.endswith(".tif")
    ]
    for scene in neg_scenes:
        img, width, height, transform, meta = tiff.process_raster(
            f"{args.test_neg_scenes_dir}/{scene}"
        )

        # Store scene stats
        store_scene_stats(scene=scene, width=width, height=height, transform=transform,
                          out_path=f"{args.output_dir}/scene_stats.csv")

        # Tile input scene
        tile_image(
            img=img,
            patch_size=args.patch_size,
            stride=args.stride,
            scene=scene,
            out_dir=f"{args.output_dir}/x",
        )


if __name__ == "__main__":
    main()
