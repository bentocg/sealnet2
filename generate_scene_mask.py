import rasterio
import geopandas as gpd
from argparse import ArgumentParser


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
        help="Path to output dir to save centroids binary mask.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Read annotations
    annotations = gpd.read_file(args.input_shapefile)
    scenes = annotations.scene.unique()

    # Loop over scenes
    for scene in scenes:
        with rasterio.open(f"{args.scenes_folder}/{scene}") as scn:
            print(scn)


if __name__ == "__main__":
    main()
