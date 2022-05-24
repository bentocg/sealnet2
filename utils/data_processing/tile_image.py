import os
from itertools import product
import cv2
import numpy as np


def tile_image(
    img: np.ndarray, patch_size: int, stride: float, out_dir: str, scene: str
) -> None:
    """
    Tiles a scene into patches.

    :param img: image from rasterio.open
    :param patch_size: height and width in pixels
    :param stride: distance between neighboring tiles as a factor of patch_size
    :param out_dir: directory for saving tiles
    :param scene: scene name
    """

    # Create output folder if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Extract raster dimensions
    _, height, width = img.shape

    # Loop through tiles
    for corner in product(
        list(range(0, height - patch_size, int(patch_size * stride)))
        + [height - patch_size],
        list(range(0, width - patch_size, int(patch_size * stride)))
        + [width - patch_size],
    ):
        left, down = corner
        if left > height - patch_size:
            left = height - patch_size
        if down > width - patch_size:
            down = width - patch_size
        right, top = left + patch_size, down + patch_size
        crop_img = img[0, left:right, down:top]

        # Check if crop has non-missing data
        if crop_img.sum() == 0:
            continue

        # Write to file
        filename = (
            f'{out_dir}/{scene.replace(".tif", "")}_{left}_{down}_{right}_{top}.tif'
        )
        cv2.imwrite(filename, crop_img)
