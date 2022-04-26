__all__ = ["write_output"]

import cv2
import numpy as np
import os

import torch


def write_output(out: torch.Tensor, img_names: list, out_dir: str) -> None:
    """
    Writes tile predictions to disk.

    :param out: predictions for a batch
    :param img_names: list with input image names (includes raster coordinates)
    :param out_dir: directory to save temporary output
    """

    # Cast to CPU
    out = out.cpu().numpy().astype(np.uint8)

    # Loop through predictions
    for idx, img in enumerate(out):

        # Extract scene name from image name
        scn_dir = f"{'_'.join(os.path.basename(img_names[idx]).split('_')[:-4])}.tif"
        out_dir_curr = f"{out_dir}/{scn_dir}"
        os.makedirs(out_dir_curr, exist_ok=True)

        # Write results to scene specific folder to prepare for output mosaicing
        cv2.imwrite(
            f"{out_dir_curr}/{'_'.join(os.path.basename(img_names[idx]).split('_')[-4:])}",
            img,
        )
