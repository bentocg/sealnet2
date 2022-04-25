__all__ = ["write_output"]

import cv2
import numpy as np
import os

import torch


def write_output(out: torch.Tensor, img_names, out_dir):

    out = out.cpu().numpy().astype(np.uint8)
    for idx, img in enumerate(out):
        scn_dir = f"{'_'.join(os.path.basename(img_names[idx]).split('_')[:-4])}.tif"
        out_dir_curr = f"{out_dir}/{scn_dir}"
        os.makedirs(out_dir_curr, exist_ok=True)
        cv2.imwrite(
            f"{out_dir_curr}/{'_'.join(os.path.basename(img_names[idx]).split('_')[-4:])}",
            img,
        )
