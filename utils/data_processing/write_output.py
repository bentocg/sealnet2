__all__ = ["write_output"]

import cv2
import numpy as np
import os


def write_output(out, img_names, out_dir):
    if type(out) != list:
        out = out.cpu().numpy().astype(np.uint8)
    for idx, img in enumerate(out):
        out_dir = f"{out_dir}/{os.path.basename(img_names[idx]).split('_')[0]}"
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(f"{out_dir}/{os.path.basename(img_names[idx])}", img)
