__all__ = ["merge_output"]


import numpy as np
import cv2
import os


def merge_output(shape, tiles_dir):
    final_output = np.zeros(shape, dtype=np.uint8)
    denominator = np.zeros(shape, dtype=np.uint8)
    for ele in os.listdir(tiles_dir):
        left, down, right, top = ele.split("_")[-4:]
        top = top.split(".")[0]
        tile_out = cv2.imread(f"{tiles_dir}/{ele}", cv2.IMREAD_GRAYSCALE)
        final_output[int(left) : int(right), int(down) : int(top)] += tile_out
        denominator[int(left) : int(right), int(down) : int(top)] += 1
    denominator[denominator == 0] = 1
    final_output = final_output / denominator
    return final_output.astype(np.uint8)
