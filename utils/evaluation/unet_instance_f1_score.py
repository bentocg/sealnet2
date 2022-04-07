from typing import List

import torch
import numpy as np
from multiprocessing import Pool


def getxy_max(input_array: np.ndarray, n: int, min_dist: int = 3):
    """[Helper function to get n highest points in 2D array keeping a minimum distance between hits]

    Arguments:
        input_array {np.array} -- [2D array, typically an image]
        n {int} -- [number of points]

    Keyword Arguments:
        min_dist {int} -- [minimun distance between hits] (default: {3})

    Returns:
        [list] -- [list of x, y tuples with indices for hotspots]
    """

    # check for invalid inputs and copy array
    assert len(input_array.shape) in [2, 3], "invalid input dimensions"
    if len(input_array.shape) == 3:
        array = input_array.copy()[0, :, :]
    else:
        array = input_array.copy()

    # get shape
    n_rows, n_cols = array.shape

    # get sorted indices
    sorted_idcs = np.argsort(-array, axis=None)

    # get n locations with the highest value, keeping a minimum distance between hits
    out_xy = []
    idx = 0
    while len(out_xy) < n and idx < len(sorted_idcs):
        # add maximum
        row = sorted_idcs[idx] // n_cols
        col = sorted_idcs[idx] % n_cols

        # skip value if it is too close to a previous point
        if array[row, col] == 0:
            idx += 1
            continue

        # add point to output list and mask out nearby points otherwise
        else:
            idx += 1
            out_xy.append((row, col))
            array[
                max(0, row - min_dist) : min(n_rows, row + min_dist),
                max(0, col - min_dist) : min(n_cols, col + min_dist),
            ] = 0

    # return indices
    return out_xy


def unet_instance_f1_score(
    true_masks: torch.float32,
    true_counts: torch.float32,
    pred_masks: torch.float32,
    pred_counts: torch.float32,
    matching_tolerance: int = 3,
    max_pred_count: int = 30,
):
    # Store results
    fn = 0
    fp = 0
    tp = 0

    # Calculate GT and predicted points
    ground_truth_xy = [
        getxy_max(mask, true_counts[idx]) for idx, mask in enumerate(true_masks.numpy())
    ]
    pred_xy = [
        getxy_max(mask, min(int(pred_counts[idx]), max_pred_count))
        for idx, mask in enumerate(pred_masks.numpy())
    ]

    # Match GT with predictions
    for idx, gt_points in enumerate(ground_truth_xy):
        n_matches = 0
        if len(gt_points) == 0:
            fp += len(pred_xy[idx])
        else:
            matched_gt = set([])
            matched_pred = set([])

            for gt_idx, pnt in enumerate(gt_points):
                pnt = np.array(pnt)

                for pred_idx, pnt2 in enumerate(pred_xy[idx]):
                    pnt2 = np.array(pnt2)
                    if gt_idx in matched_gt:
                        continue
                    if (
                        pred_idx not in matched_pred
                        and np.linalg.norm(pnt - pnt2) < matching_tolerance
                    ):
                        n_matches += 1
                        matched_pred.add(pred_idx)
                        matched_gt.add(gt_idx)

            tp += n_matches
            fp += len(pred_xy[idx]) - n_matches
            fn += len(gt_points) - n_matches

    # Return f1 score, precision and recall
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall / (precision + recall + eps))

    return f1, precision, recall
