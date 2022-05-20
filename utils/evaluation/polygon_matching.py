from typing import List, Union, Optional, Tuple, Dict

from shapely.geometry import Polygon, Point
import cv2
import numpy as np


def match_polygons(true_mask: np.ndarray, pred_mask: np.ndarray) -> (int, int, int):
    """
    Matches predicted masks with ground truth masks by converting them to shapely polygons and
    looking for intersections.

    :param true_mask: array with true seal centroids
    :param pred_mask: array with binarized predicted seal locations
    :return:  true positives, false positives and false negatives, respectively
    """

    # Find elements in true and pred masks
    true_contours, _ = cv2.findContours(
        true_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    pred_contours, _ = cv2.findContours(
        pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Reformat contours
    true_contours = [
        np.array([ele[0] for ele in coords])
        for coords in true_contours
        if len(coords) > 2
    ]
    pred_contours = [
        np.array([ele[0] for ele in coords])
        for coords in pred_contours
        if len(coords) > 2
    ]

    # Convert into polygons
    true_polygons = [Polygon(coords).buffer(0) for coords in true_contours]
    pred_polygons = [Polygon(coords).buffer(0) for coords in pred_contours]

    # Match true polygons
    matched = set([])
    for idx_true, true_pol in enumerate(true_polygons):
        for idx_pred, pred_pol in enumerate(pred_polygons):
            if idx_pred in matched:
                continue
            if true_pol.intersects(pred_pol):
                matched.add(idx_pred)
                break

    tp = len(matched)
    fp = len(pred_polygons) - len(matched)
    fn = len(true_polygons) - len(matched)

    return tp, fp, fn


def match_points(
    true_points: List[Point],
    pred_points: List[Point],
    pred_counts: List[float],
    match_distance: float,
    cutoffs: Tuple[float],
) -> Tuple[Dict[float, int], Dict[float, int], Dict[float, int]]:
    """
    Matches GT and pred points to get instance level f1-score

    :param cutoffs: list with cutoffs for using
    :param true_points: list with GT points
    :param pred_points: list with predicted points
    :param pred_counts: list with predicted counts for each points' patch
    :param match_distance: distance at which two points are considered a match

    :return: true positives, false positives and false negatives, respectively
    """
    # Match true polygons
    matched = dict([])
    for idx_true, true_point in enumerate(true_points):
        for idx_pred, pred_point in enumerate(pred_points):
            if idx_pred in matched:
                continue
            if true_point.distance(pred_point) <= match_distance:
                matched[idx_pred] = pred_counts[idx_pred]
                break

    tp = {}
    fp = {}
    fn = {}
    for cutoff in cutoffs:
        pred_points_cutoff = [
            ele for idx, ele in enumerate(pred_points) if pred_counts[idx] > cutoff
        ]
        matched_cutoff = [key for key, val in matched.items() if val > cutoff]
        tp[cutoff] = len(matched_cutoff)
        fp[cutoff] = len(pred_points_cutoff) - len(matched_cutoff)
        fn[cutoff] = len(true_points) - len(matched_cutoff)

    return tp, fp, fn


def bbox_to_pol(xmin, ymin, xmax, ymax):
    return Polygon([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])


def match_bbox_pols(gt_pols, pred_pols, min_iou_match: float = 0.5):
    matched = set([])
    for idx_true, true_pol in enumerate(gt_pols):
        for idx_pred, pred_pol in enumerate(pred_pols):
            if idx_pred in matched:
                continue
            if true_pol.intersection(pred_pol).area / true_pol.union(pred_pol).area > min_iou_match:
                matched.add(idx_pred)
                break

    tp = len(matched)
    fp = len(pred_pols) - tp
    fn = len(gt_pols) - tp

    return tp, fp, fn
