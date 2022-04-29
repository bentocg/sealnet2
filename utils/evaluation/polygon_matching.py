from typing import List

from shapely.geometry import Polygon, Point
import cv2
import numpy as np


def match_polygons(
    true_mask: np.ndarray, pred_mask: np.ndarray
) -> (float, float, float):
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
    true_points: List[Point], pred_points: List[Point], match_distance: float
) -> (float, float, float):
    """
    Matches GT and pred points to get instance level f1-score

    :param true_points: list with GT points
    :param pred_points: list with predicted points
    :param match_distance: distance at which two points are considered a match

    :return: true positives, false positives and false negatives, respectively
    """
    # Match true polygons
    matched = set([])
    for idx_true, true_point in enumerate(true_points):
        for idx_pred, pred_point in enumerate(pred_points):
            if idx_pred in matched:
                continue
            if true_point.distance(pred_point) <= match_distance:
                matched.add(idx_pred)
                break

    tp = len(matched)
    fp = len(pred_points) - len(matched)
    fn = len(true_points) - len(matched)

    return tp, fp, fn
