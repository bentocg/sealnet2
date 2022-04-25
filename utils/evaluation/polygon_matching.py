from shapely.geometry import Polygon
import cv2
import numpy as np


def match_polygons(true_mask, pred_mask):

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
