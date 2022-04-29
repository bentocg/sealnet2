from typing import List, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon


def extract_centroids(pred_mask: np.ndarray) -> List[Tuple]:
    """

    :param pred_mask:
    :return:
    """

    pred_contours, _ = cv2.findContours(
        pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    pred_contours = [
        np.array([ele[0] for ele in coords])
        for coords in pred_contours
        if len(coords) > 2
    ]

    pred_centroids = []

    if pred_contours:

        # Convert into polygon centroids
        pred_centroids = [
            Polygon(coords).buffer(0).centroid for coords in pred_contours
        ]

        # Extract coordinates
        pred_centroids = [
            (pnt.coords.xy[0][0], pnt.coords.xy[1][0])
            for pnt in pred_centroids
            if pnt.coords  # Make sure points are valid
        ]

    return pred_centroids
