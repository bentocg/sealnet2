import numpy as np
import torch
import cv2

from utils.data_processing import inv_normalize


def overlay_boxes(image, predictions, color):
    """
    Adds the predicted boxes on top of the image
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions["labels"]
    boxes = predictions["boxes"]

    boxes = [ele for idx, ele in enumerate(boxes) if labels[idx] == 1]

    for box in boxes:
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), color, 3
        )

    return image


def create_vizualization_grid(images, outputs, targets, n=4):
    p = torch.tensor([1 / len(images)] * len(images))
    idcs = p.multinomial(min(n, len(images)))
    images = [ele for idx, ele in enumerate(images) if idx in idcs]
    outputs = [ele for idx, ele in enumerate(outputs) if idx in idcs]
    targets = [ele for idx, ele in enumerate(targets) if idx in idcs]
    images_denorm = inv_normalize(torch.vstack(images)).squeeze(0).detach().cpu().numpy()
    images_denorm = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in images_denorm]
    images_denorm = [overlay_boxes(ele, outputs[idx], color=(150, 0, 0)) for idx, ele in
                     enumerate(images_denorm)]
    images_denorm = [overlay_boxes(ele, targets[idx], color=(0, 150, 0)) for idx, ele in
                     enumerate(images_denorm)]
    patch_size = int(images_denorm[0].shape[0])
    grid_side = int(patch_size * (n ** 0.5))
    assert grid_side % 1 == 0, "Must be a number with an integer square root"
    grid = np.zeros((grid_side, grid_side, 3))
    for idx, ele in enumerate(images_denorm):
        x_start = (idx % 2) * patch_size
        y_start = (idx // 2) * patch_size
        grid[x_start: x_start + patch_size, y_start: y_start + patch_size, :] = ele
    return grid

