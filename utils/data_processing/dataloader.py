__all__ = ["provider"]

from torch.utils.data import DataLoader, WeightedRandomSampler
from utils.data_processing import SealsDataset
import numpy as np
import torch


def provider(
    annotation_ds: str,
    data_folder: str,
    phase: str,
    patch_size: int,
    batch_size: int = 8,
    num_workers: int = 4,
    augmentation_mode: str = "simple",
    uniform_group_weights: bool = False,
    neg_to_pos_ratio: float = 1.0,
):

    image_dataset = SealsDataset(
        annotation_ds=annotation_ds,
        data_folder=data_folder,
        phase=phase,
        patch_size=patch_size,
        augmentation_mode=augmentation_mode,
    )

    num_pos = sum(image_dataset.bin_labels)
    num_neg = len(image_dataset) - num_pos
    neg_to_pos_ratio += 1E-7  # Avoid division by zero
    class_weight = 1.0 / np.array([num_neg, num_pos * neg_to_pos_ratio])
    weights = torch.Tensor(
        [class_weight[ele] for ele in image_dataset.bin_labels]
    ).double()
    if uniform_group_weights:
        sum_pos_prev = class_weight[1] * num_pos
        sum_pos_now = 0

        # Weight down positive sample based on number of seals within patch and patch size relative
        # to original input image size (768 x 768)
        for idx, ele in enumerate(image_dataset.bin_labels):
            if ele:
                weights[idx] = weights[idx] / (
                    1
                    + (
                        (image_dataset.base_count[idx] - 1)
                        * (patch_size ** 2 / 768.0 ** 2)
                    )
                )
                sum_pos_now += weights[idx]

        # Re-weight positive samples based on the ratio of the original total weight and the new
        # total weight
        for idx, ele in enumerate(image_dataset.bin_labels):
            if ele:
                weights[idx] *= sum_pos_prev / sum_pos_now

    sampler = WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True
    )
    if phase == "training":
        dataloader = DataLoader(
            image_dataset,
            num_workers=num_workers,
            sampler=sampler,
            batch_size=batch_size,
            shuffle=False,
        )

    else:
        dataloader = DataLoader(
            image_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            shuffle=False,
        )

    return dataloader
