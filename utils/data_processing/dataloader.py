__all__ = ["provider"]

from torch.utils.data import DataLoader, WeightedRandomSampler
from utils.data_processing import SealsDataset
import torch


def provider(
    annotation_ds,
    data_folder,
    phase,
    patch_size,
    batch_size=8,
    num_workers=4,
    augmentation_mode="simple",
    neg_to_pos_ratio=1,
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
    total_prob_neg = neg_to_pos_ratio / (neg_to_pos_ratio + 1)
    prob_neg = total_prob_neg / max(1, num_neg)
    prob_pos = (1 - total_prob_neg) / max(1, num_pos)
    weights = torch.Tensor(
        [prob_pos if ele else prob_neg for ele in image_dataset.bin_labels]
    )

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
    )
    if phase == "training":
        dataloader = DataLoader(
            image_dataset,
            num_workers=num_workers,
            sampler=sampler,
            batch_size=batch_size,
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
