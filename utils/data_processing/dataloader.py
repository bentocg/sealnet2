__all__ = ["provider"]

from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
from utils.data_processing import SeaIceDataset
import torch
import numpy as np


def provider(
    df_path,
    data_folder,
    phase,
    size,
    tsets,
    batch_size=8,
    num_workers=4,
    segmentation=False,
    augmentation_mode="simple",
    neg_to_pos_ratio=1,
):
    df = pd.read_csv(df_path)

    image_dataset = SeaIceDataset(
        df=df,
        data_folder=data_folder,
        phase=phase,
        tsets=tsets,
        size=size,
        segmentation=segmentation,
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
