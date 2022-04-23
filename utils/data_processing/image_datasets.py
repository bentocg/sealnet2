__all__ = ["SealsDataset", "TestDataset"]

import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch
import os
import cv2
from utils.data_processing import get_transforms


class SealsDataset(Dataset):
    def __init__(
        self,
        patch_size: int,
        data_folder: str,
        annotation_ds: str,
        phase: str = "training",
        augmentation_mode: str = "simple",
    ):

        # Read dataframe with filenames and associated labels
        self.root = data_folder
        self.transforms = get_transforms(phase, patch_size, augmentation_mode)
        self.patch_size = patch_size
        self.split = phase

        # Subset to training sets of interest
        self.ds = pd.read_csv(annotation_ds)
        self.ds = self.ds.loc[self.ds.split == self.split]
        self.ds = self.ds.reset_index()

        # Get labels and img names
        self.bin_labels = (
            self.ds["label"].isin(["crabeater", "weddell"]).astype(np.uint8)
        )

        self.img_names = [
            f"{self.root}/{self.split}/x/{file}"
            for idx, file in enumerate(self.ds.img_name.values)
        ]

        self.mask_names = [
            False if "negative" in ele else ele.replace("/x/", "/y/")
            for ele in self.img_names
        ]

        self.base_count = [
             cv2.imread(filename).sum() // 5 if filename else 0 for filename in self.mask_names
        ]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # read img and apply transforms
        img_path = self.img_names[idx]
        label = self.bin_labels[idx]
        img = cv2.imread(img_path, 0)
        mask_path = self.mask_names[idx]
        if mask_path:
            mask = cv2.imread(mask_path, 0)
        else:
            mask = np.zeros([self.patch_size, self.patch_size, 1], dtype=np.uint8)

        if self.transforms is not None:
            if "negative" in img_path:
                augmented = self.transforms(image=img, mask=mask)

                img = augmented["image"]
                mask = augmented["mask"].reshape([1, self.patch_size, self.patch_size])
            else:
                while True:
                    augmented = self.transforms(image=img, mask=mask)

                    img_aug = augmented["image"]
                    mask_aug = augmented["mask"].reshape([1, self.patch_size, self.patch_size])
                    if mask_aug.sum() > 0:
                        img = img_aug
                        mask = mask_aug
                        del img_aug
                        del mask_aug
                        break

        if mask.sum() == 0:
            label = 0
        count = (mask.sum() / 255.0 / 5.0).round()
        return img, count, label, (mask / 255).float()


class TestDataset(Dataset):
    def __init__(self, data_folder):

        self.img_names = [f"{data_folder}/{ele}" for ele in os.listdir(data_folder)]
        self.transforms = get_transforms(phase="test")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # read img and apply transforms
        img_name = self.img_names[idx]
        img = cv2.imread(img_name, 0)
        try:
            img = self.transforms(image=img)["image"]
        except:
            print("failed")
            print(img_name)
            idx -= 5
        patch_size = img.shape[-1]
        return img.reshape([1, patch_size, patch_size]), img_name
