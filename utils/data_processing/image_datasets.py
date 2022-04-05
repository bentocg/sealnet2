__all__ = ["SealsDataset", "TestDataset"]

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import os
import geopandas as gpd
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

        # read dataframe with filenames and associated labels
        self.root = data_folder
        self.transforms = get_transforms(phase, patch_size, augmentation_mode)
        self.patch_size = patch_size
        self.split = phase

        # subset to training sets of interest
        self.ds = gpd.read_file(annotation_ds)
        self.ds = self.ds.loc[self.ds.split == self.split]

        # get labels and img names
        self.bin_labels = (
            self.ds["label"].isin(["crabeater", "weddell"]).astype(np.uint8)
        )

        self.img_names = [
            f"{self.root}/{self.ds.training_set.iloc[idx]}/{self.split}/x/{file}"
            for idx, file in enumerate(self.ds.img_name.values)
        ]

        self.ds = self.ds.loc[(self.ds.has_mask == 1) | (self.ds.pack_ice == 0)]
        self.mask_names = [
            False if "negative" in ele else ele.replace("/x/", "/y/")
            for ele in self.img_names
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
            mask = np.zeros([img.shape[0], img.shape[1], 1], dtype=np.uint8)

        if self.transforms is not None:
            try:
                augmented = self.transforms(image=img, mask=mask)

                img = augmented["image"]
                mask = augmented["mask"].reshape([1, self.patch_size, self.patch_size])

            except RuntimeError:
                idx -= 5
                img_path = self.img_names[idx]
                mask_path = self.mask_names[idx]
                label = self.bin_labels[idx]

                img = cv2.imread(img_path)
                if mask_path:
                    mask = cv2.imread(mask_path, 0)
                else:
                    mask = np.zeros([self.patch_size, self.patch_size], dtype=np.uint8)
                augmented = self.transforms(image=img, mask=mask)

                img = augmented["image"].reshape([1, self.patch_size, self.patch_size])
                mask = augmented["mask"].reshape([1, self.patch_size, self.patch_size])

        if mask.sum() == 0:
            label = 0

        count = round(mask.sum() / 255.0 / 5)
        return img, count, label, mask


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
