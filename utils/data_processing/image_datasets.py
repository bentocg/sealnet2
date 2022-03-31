__all__ = ["SeaIceDataset", "TestDataset"]

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import os
import cv2
from utils.data_processing import get_transforms


class SeaIceDataset(Dataset):
    def __init__(
        self,
        df,
        data_folder,
        tsets=("hand"),
        size=256,
        phase="training",
        segmentation=False,
        augmentation_mode="simple",
    ):

        # read dataframe with filenames and associated labels
        self.ds = df
        self.root = data_folder
        self.transforms = get_transforms(phase, size, augmentation_mode)
        self.segmentation = segmentation
        self.size = size
        self.split = phase
        if phase == "validation":
            self.split = f"{phase}{size}"

        # subset to training sets of interest
        self.ds = self.ds.loc[self.ds.split == self.split]
        if phase == "training":
            self.ds = self.ds.loc[self.ds.training_set.isin(tsets)]
            self.ds = self.ds.drop_duplicates(subset="img_name")

        # get labels and img names
        self.long_labels = self.ds.label.values
        self.bin_labels = self.ds["pack_ice"].values.astype(np.uint8)
        self.img_names = [
            f"{self.root}/{self.ds.training_set.iloc[idx]}/{self.split}/x/{file}"
            for idx, file in enumerate(self.ds.img_name.values)
        ]

        # get which labels come from the hand-annotated set
        self.is_hand = [int(ele == "hand") for ele in self.ds.training_set]

        if self.segmentation:
            self.ds = self.ds.loc[(self.ds.has_mask == 1) | (self.ds.pack_ice == 0)]
            self.mask_names = [ele.replace("/x/", "/y/") for ele in self.img_names]
            self.mask_names = [
                self.mask_names[idx] if ele else False
                for idx, ele in enumerate(self.ds.has_mask)
            ]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # read img and apply transforms
        img_path = self.img_names[idx]
        label = self.bin_labels[idx]
        is_hand = self.is_hand[idx]
        img = cv2.cvtColor(np.array(Image.open(img_path)), cv2.COLOR_BGR2GRAY)
        if self.segmentation:
            mask_path = self.mask_names[idx]
            if mask_path:
                mask = np.array(Image.open(mask_path), dtype=np.uint8)
            else:
                mask = np.zeros([img.shape[0], img.shape[1], 1], dtype=np.uint8)

        if self.transforms is not None:
            if self.segmentation:
                try:
                    augmented = self.transforms(image=img, mask=mask)

                    img = augmented["image"]
                    mask = augmented["mask"].reshape([1, self.size, self.size])

                except RuntimeError:
                    idx -= 5
                    img_path = self.img_names[idx]
                    mask_path = self.mask_names[idx]
                    label = self.bin_labels[idx]

                    img = cv2.cvtColor(np.array(Image.open(img_path)), cv2.COLOR_BGR2GRAY)
                    if mask_path:
                        mask = np.array(Image.open(mask_path), dtype=np.uint8)
                    else:
                        mask = np.zeros([self.size, self.size, 1], dtype=np.uint8)
                    augmented = self.transforms(image=img, mask=mask)

                    img = augmented["image"]
                    mask = augmented["mask"].reshape([1, self.size, self.size])
            else:
                try:
                    img = self.transforms(image=img)["image"]
                    label = self.bin_labels[idx]
                except:
                    print("failed")
                    print(self.img_names[idx])
                    idx -= 5
                    img_path = self.img_names[idx]
                    img = cv2.cvtColor(np.array(Image.open(img_path)), cv2.COLOR_BGR2GRAY)
                    img = self.transforms(image=img)["image"]
                    label = self.bin_labels[idx]
        is_hand = self.is_hand[idx]
        if self.segmentation:
            if mask.sum() == 0:
                label = 0
            return img, is_hand, label, mask

        else:
            return img, is_hand, label


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
        img = cv2.cvtColor(np.array(Image.open(img_name)), cv2.COLOR_BGR2GRAY)
        try:
            img = self.transforms(image=img)["image"]
        except:
            print("failed")
            print(img_name)
            idx -= 5
        return img, img_name
