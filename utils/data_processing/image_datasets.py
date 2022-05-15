__all__ = ["SealsDataset", "TestDataset"]

import gc
import json

import pandas as pd
from fiftyone.zoo.datasets.torch import torchvision
from torch.utils.data import Dataset
import numpy as np
import torch
import os
import cv2
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

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
            cv2.imread(filename).sum() // 5 if filename else 0
            for filename in self.mask_names
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
                    mask_aug = augmented["mask"].reshape(
                        [1, self.patch_size, self.patch_size]
                    )
                    if mask_aug.sum() > 0:
                        img = img_aug
                        mask = mask_aug
                        del img_aug
                        del mask_aug
                        gc.collect()
                        break

        if mask.sum() == 0:
            label = 0
        count = (mask.sum() / 255.0 / 5.0).round()
        return img, count, label, (mask / 255).float()


class TestDataset(Dataset):
    def __init__(self, data_folder):

        self.img_names = [f"{data_folder}/{ele}" for ele in os.listdir(data_folder)]
        self._transforms = get_transforms(phase="test")

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


class SealNetInstanceDataset(Dataset):
    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        return_masks=True,
        phase: str = "training",
        patch_size: int = 256,
        augmentation_mode: str = "simple",
    ):
        self.img_folder = img_folder
        self.annotations = json.load(open(ann_file))
        self.ids = sorted([int(ele) for ele in self.annotations["images"].keys()])
        self._transforms = get_transforms(
            mode=augmentation_mode, phase=phase, size=patch_size, bbox=True
        )
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        idx = self.ids[idx]
        img_path = (
            f"{self.img_folder}/{self.annotations['images'][str(idx)]['file_name']}"
        )
        img = cv2.imread(img_path, 0)
        targets = self.annotations["annotations"][str(idx)]
        target = {"image_id": idx, "annotations": targets}
        img, target = self.prepare(img, target)

        out = self._transforms(
            image=img,
            mask=target["masks"],
            bboxes=target["boxes"],
            category_ids=target["labels"],
        )

        img = out["image"]
        target = {
            "mask": out["mask"],
            "bboxes": out["bboxes"],
            "category_ids": out["category_ids"],
        }
        return img, target

    def __len__(self):
        return len(self.ids)


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=True):
        self.return_masks = return_masks

    def __call__(self, image: np.ndarray, target: dict):
        w, h = image.shape

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]

        classes = [obj["category_id"] for obj in anno]

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks.numpy().transpose([1, 2, 0])
        target["image_id"] = image_id

        # For conversion to coco api
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def convert_to_coco_api(ds):
    coco_ds = COCO()
    ann_id = 0
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"]
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds
