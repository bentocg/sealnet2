__all__ = ["get_transforms", "inv_normalize"]

from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms


def train_transform(size: int, mode: str, bbox_params: Optional[A.BboxParams] = None):
    if mode == "simple":
        return A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RandomCrop(height=size, width=size),
                A.RandomBrightnessContrast(p=0.5),
                A.Flip(p=0.5),
                A.RandomRotate90(p=1),

                A.Normalize(mean=0.5, std=0.25),
                ToTensorV2(),
            ],
            bbox_params=bbox_params,
        )

    elif mode == "complex":
        return A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.25, scale_limit=0.25, rotate_limit=15, p=1.0
                ),
                A.RandomCrop(height=size, width=size),
                A.OneOf(
                    [
                        A.IAAAdditiveGaussianNoise(),
                        A.GaussNoise(),
                    ],
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.CLAHE(clip_limit=8),
                        A.IAASharpen(),
                        A.IAAEmboss(),
                        A.RandomContrast(),
                        A.RandomBrightness(),
                    ],
                    p=0.3,
                ),
                A.HueSaturationValue(
                    p=0.3, hue_shift_limit=0, val_shift_limit=15, sat_shift_limit=0
                ),
                A.Flip(p=0.66),
                A.Normalize(mean=0.5, std=0.25),
                ToTensorV2(),
            ],
            bbox_params=bbox_params,
        )


def val_transform(size: int, bbox_params: Optional[A.BboxParams] = None):
    return A.Compose(
        [
            A.CenterCrop(height=size, width=size),
            A.Normalize(mean=0.5, std=0.25),
            ToTensorV2(),
        ],
        bbox_params=bbox_params,
    )


def test_transform():
    return A.Compose(
        [
            A.Normalize(mean=0.5, std=0.25),
            ToTensorV2(),
        ]
    )


def get_transforms(phase: str, size: int = 256, mode: str ="simple", bbox: bool = False):
    bbox_params = None
    if bbox:
        bbox_params = A.BboxParams(format="coco", label_fields=['category_ids'], min_visibility=0.5)

    if phase == "training":
        return train_transform(size=size, mode=mode, bbox_params=bbox_params)
    elif phase == "test":
        return test_transform()
    else:
        return val_transform(size=size, bbox_params=bbox_params)


inv_normalize = transforms.Normalize(
    mean=[-0.5 / 0.25],
    std=[1 / 0.25],
)
