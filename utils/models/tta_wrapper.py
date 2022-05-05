import torch
import torch.nn as nn
from typing import Optional, Tuple

from ttach.base import Merger, Compose


class SegmentationRegTTAWrapper(nn.Module):
    """Wrap PyTorch nn.Module (segmentation model with regression head) with test time augmentation
    transforms
    Args:
        model (torch.nn.Module): segmentation model with single input and single output
            (.forward(x) should return either torch.Tensor or Mapping[str, torch.Tensor])
        transforms (ttach.Compose): composition of test time transforms
        merge_mode (str): method to merge augmented predictions mean/gmean/max/min/sum/tsharpen
        output_mask_key (str): if model output is `dict`, specify which key belong to `mask`
    """

    def __init__(
        self,
        model: nn.Module,
        transforms: Compose,
        merge_mode: str = "mean",
        output_mask_key: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.merge_mode = merge_mode
        self.output_key = output_mask_key

    def forward(
        self, image: torch.Tensor, *args
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        merger_mask = Merger(type=self.merge_mode, n=len(self.transforms))
        merger_count = Merger(type=self.merge_mode, n=len(self.transforms))

        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            augmented_output_mask, augmented_output_count = self.model(augmented_image, *args)
            augmented_output_mask = transformer.deaugment_mask(augmented_output_mask)
            merger_mask.append(augmented_output_mask)
            merger_count.append(augmented_output_count)

        return merger_mask.result, merger_count.result
