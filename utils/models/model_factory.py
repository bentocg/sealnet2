import sys


# Helper function to monkey-patch maskrcnn loss from BCE to DICE
def uncache(exclude):
    """Remove package modules from cache except excluded ones.
    On next import they will be reloaded.

    Args:
        exclude (iter<str>): Sequence of module paths.
    """
    pkgs = []
    for mod in exclude:
        pkg = mod.split(".", 1)[0]
        pkgs.append(pkg)

    to_uncache = []
    for mod in sys.modules:
        if mod in exclude:
            continue

        if mod in pkgs:
            to_uncache.append(mod)
            continue

        for pkg in pkgs:
            if mod.startswith(pkg + "."):
                to_uncache.append(mod)
                break

    for mod in to_uncache:
        del sys.modules[mod]


from typing import Union

import segmentation_models_pytorch as smp
import torchvision
import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.roi_heads import project_masks_on_boxes

from .transunet import TransUnet
from utils.loss_functions import DiceLoss

from torchvision.models.detection import roi_heads


# Replace the original loss with DiceLoss
def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    """
    Args:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """

    dice_loss = DiceLoss()

    discretization_size = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    mask_loss = dice_loss(
        mask_logits[torch.arange(labels.shape[0], device=labels.device), labels],
        mask_targets,
    )

    return mask_loss


roi_heads.maskrcnn_loss = maskrcnn_loss

uncache(["torchvision.models.detection.roi_heads"])

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from efficientnet_pytorch import EfficientNet


def round_filters(filters, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def get_semantic_segmentation_model(
    model_architecture: str,
    dropout_regression: float,
    patch_size: int,
    tta: bool = False,
) -> Union[smp.Unet, TransUnet]:
    """
    Model factory.

    :param model_architecture: string with model architecture
    :param dropout_regression: dropout for regression head [0, 1)
    :param patch_size: patch size
    :param tta: omit regression head for test-time-augmentation?
    :return: pytorch model object
    """
    # Define parameters for classification head
    aux_params = {"pooling": "avg", "classes": 1, "dropout": dropout_regression}
    if tta:
        aux_params = None

    if model_architecture == "UnetEfficientNet-b0":
        net = smp.Unet(
            encoder_name="efficientnet-b0", in_channels=1, aux_params=aux_params
        )
    elif model_architecture == "UnetEfficientNet-b1":
        net = smp.Unet(
            encoder_name="efficientnet-b1", in_channels=1, aux_params=aux_params
        )
    elif model_architecture == "UnetEfficientNet-b2":
        net = smp.Unet(
            encoder_name="efficientnet-b2", in_channels=1, aux_params=aux_params
        )
    elif model_architecture == "UnetEfficientNet-b3":
        net = smp.Unet(
            encoder_name="efficientnet-b3", in_channels=1, aux_params=aux_params
        )
    elif model_architecture == "TransUnet":
        if tta:
            raise Exception("Tta currently not supported for TransUnet")
        net = TransUnet(
            in_channels=1,
            classes=1,
            img_dim=patch_size,
            dropout_regression=dropout_regression,
        )
    else:
        net = smp.Unet(encoder_name="resnet34", in_channels=1, aux_params=aux_params)

    return net


def get_instance_segmentation_model(num_classes, model_name="maskrcnn_resnet50_fpn",
                                    box_fg_iou_thresh=0.5):
    # Load a pre-trained model for classification
    # and return only the features
    if model_name.startswith("efficientnet"):
        backbone = EfficientNet.from_pretrained(
            model_name=model_name, num_classes=num_classes, in_channels=1
        )
        # Number of output channels
        backbone.out_channels = int(round_filters(1280, backbone._global_params))
        model = MaskRCNN(
            backbone,
            num_classes,
            box_fg_iou_thresh=box_fg_iou_thresh,
            box_bg_iou_thresh=box_fg_iou_thresh - 0.2,
            box_nms_thresh=0.2,
            rpn_nms_thresh=0.2,
            rpn_score_thresh=0.8,
        )

    else:
        # Load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.__dict__[model_name](
            pretrained=True,
            box_fg_iou_thresh=box_fg_iou_thresh,
            box_bg_iou_thresh=box_fg_iou_thresh - 0.2,
            box_nms_thresh=0.2,
            rpn_nms_thresh=0.2,
            rpn_score_thresh=0.8,
        )

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if model_name.startswith("mask") or model_name.startswith("efficientnet"):
        # Get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256

        # Replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )

    # Replace anchor generator with one that has reasonable sizes
    anchor_generator = AnchorGenerator(
        sizes=((8, 11, 14, 17),), aspect_ratios=((0.8, 1.0, 1.2),)
    )

    model.rpn_anchor_generator = anchor_generator

    for param in model.parameters():
        param.requires_grad = True

    return model
