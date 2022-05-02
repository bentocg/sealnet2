from typing import Union

import segmentation_models_pytorch as smp
from transunet import TransUnet


def model_factory(
    model_architecture: str, dropout_regression: float, patch_size: int
) -> Union[smp.Unet, TransUnet]:
    """
    Model factory.

    :param model_architecture: string with model architecture
    :param dropout_regression: dropout for regression head [0, 1)
    :param patch_size: patch size
    :return: pytorch model object
    """
    # Define parameters for classification head
    aux_params = {"pooling": "avg", "classes": 1, "dropout": dropout_regression}

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
        net = TransUnet(
            in_channels=1,
            classes=1,
            img_dim=patch_size,
            dropout_regression=dropout_regression,
        )
    else:
        net = smp.Unet(encoder_name="resnet34", in_channels=1, aux_params=aux_params)

    return net
