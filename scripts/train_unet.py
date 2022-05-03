import argparse
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torchvision
import wandb
from torch import optim
from tqdm import tqdm

sys.path.insert(0, "../")

from utils.models.model_factory import model_factory
from utils.models.transunet import TransUnet
from utils.training.utility import seed_all
from utils.data_processing import provider, inv_normalize
from utils.loss_functions import SoftDiceLoss, FocalLoss, DiceLoss, MixedLoss
from utils.evaluation.eval_unet import validate_unet, test_unet

dir_checkpoint = Path("./checkpoints/")


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser.add_argument(
        "--training-dir",
        "-tr",
        dest="training_dir",
        type=str,
        default="training_set",
        help="Path to training set",
    )
    parser.add_argument(
        "--alpha-count",
        "-a",
        dest="alpha_count",
        type=float,
        default=0.5,
        help="Relative weight for count loss",
    )
    parser.add_argument(
        "--uniform-group-weights",
        "-u",
        dest="uniform_group_weights",
        type=int,
        default=False,
        help="Use weighted sampler to have uniform group sizes on positive samples?",
    )
    parser.add_argument(
        "--epochs", "-e", metavar="E", type=int, default=5, help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        metavar="B",
        type=int,
        default=1,
        help="Batch size for dataloader, multiplied by 2 for validation",
    )
    parser.add_argument(
        "--patch-size",
        "-ps",
        dest="patch_size",
        type=int,
        default=256,
        help="Patch size for input tiles",
    )
    parser.add_argument(
        "--learning-rate",
        "-l",
        metavar="LR",
        type=float,
        default=1e-5,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--load", "-f", type=str, default=False, help="Load model from a .pth file"
    )
    parser.add_argument(
        "--neg-to-pos-ratio",
        "-n",
        dest="neg_to_pos_ratio",
        type=float,
        default=1.0,
        help="Scale between number of negative and positive samples in train dataloader",
    )
    parser.add_argument(
        "--patience", "-p", type=int, default=3, help="Number of non-impro"
    )
    parser.add_argument(
        "--augmentation-mode",
        "-g",
        type=str,
        dest="augmentation_mode",
        default="simple",
        help="Augmentation mode",
    )
    parser.add_argument(
        "--amp", "-m", type=bool, default=False, help="Use mixed precision"
    )
    parser.add_argument(
        "--criterion-mask",
        "-c",
        type=str,
        default="Dice",
        dest="criterion_mask",
        help="Loss function for training U-Net masks, one of {'Dice', 'SoftDice', 'Focal', 'Mixed'}",
    )
    parser.add_argument(
        "--num-workers",
        "-w",
        dest="num_workers",
        type=int,
        default=1,
        help="Number of workers for dataloaders",
    )
    parser.add_argument(
        "--val-rounds-per-epoch",
        "-v",
        dest="val_rounds_per_epoch",
        type=int,
        default=3,
        help="Number of validation rounds per epoch",
    )
    parser.add_argument(
        "--data-parallel",
        "-dp",
        dest="data_parallel",
        default=False,
        help="Use data parallelism? (multi-gpu)",
    )
    parser.add_argument(
        "--test-gdf",
        "-tgt",
        dest="test_gdf",
        default="../shapefiles/seal-points-test-consensus.shp",
        help="Path to shapefile with test GT points",
    )
    parser.add_argument(
        "--model-architecture",
        "-ma",
        dest="model_architecture",
        type=str,
        default="UnetResnet34",
        help="Model architecture name",
    )
    parser.add_argument(
        "--dropout-regression",
        "-dr",
        dest="dropout_regression",
        type=float,
        default=0.0,
        help="Dropout for regression head",
    )
    parser.add_argument(
        "--tta",
        "-t",
        dest="tta",
        type=int,
        default=0.0,
        help="Use test-time-augmentation?",
    )
    return parser.parse_args()


def train_net(
    net: Union[nn.DataParallel, smp.Unet, TransUnet],
    device: torch.device,
    experiment_id: str,
    alpha_count: float = 0.5,
    epochs: int = 5,
    batch_size: int = 1,
    patch_size: int = 256,
    num_workers: int = 1,
    learning_rate: float = 1e-5,
    criterion_mask: nn.Module = SoftDiceLoss(),
    patience: int = 3,
    decay_factor: float = 0.5,
    criterion_count: nn.Module = nn.SmoothL1Loss(),
    neg_to_pos_ratio: float = 1.0,
    val_rounds_per_epoch: int = 3,
    augmentation_mode: str = "simple",
    uniform_group_weights: bool = False,
    save_checkpoint: bool = True,
    amp: bool = False,
) -> None:
    """
    Training loop for SealNet2.0, supports several options for hyperparameter tuning. Stores
    training statistics in wandb project.

    :param net: Unet from smp with a regression head
    :param device: device for running training loop
    :param experiment_id: experiment id for wandb
    :param alpha_count: relative weight for regression loss [0, 1]
    :param epochs: number of epochs to run training for
    :param batch_size: batch size for training dataloader (multiplied x2 for val and test)
    :param patch_size: patch size for training images
    :param num_workers: number of workers for train/val dataloaders
    :param learning_rate: learning rate
    :param criterion_mask: criterion for segmentation loss
    :param patience: number of rounds without improvement until reducing learning rate
    :param decay_factor: multiplier for reducing learning rate
    :param criterion_count: criterion for regression loss
    :param neg_to_pos_ratio: ratio of negative to positive images on training batches
    :param val_rounds_per_epoch: number of validation rounds within one epoch
    :param augmentation_mode: data augmentation mode E{simple, complex}
    :param uniform_group_weights: use uniform group weights on training batches?
    :param save_checkpoint: save model checkpoints?
    :param amp: use auto mixed-precision? (make sure your GPU supports amp)

    """

    # Create data loaders
    train_loader = provider(
        data_folder="../training_set",
        annotation_ds="../training_set/annotations_df.csv",
        num_workers=num_workers,
        augmentation_mode=augmentation_mode,
        uniform_group_weights=uniform_group_weights,
        batch_size=batch_size,
        neg_to_pos_ratio=neg_to_pos_ratio,
        phase="training",
        patch_size=patch_size,
    )
    val_loader = provider(
        data_folder="../training_set",
        annotation_ds="../training_set/annotations_df.csv",
        num_workers=num_workers,
        augmentation_mode=augmentation_mode,
        batch_size=batch_size * 2,
        phase="validation",
        patch_size=patch_size,
    )

    n_train = len(train_loader) * batch_size
    n_val = len(val_loader) * batch_size * 2

    # Initialize logging
    experiment = wandb.init(
        project="SealNet2.0",
        resume="allow",
        anonymous="allow",
        entity="bentocg",
        id=experiment_id,
    )
    experiment.config.update(
        dict(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            augmentation_mode=augmentation_mode,
            save_checkpoint=save_checkpoint,
            neg_to_pos_ratio=neg_to_pos_ratio,
            uniform_group_weights=uniform_group_weights,
            criterion_mask=args.criterion_mask,
            alpha_count=alpha_count,
            patience=patience,
            amp=amp,
            model_architecture=args.model_architecture,
            dropout_regression=args.dropout_regression,
            test_time_augmentation=args.tta,
        )
    )

    logging.info(
        f"""Starting training experiment {experiment_id}:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Patience:        {patience}
        Decay factor:    {decay_factor}
        Criterion count: {criterion_count}
        Criterion mask:  {criterion_mask}
        Count loss weight:     {alpha_count}
        Validation rounds per epoch: {val_rounds_per_epoch}
        Uniform group weights: {uniform_group_weights}
        Negative to positive ratio:  {neg_to_pos_ratio}
        Augmentation mode: {augmentation_mode}
        Mixed Precision: {amp}
        Test-time-augmentation: {args.tta}
    """
    )

    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=patience, factor=decay_factor
    )  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0
    non_improving = 0  # Number of validation epochs without improvement (quit after 15)
    best_f1 = 0

    # Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(
            total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img"
        ) as pbar:
            for images, true_counts, _, true_masks in train_loader:

                assert images.shape[1] == 1, (
                    f"Network has been defined with 1 input channel, "
                    f"but loaded images have {images.shape[1]} channels. Please check that "
                    "the images are loaded correctly."
                )

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                true_counts = true_counts.to(
                    device=device, dtype=torch.float32
                ).reshape(-1, 1)

                with torch.cuda.amp.autocast(enabled=amp):
                    pred_masks, pred_counts = net(images)
                    loss_mask = criterion_mask(pred_masks, true_masks)
                    loss_count = criterion_count(pred_counts, true_counts)
                    loss = (1 - alpha_count) * loss_mask + alpha_count * loss_count

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log(
                    {
                        "train loss (total)": loss.item(),
                        "train loss (count)": loss_count.item(),
                        "train loss (mask)": loss_mask.item(),
                        "step": global_step,
                        "epoch": epoch,
                    }
                )
                pbar.set_postfix(**{"total loss (batch)": loss.item()})
                pbar.set_postfix(**{"mask loss (batch)": loss_mask.item()})
                pbar.set_postfix(**{"count loss (batch)": loss_count.item()})

                # Evaluation round (n rounds per epoch)
                division_step = n_train // (batch_size * val_rounds_per_epoch)
                if division_step > 0:
                    if global_step % division_step == 0:
                        with torch.cuda.amp.autocast(enabled=amp):
                            (
                                f1_score,
                                precision,
                                recall,
                                dice_score,
                                count_mae,
                            ) = validate_unet(net, val_loader, device)
                            scheduler.step(f1_score)

                        logging.info("Validation F1 score: {}".format(f1_score))
                        grid_size = 6
                        p = torch.tensor([1 / len(images)] * len(images))
                        idcs = p.multinomial(min(grid_size, len(images)))
                        images = inv_normalize(images)[idcs].detach()
                        pred_masks = torch.sigmoid(pred_masks[idcs])
                        pred_masks = (pred_masks > 0.5).detach().float() * 255
                        true_masks = true_masks[idcs] * 255
                        images = torch.clamp(images, 0, 1) * 255
                        grid = torchvision.utils.make_grid(
                            torch.vstack(
                                [
                                    images,
                                    true_masks.repeat(1, 1, 1, 1),
                                    pred_masks.repeat(1, 1, 1, 1),
                                ]
                            ),
                            nrow=grid_size,
                            value_range=(0, 255),
                            scale_each=True,
                        )
                        grid = torch.unsqueeze(grid, 0)
                        experiment.log(
                            {
                                "learning rate": optimizer.param_groups[0]["lr"],
                                "validation instance f1": f1_score,
                                "validation instance precision": precision,
                                "validation instance recall": recall,
                                "validation count MAE": count_mae,
                                "validation pixel dice": dice_score,
                                "output": wandb.Image(grid),
                                "step": global_step,
                                "epoch": epoch,
                            }
                        )

                        # Check if f1-score improved, stop if it didn't for 15 validation rounds
                        if f1_score > best_f1:
                            best_f1 = f1_score
                            experiment.log({"best validation instance f1": best_f1})
                            non_improving = 0

                        else:
                            non_improving += 1
                            if non_improving > 3 * val_rounds_per_epoch:
                                return None

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), f"{dir_checkpoint}/{experiment_id}.pth")
            logging.info(
                f"Checkpoint for experiment {experiment_id} epoch {epoch + 1} saved!"
            )


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    experiment_id = str(uuid.uuid4())

    # Make sure criterion for masks is valid
    assert args.criterion_mask in [
        "Dice",
        "SoftDice",
        "Focal",
        "Mixed",
    ], f"Invalid Criterion choice: {args.criterion_mask}."
    if args.criterion_mask == "Dice":
        criterion_mask = DiceLoss()
    elif args.criterion_mask == "SoftDice":
        criterion_mask = SoftDiceLoss()
    elif args.criterion_mask == "Focal":
        criterion_mask = FocalLoss()
    else:
        criterion_mask = MixedLoss()

    # Make sure model architecture is supported
    assert args.model_architecture in [
        "UnetEfficientNet-b3",
        "UnetEfficientNet-b2",
        "UnetEfficientNet-b1",
        "UnetEfficientNet-b0",
        "UnetResnet34",
        "TransUnet",
    ], f"Invalid model architecture: {args.model_architecture}."

    net = model_factory(
        model_architecture=args.model_architecture,
        patch_size=args.patch_size,
        dropout_regression=args.dropout_regression,
    )
    net.to(device=device)

    if args.data_parallel:
        device_ids = [int(ele) for ele in args.data_parallel.split("_")]
        net = nn.DataParallel(net, device_ids=device_ids)

    # Set random seed
    seed_all(0)

    # Start training loop
    try:
        train_net(
            net=net,
            epochs=args.epochs,
            alpha_count=args.alpha_count,
            uniform_group_weights=args.uniform_group_weights,
            augmentation_mode=args.augmentation_mode,
            patch_size=args.patch_size,
            num_workers=args.num_workers,
            experiment_id=experiment_id,
            batch_size=args.batch_size,
            criterion_mask=criterion_mask,
            neg_to_pos_ratio=args.neg_to_pos_ratio,
            patience=args.patience,
            val_rounds_per_epoch=args.val_rounds_per_epoch,
            learning_rate=args.lr,
            device=device,
            amp=args.amp,
        )
    except KeyboardInterrupt:
        logging.info("Training interrupted, continuing to testing")

    # Start test loop
    logging.info("Started testing")
    if args.tta:
        cp = net.state_dict()
        net = model_factory(model_architecture=args.model_architecture, dropout_regression=0.0,
                            patch_size=args.patch_size, tta=True)
        net.to(device)
        net.load_state_dict(cp, strict=False)
        del cp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        test_unet(
            device=device,
            net=net,
            test_dir="../training_set/test",
            experiment_id=experiment_id,
            batch_size=args.batch_size * 2,
            num_workers=args.num_workers,
            amp=args.amp,
            threshold=0.5,
            match_distance=1.5,
            nms_distance=1.0,
            ground_truth_gdf=args.test_gdf,
            test_time_augmentation=args.tta,
        )
        logging.info("Testing complete saving model checkpoint")

        # Save model checkpoint
        os.makedirs("../checkpoints", exist_ok=True)
        torch.save(net.state_dict(), f"../checkpoints/{experiment_id}.pth")
    except KeyboardInterrupt:
        logging.info("Testing interruped")
        sys.exit(0)
