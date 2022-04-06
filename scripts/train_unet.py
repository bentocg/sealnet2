import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import wandb
from torch import optim
from tqdm import tqdm

sys.path.insert(0, "../")

from utils.data_processing import provider
from utils.loss_functions import MixedLoss
from scripts.eval_unet import evaluate

dir_checkpoint = Path("./checkpoints/")


def train_net(
    net,
    device,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    criterion_mask: nn.Module = MixedLoss(),
    criterion_count: nn.Module = nn.SmoothL1Loss(),
    neg_to_pos_ratio: float = 1.0,
    augmentation_mode: str = "simple",
    save_checkpoint: bool = True,
    amp: bool = False,
):

    # Create data loaders
    train_loader = provider(
        data_folder="../training_set",
        annotation_ds="../training_set/annotations_df.csv",
        num_workers=1,
        augmentation_mode=augmentation_mode,
        batch_size=batch_size,
        neg_to_pos_ratio=neg_to_pos_ratio,
        phase="training",
        patch_size=256,
    )
    val_loader = provider(
        data_folder="../training_set",
        annotation_ds="../training_set/annotations_df.csv",
        num_workers=1,
        augmentation_mode=augmentation_mode,
        batch_size=batch_size * 2,
        phase="validation",
        patch_size=256,
    )

    n_train = len(train_loader)
    n_val = len(val_loader)

    # Initialize logging
    experiment = wandb.init(project="U-Net", resume="allow", anonymous="must")
    experiment.config.update(
        dict(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            augmentation_mode=augmentation_mode,
            save_checkpoint=save_checkpoint,
            neg_to_pos_ratio=neg_to_pos_ratio,
            amp=amp,
        )
    )

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Negative to positive ratio:  {neg_to_pos_ratio}
        Augmentation mode: {augmentation_mode}
        Mixed Precision: {amp}
    """
    )

    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=3, factor=0.5
    )  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0

    # 5. Begin training
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
                true_counts = true_counts.to(device=device, dtype=torch.float32).reshape(-1, 1)

                with torch.cuda.amp.autocast(enabled=amp):
                    pred_masks, pred_counts = net(images)
                    loss = criterion_mask(pred_masks, true_masks) + criterion_count(
                        pred_counts, true_counts
                    )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log(
                    {"train loss": loss.item(), "step": global_step, "epoch": epoch}
                )
                pbar.set_postfix(**{"loss (batch)": loss.item()})

                # Evaluation round (2 rounds per epoch)
                division_step = n_train // (batch_size * 2)
                if division_step > 0:
                    if global_step % division_step == 0:

                        f1_score, precision, recall, dice_score = evaluate(net, val_loader, device)
                        scheduler.step(f1_score)

                        logging.info("Validation F1 score: {}".format(f1_score))
                        experiment.log(
                            {
                                "learning rate": optimizer.param_groups[0]["lr"],
                                "validation instance f1": f1_score,
                                "validation instance precision": precision,
                                "validation instance recall": recall,
                                "validation pixel dice": dice_score,
                                "images": wandb.Image(images[0].cpu()),
                                "masks": {
                                    "true": wandb.Image(true_masks[0].float().cpu()),
                                    "pred": wandb.Image(
                                        torch.softmax(pred_masks, dim=1)
                                        .argmax(dim=1)[0]
                                        .float()
                                        .cpu()
                                    ),
                                },
                                "step": global_step,
                                "epoch": epoch,
                            }
                        )

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(
                net.state_dict(),
                str(dir_checkpoint / "checkpoint_epoch{}.pth".format(epoch + 1)),
            )
            logging.info(f"Checkpoint {epoch + 1} saved!")


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
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
        help="Batch size",
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
        "--neg_to_pos_ratio",
        "-n",
        type=float,
        default=1.0,
        help="Scale between number of negative and positive samples in train dataloader",
    )
    parser.add_argument(
        "--augmentation_mode",
        "-g",
        type=str,
        default="simple",
        help="Augmentation mode",
    )
    parser.add_argument(
        "--amp", action="store_true", default=False, help="Use mixed precision"
    )
    parser.add_argument(
        "--bilinear", action="store_true", default=False, help="Use bilinear upsampling"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # Define parameters for classification head
    aux_params = {"pooling": "avg", "classes": 1}

    net = smp.Unet(encoder_name="efficientnet-b3", in_channels=1, aux_params=aux_params)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f"Model loaded from {args.load}")

    net.to(device=device)
    try:
        train_net(
            net=net,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            amp=args.amp,
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")
        sys.exit(0)
