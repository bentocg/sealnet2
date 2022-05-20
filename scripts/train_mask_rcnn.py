import logging
import os
import uuid
from argparse import ArgumentParser
from typing import Union
import torch
import torch.nn as nn
import sys

import wandb
from torch import optim
from torchvision.models.detection import MaskRCNN, FasterRCNN

from torch.utils.data import DataLoader


from tqdm import tqdm

sys.path.insert(0, "../")

from utils.training.utility import seed_all
from utils.evaluation.eval_maskrcnn import test_maskrcnn, validate_maskrcnn
from utils.models.maskrcnn_utils import reduce_dict
from utils.models.model_factory import get_instance_segmentation_model
from utils.data_processing.image_datasets import SealNetInstanceDataset


def parse_args():
    parser = ArgumentParser("Mask-RCNN training script")
    parser.add_argument(
        "--training-dir",
        "-tr",
        dest="training_dir",
        type=str,
        default="../training_set",
        help="Path to training set",
    )
    parser.add_argument(
        "--learning_rate", "-lr", type=float, help="learning rate for training"
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=5, help="Number of epochs for training"
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        default=4,
        help="Batch size for training, " "doubles for validation",
    )
    parser.add_argument(
        "--num_workers",
        "-w",
        type=int,
        default=4,
        help="Number of workers for data loader",
    )
    parser.add_argument("--amp", "-m", type=int, default=0, help="Use mixed precision")
    parser.add_argument(
        "--patch-size",
        "-ps",
        dest="patch_size",
        type=int,
        default=256,
        help="Patch size for input tiles",
    )
    parser.add_argument(
        "--data-parallel",
        "-dp",
        dest="data_parallel",
        default=False,
        help="Use data parallelism? (multi-gpu)",
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
        "--patience",
        "-p",
        type=int,
        default=3,
        help="Number of non-improving epochs until " "learning rate is decreased",
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
        "--test-gdf",
        "-tgt",
        dest="test_gdf",
        default="../shapefiles/seal-points-test-consensus.shp",
        help="Path to shapefile with test GT points",
    )
    parser.add_argument(
        "--min-val-f1-test",
        "-mf",
        dest="min_val_f1_test",
        type=float,
        default=0.5,
        help="Runs with best validation f1-score below this value will terminate early without "
        "a testing phase.",
    )
    parser.add_argument(
        "--model-architecture",
        "-ma",
        dest="model_architecture",
        type=str,
        default="maskrcnn_resnet50_fpn",
        choices=[
            "maskrcnn_resnet50_fpn",
            "fasterrcnn_resnet50_fpn",
        ],
        help="Model architecture",
    )
    parser.add_argument(
        "--box-iou-thresh",
        "-biou",
        dest="box_iou_thresh",
        type=float,
        default=0.5,
        help="Minimum IoU for matching pred box with GT during training"
    )

    return parser.parse_args()


def train_net(
    net: Union[MaskRCNN, FasterRCNN, nn.DataParallel],
    device: torch.device,
    experiment_id: str,
    training_dir: str = "training_set",
    epochs: int = 5,
    batch_size: int = 1,
    patch_size: int = 256,
    num_workers: int = 1,
    learning_rate: float = 1e-5,
    patience: int = 3,
    decay_factor: float = 0.5,
    val_rounds_per_epoch: int = 2,
    augmentation_mode: str = "simple",
    amp: bool = False,
) -> float:
    """

    :param net:
    :param device:
    :param experiment_id:
    :param training_dir:
    :param epochs:
    :param batch_size:
    :param patch_size:
    :param num_workers:
    :param learning_rate:
    :param patience:
    :param decay_factor:
    :param neg_to_pos_ratio:
    :param augmentation_mode:
    :param amp:
    :return:
    """
    # Create train loader
    train_loader = DataLoader(
        dataset=SealNetInstanceDataset(
            img_folder=f"{training_dir}/training/x",
            ann_file=f"{training_dir}/training/annotations_coco.json",
            patch_size=patch_size,
            augmentation_mode=augmentation_mode,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    # Create validation loader
    val_loader = DataLoader(
        dataset=SealNetInstanceDataset(
            img_folder=f"{training_dir}/validation/x",
            ann_file=f"{training_dir}/validation/annotations_coco.json",
            patch_size=patch_size,
            augmentation_mode=augmentation_mode,
        ),
        batch_size=batch_size * 2,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
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
            model_architecture=args.model_architecture,
            num_workers=num_workers,
            patience=patience,
            amp=amp,
        )
    )

    logging.info(
        f"""Starting training experiment {experiment_id}:
           Epochs:          {epochs}
           Batch size:      {batch_size}
           Learning rate:   {learning_rate}
           Device:          {device.type}
           Patience:        {patience}
           Decay factor:    {decay_factor}
           Model architecture: {args.model_architecture}
           Num workers: {num_workers}
           Augmentation mode: {augmentation_mode}
           Mixed Precision: {amp}
           
       """
    )

    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=patience, factor=decay_factor
    )  # Goal: maximize Dice score

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0
    non_improving = 0  # Number of validation epochs without improvement (quit after 15)
    best_f1 = 0

    net.train()

    for epoch in range(epochs):

        with tqdm(
            total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img"
        ) as pbar:
            for images, targets in train_loader:
                images = list(image.to(device) for image in images)
                targets = [
                    {
                        k: v.to(device)
                        for k, v in t.items()
                        if k in ["masks", "boxes", "labels"]
                    }
                    for t in targets
                ]

                with torch.cuda.amp.autocast(enabled=amp):
                    loss_dict = net(images, targets)

                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(losses).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                # Reduce losses over all GPUs for logging purposes
                loss_dict_reduced = reduce_dict(loss_dict)

                pbar.update(len(images))
                global_step += 1
                metrics_dict = {
                    key: val.item() for key, val in loss_dict_reduced.items()
                }
                metrics_dict["total_loss"] = sum([v for k, v in metrics_dict.items()])
                metrics_dict["epoch"] = epoch
                metrics_dict["global_step"] = global_step
                experiment.log(metrics_dict)

                # Evaluation round (n rounds per epoch)
                division_step = n_train // (batch_size * val_rounds_per_epoch)
                if division_step > 0:
                    if global_step % division_step == 0:
                        with torch.cuda.amp.autocast(enabled=amp):
                            (
                                val_f1,
                                val_precision,
                                val_recall,
                                output_grid,
                            ) = validate_maskrcnn(net, val_loader, device)
                            scheduler.step(val_f1)

                            experiment.log(
                                {
                                    "learning rate": optimizer.param_groups[0]["lr"],
                                    "validation instance f1": val_f1,
                                    "validation instance precision": val_precision,
                                    "validation instance recall": val_recall,
                                    "output": wandb.Image(output_grid),
                                    "step": global_step,
                                    "epoch": epoch,
                                }
                            )

                            if val_f1 > best_f1:
                                best_f1 = val_f1
                                non_improving = 0

                            else:
                                non_improving += 1
                                if non_improving > 3 * val_rounds_per_epoch:
                                    return best_f1
    return best_f1


if __name__ == "__main__":
    # Read arguments
    args = parse_args()

    # Start logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    experiment_id = str(uuid.uuid4())

    # Instantiate model
    net = get_instance_segmentation_model(
        num_classes=2, model_name=args.model_architecture, box_fg_iou_thresh=args.box_iou_thresh
    )
    net.to(device=device)

    # Use data-parallel when requested
    if args.data_parallel:
        device_ids = [int(ele) for ele in args.data_parallel.split("_")]
        net = nn.DataParallel(net, device_ids=device_ids)

    # Set random seed
    seed_all(0)

    # Start training loop
    best_f1 = 0
    try:
        best_f1 = train_net(
            net=net,
            epochs=args.epochs,
            training_dir=args.training_dir,
            augmentation_mode=args.augmentation_mode,
            patch_size=args.patch_size,
            num_workers=args.num_workers,
            experiment_id=experiment_id,
            batch_size=args.batch_size,
            patience=args.patience,
            learning_rate=args.lr,
            device=device,
            amp=bool(args.amp),
        )
    except KeyboardInterrupt:
        "Interrupted training, continuing to testing"

    if best_f1 < args.min_val_f1_test:
        logging.info("Best validation f-1 score too low, skipping testing")
        exit()

    test_maskrcnn(
        device=device,
        net=net,
        test_dir=f"{args.training_dir}/test",
        experiment_id=experiment_id,
        batch_size=args.batch_size * 2,
        num_workers=args.num_workers,
        amp=bool(args.amp),
        match_distance=1.5,
        nms_distance=1.0,
        ground_truth_gdf=args.test_gdf,
    )
    logging.info("Testing complete saving model checkpoint")

    # Save model checkpoint
    os.makedirs("../checkpoints", exist_ok=True)
    torch.save(net.state_dict(), f"../checkpoints/{experiment_id}.pth")
