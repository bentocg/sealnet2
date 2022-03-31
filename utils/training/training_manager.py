
from utils.loss_functions.mixed_loss import DicePerimeterLoss, LogCoshLoss
from utils.training import Meter, epoch_log
from utils.loss_functions import (
    MixedLoss,
    DiceLoss,
    FocalLoss,
    LogCoshLoss,
    DicePerimeterLoss,
)
from utils.data_processing import provider
from torch.utils.tensorboard import SummaryWriter

import torch
import torchvision
from torch import optim
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
import time
import copy
import os
import numpy as np
import cv2


class Trainer(object):
    """This class takes care of training and validation of our segmentation models"""

    def __init__(
        self,
        model,
        model_name,
        device="cuda:0",
        batch_size=(64, 128),
        patch_size=256,
        epochs=20,
        lr=1e-3,
        patience=3,
        tsets=("hand"),
        data_folder="training_set_synthetic",
        segmentation=True,
        state_dict=None,
        neg_to_pos_ratio=1.0,
        num_workers=4.0,
        loss="BCE",
        save_output=False,
        augmentation_mode="simple",
    ):
        self.num_workers = num_workers
        self.batch_size = {"training": batch_size[0], "validation": batch_size[1]}
        self.lr = lr
        self.num_epochs = epochs
        self.start_epoch = 0
        self.best_iou = 0.0
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.best_dice = 0.0
        self.phases = ["training", f"validation"]
        self.device = device
        self.segmentation = segmentation
        self.model_name = model_name
        self.save_output = save_output
        self.global_step = 0
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            torch.multiprocessing.set_start_method("spawn")
        self.net = model

        if loss == "BCE":
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif loss == "DICE":
            self.criterion = DiceLoss()
        elif loss == "Focal":
            self.criterion = FocalLoss()
        elif loss == "Mixed":
            self.criterion = MixedLoss()
        elif loss == "DicePerimeter":
            self.criterion = DicePerimeterLoss()
        elif loss == "LogCosh":
            self.criterion = LogCoshLoss()
        else:
            raise ValueError(f"Loss function {loss} is not currently supported")

        self.optimizer = optim.AdamW(self.net.parameters(), lr=self.lr)
        if state_dict:
            self.start_epoch = state_dict["epoch"] + 1
            self.global_step = state_dict["global_step"]
            self.best_iou = state_dict["best_iou"]
            self.best_dice = state_dict["best_dice"]
            self.net.load_state_dict(state_dict["state_dict"])
            self.optimizer.load_state_dict(state_dict["optimizer"])
            self.state = state_dict
        else:
            self.state = {
                "epoch": 0,
                "global_step": 0,
                "best_dice": 0,
                "best_iou": 0,
                "state_dict": copy.deepcopy(self.net.state_dict()),
                "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            }
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", patience=patience, verbose=True, factor=0.5
        )
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                segmentation=self.segmentation,
                tsets=tsets,
                data_folder=data_folder,
                df_path=f"{data_folder}/labels.csv",
                phase=phase,
                size=patch_size,
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
                neg_to_pos_ratio=neg_to_pos_ratio,
                augmentation_mode=augmentation_mode,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.writer = SummaryWriter(f"runs/{self.model_name}")
        self.inv_normalize = transforms.Normalize(
            mean=[-0.5 / 0.25],
            std=[1 / 0.25],
        )

    def forward(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        outputs = self.net(images)
        if self.segmentation:
            loss = self.criterion(outputs, targets)
        else:
            loss = self.criterion(outputs.view(-1, 1), targets.view(-1, 1))
        return loss, outputs

    @staticmethod
    def put_text(images, targets, preds):
        result = np.empty_like(images)
        for i in range(images.shape[0]):
            label = f"truth={targets[i].item()}"
            pred = f"pred={preds[i].item()}"
            patch_size = images.shape[-1]
            image = cv2.UMat(np.float32(images[i, :, :, :]).copy().transpose(1, 2, 0))
            image = cv2.putText(
                image,
                str(label),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                (0.5 / 256) * patch_size,
                color=(0.1, 1, 0.2),
                thickness=2,
            )
            image = cv2.putText(
                image,
                str(pred),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                (0.5 / 256) * patch_size,
                color=(0.1, 1, 0.2),
                thickness=2,
            )

            result[i, :, :, :] = image.get().transpose(2, 0, 1)

        return result

    def iterate(self, epoch, phase):
        meter = Meter(segmentation=self.segmentation)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "training")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_pos = 0.0
        total = 0.0
        total_batches = len(dataloader)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):
            images, _, targets = batch[0], batch[1], batch[-1]
            targets = targets.type_as(images)
            loss, outputs = self.forward(images, targets)
            if phase == "training":
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            if self.global_step % 10 == 0:
                self.writer.add_scalar(f"Loss/{phase}", loss.item(), self.global_step)
                self.writer.add_scalar(f"num_pos/{phase}", total_pos, self.global_step)
                self.writer.add_scalar(f"total/{phase}", total, self.global_step)
            running_loss += loss.item()
            total_pos += targets.sum().item()
            total += len(targets)
            meter.update(targets, outputs)
            self.global_step += 1

            if self.global_step % 100 == 0 and self.save_output:
                p = torch.tensor([1 / len(images)] * len(images))
                idcs = p.multinomial(min(6, len(images)))
                images = self.inv_normalize(images)[idcs].detach()
                outputs = torch.sigmoid(outputs[idcs])
                outputs = (outputs > 0.5).detach().float()
                targets = targets[idcs]
                images = torch.clamp(images, 0, 1)

                if self.segmentation:
                    targets *= 255
                    outputs = outputs.cpu()
                    grid = torchvision.utils.make_grid(
                        torch.vstack(
                            [
                                images,
                                targets.repeat(1, 1, 1, 1),
                                outputs.repeat(1, 1, 1, 1),
                            ]
                        ),
                        nrow=6,
                        value_range=(0, 255),
                        scale_each=True,
                    )
                    grid = torch.unsqueeze(grid, 0)
                    self.writer.add_images(f"input_images/{phase}", grid, epoch)

                else:
                    self.writer.add_images(
                        f"labelled_images/{phase}",
                        self.put_text(
                            images.detach().float(), targets.detach().float(), outputs
                        ),
                        epoch,
                    )

        epoch_loss = running_loss / (total_batches * batch_size)
        dice, iou = epoch_log(epoch_loss, meter)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        self.writer.add_scalar(f"Dice/{phase}", dice, epoch)
        self.writer.add_scalar(f"IoU/{phase}", iou, epoch)
        self.writer.add_scalar("learning rate", self.optimizer.param_groups[0]["lr"], epoch)

        torch.cuda.empty_cache()
        self.state["epoch"] = epoch
        self.state["global_step"] = self.global_step
        return dice, iou

    def start(self):
        since = 0
        os.makedirs("checkpoints", exist_ok=True)
        for epoch in range(self.start_epoch, self.num_epochs):
            since += 1
            self.net.train()
            self.iterate(epoch, "training")

            with torch.no_grad():
                self.net.eval()
                val_dice, val_iou = self.iterate(epoch, "validation")
            self.scheduler.step(val_dice)
            torch.save(self.state, f"checkpoints/{self.model_name}_last.pth")
            if val_dice > self.best_dice:
                since = 0
                # remove previous best checkpoint for this model
                prev = [
                    f"checkpoints/{ele}"
                    for ele in os.listdir("checkpoints")
                    if ele.startswith(f"{self.model_name}_dice")
                ]
                for ele in prev:
                    os.remove(ele)

                print("******** New optimal found, saving state ********")
                self.state["best_dice"] = self.best_dice = val_dice
                self.state["best_iou"] = self.best_iou = val_iou
                self.state["state_dict"] = copy.deepcopy(self.net.state_dict())
                self.state["optimizer"] = copy.deepcopy(self.optimizer.state_dict())
                torch.save(
                    self.state,
                    f"checkpoints/{self.model_name}_dice-{self.best_dice}"
                    f"_iou-{self.best_iou}_epoch-{epoch}.pth",
                )
            print()
            if since == 6:

                print(f"Did not improve for {since} epochs, early stopping triggered")
                print(f"Running best state on test set")
                quit()
        quit()
