import torch
from tqdm import tqdm
from utils.evaluation.unet_instance_f1_score import unet_instance_f1_score
import torch.nn.functional as F
from utils.evaluation.dice_score import dice_coeff


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    f1_score = 0
    precision = 0
    recall = 0
    dice_score = 0

    # Iterate over the validation set
    for images, true_counts, _, true_masks in tqdm(
        dataloader,
        total=num_val_batches,
        desc="Validation round",
        unit="batch",
        leave=False,
    ):
        # Move images and labels to correct device and type
        images = images.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # Predict the mask and count
            pred_masks, pred_counts = net(images)

            # Calculate instance level f1 score after thresholding
            batch_f1, batch_precision, batch_recall = unet_instance_f1_score(
                true_masks=true_masks,
                true_counts=true_counts,
                pred_masks=pred_masks,
                pred_counts=pred_counts,
            )
            f1_score += batch_f1
            precision += batch_precision
            recall += batch_recall

            # Calculate dice coefficient
            pred_masks = (F.sigmoid(pred_masks) > 0.5).float()
            true_masks = true_masks.to(device=device, dtype=torch.float32)
            dice_score += dice_coeff(pred_masks, true_masks, reduce_batch_first=False)

    # Revert network to training
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return f1_score, precision, recall, dice_score
    return (
        f1_score / num_val_batches,
        precision / num_val_batches,
        recall / num_val_batches,
        dice_score / num_val_batches,
    )
