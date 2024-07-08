import math
import sys
import torch
import wandb

import utils
import albumentations as A

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import ops

from models import maskrcnn2d
from datasets import T4SegmentationDataset2DDepthAsClass

# Necessary to load some images without crashing
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_one_epoch(model, optimizer, data_loader, device, epoch, log_interval=10, logging_enabled=True, batches=0):
    model.train()

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(
            optimizer, warmup_iters, warmup_factor)

    for images, targets in data_loader:
        batches += 1
        images = list(torch.from_numpy(image).to(
            device).permute(2, 0, 1).float() for image in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Logging
        if logging_enabled and batches % log_interval == 0:
            wandb.log({'loss': losses_reduced}, step=batches)
            wandb.log(loss_dict_reduced, step=batches)
            wandb.log({'lr': optimizer.param_groups[0]["lr"]}, step=batches)
    return batches

def eval_step(model, data_loader, device, batches):
    matched_ious = []
    total_cells = 0
    for image, target in data_loader:
        try:
            image = image.to(device).permute(0, 3, 1, 2).float()
            pred = model(image)
            pred = pred[0]
            pred_boxes = pred['boxes'].cpu().detach().squeeze().int()
            gt_boxes = target['boxes'].cpu().detach().squeeze()
            pred_depths = pred['labels'].cpu().detach().squeeze()
            gt_depths = target['labels'].cpu().detach().squeeze()
            total_cells += len(gt_boxes)
            # Add extra dimension to GT if there is only one cell in the image
            if len(gt_boxes.shape) == 1:
                gt_boxes = gt_boxes.unsqueeze(0)
            if len(gt_depths.shape) == 0:
                gt_depths = gt_depths.unsqueeze(0)
            # Calculate IOUs
            ious = ops.box_iou(gt_boxes, pred_boxes)
            # Calculate differences in depths between predictions
            depth_difference = torch.tensor(
                [
                    [
                        abs(d1.item() - d2.item()) for d2 in pred_depths
                    ] for d1 in gt_depths
                ]
            )
            # Mask out any IOUs where the depth distance > 1 FP
            depth_difference_mask = depth_difference <= 1 
            ious *= depth_difference_mask
            # Hunt down predicted bboxes with biggest IOUs with ground truth
            if len(ious) > 0:
                for iou_row in ious:
                    # Ignore if no bboxes exist
                    iou = torch.max(iou_row)
                    if iou > 0:
                        matched_ious.append(iou.item())
        except Exception as e:
            print(e)
    if logging_enabled:
        try:
            wandb.log({'Val Proportion Detected': len(matched_ious) / total_cells}, step=batches)
            wandb.log({'Val IoU': sum(matched_ious) / len(matched_ious)}, step=batches)
        except ZeroDivisionError:
            wandb.log({'Val Proportion Detected': 0}, step=batches)
            wandb.log({'Val IoU': 0}, step=batches)


if __name__ == '__main__':
    # Variables
    batch_size = 8
    learning_rate = 0.005
    momentum = 0.9
    weight_decay = 0.0005
    logging_enabled = True
    log_interval = 10
    for fold in range(3):
        # Set up
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        dataset = T4SegmentationDataset2DDepthAsClass(
            '/datasets/train/stacks/static',
            '/datasets/train/seg/static',
            use_augmentations=True,
            num_folds=5,
            fold=fold,
            augmentations=[
                A.RandomRotate90(p=0.3),
                A.Flip(p=0.3),
                A.GaussNoise(var_limit=0.002, p=0.3),
                A.Cutout(num_holes=30, max_h_size=20, max_w_size=20, p=0.3),
                A.RandomSizedBBoxSafeCrop(height=400, width=400, p=0.7)
            ]
            )
        dataset_val = T4SegmentationDataset2DDepthAsClass(
            '/datasets/val/stacks/static',
            '/datasets/val/seg/static',
            use_augmentations=True
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=utils.collate_fn)
        val_loader = DataLoader(
            dataset_val, batch_size=1, shuffle=True, num_workers=1)
        model = maskrcnn2d(12)
        # model.load_state_dict(torch.load(f'fold_{fold}_model_1000_multiscale_longer.ckpt'))
        model.to(device)

        # Init logging
        if logging_enabled:
            wandb_run = wandb.init(
                project='t4-segmentation-maskrcnn-full-audit', 
                entity='embrybros',
                reinit=True
            )
            wandb.config = {
                'learning_rate': learning_rate,
                'momentum': momentum,
                'weight_decay': weight_decay,
                'batch_size': batch_size
            }

        # Train
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=learning_rate,
                                    momentum=momentum, weight_decay=weight_decay)

        batches = 0
        for epoch in tqdm(range(2001)):
            # Do train stuff
            model.train()
            batches = train_one_epoch(model, optimizer, loader,
                                    device, epoch, log_interval=log_interval,
                                    logging_enabled=logging_enabled, batches=batches)

            # Do eval stuff
            model.eval()
            eval_step(model, val_loader, device, batches)
            # try:
            #     img = torch.tensor(dataset[0][0]).cuda()
            #     pred = model.forward([img.permute(2, 0, 1).float()])

            #     masks = pred[0]['masks'].detach().cpu()
            #     labels = pred[0]['labels'].detach().cpu()
            #     # Plot image
            #     plt.axis('off')
            #     plane = labels[0] if len(labels) > 0 else 4
            #     plt.imshow(img.cpu().detach().squeeze()[:, :, plane])
            #     if len(masks) > 0:
            #         plt.imshow(masks[0].squeeze().detach(), cmap='jet', alpha=0.4)
            #     plt.savefig(f'{epoch}.jpg')
            # except Exception as e:
            #     print(f'Oops - Exception {e}')
            if epoch % 400 == 0 or epoch == 1000:
                torch.save(model.state_dict(), f'fold_{fold}_model_{epoch}_new_data.ckpt')
        if logging_enabled:
            wandb_run.finish()
