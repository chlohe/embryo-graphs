import os.path as osp
import glob
from tqdm import tqdm

import cv2

import torch
import numpy as np

from models import maskrcnn2d
from nms.train_gnn_nms import Net
from predict_frame import load_stack, predict, generate_graph_from_segmentations, spicy_nms

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_stack_paths(embryo_path, start_time=0, end_time=75):
    dirs = glob.glob(osp.join(embryo_path, '*'))
    # Filter out the timesteps we don't want
    dirs = [d for d in dirs if float(osp.basename(
        d)) > start_time and float(osp.basename(d)) < end_time]
    # Sort them
    dirs = sorted(dirs, key=lambda x: float(osp.basename(x)))
    return dirs


def serialise_prediction(prediction, timestamp=None):
    lines = [f'== {timestamp if timestamp is not None else -1} ==']
    # Extract data from model output
    masks = prediction['masks']
    boxes = prediction['boxes']
    depths = prediction['labels']
    confidences = prediction['scores']
    # Serialise cells
    for mask, box, depth, confidence in zip(masks, boxes, depths, confidences):
        # Generate a cell outline
        mask_points = np.column_stack(
            np.where(mask.detach().cpu().squeeze().numpy() > 0.5))
        mask_outline = cv2.convexHull(mask_points)
        # Extract metadata
        x_min, y_min, x_max, y_max = tuple(box.detach().cpu().numpy())
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        verts = [(m[0][1], m[0][0]) for m in mask_outline]
        # Append to lines
        lines.append(
            f'{int(cx)} {int(cy)} {depth} {confidence} {[v for v in verts]}')
    return '\n'.join(lines)


def generate_timelapse_reconstruction(stack_paths, model, nms):
    lines = []
    for stack_path in tqdm(stack_paths):
        # Load an image stack
        stack = load_stack(stack_path)
        # Run inference on the stack
        pred = predict(stack, model, nms)
        # Generate data for the file
        timestamp = osp.basename(stack_path)
        serialised_pred = serialise_prediction(pred, timestamp)
        lines.append(serialised_pred)
    return '\n'.join(lines)


if __name__ == '__main__':
    paths = ['/datasets/test/stacks/full_timelapse/CLINIC/EMBRYO']
    # Init model
    model = maskrcnn2d(
        12).cuda() if torch.cuda.is_available() else maskrcnn2d(12)
    model.load_state_dict(
        torch.load(
            'fold_4_model_2000_new_data.ckpt'
        )
    )
    model.eval()
    for embryo_path in paths:
        # Load stack paths
        stack_paths = load_stack_paths(embryo_path)
        filename = osp.basename(embryo_path)
        # Inference
        with open(f'{filename}-with-conf.txt', 'w+') as f:
            data = generate_timelapse_reconstruction(stack_paths, model, spicy_nms)
            f.write(data)
