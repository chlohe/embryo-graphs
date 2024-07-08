import os.path as osp
import glob
import cv2

from skimage import io, exposure
from skimage.transform import resize
from skimage.draw import disk

import torch
import numpy as np
import torchvision.ops as ops
import torchvision.transforms as T

from models import maskrcnn2d

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def normalise_stack(plane_imgs, plane_size=(400, 400)):
    """ Apply normalisation across the whole stack by combining planes
        into single image and applying normalisation to that. """
    # Combine the images
    combined_img = np.zeros(
        (plane_size[0] * len(plane_imgs), plane_size[1]))
    for i in range(len(plane_imgs)):
        combined_img[i * plane_size[0]:(i + 1) * plane_size[0], 0:plane_size[1]] = plane_imgs[i]
    # Normalise combined image
    combined_img = exposure.equalize_adapthist(
        combined_img, clip_limit=0.01)
    # Unpack the combined image into planes
    return [combined_img[i * plane_size[0]:(i + 1) * plane_size[0], 0:plane_size[1]]
            for i in range(len(plane_imgs))]


def load_stack(stack_path, plane_size=(400, 400)):
    # Load plane paths and sort them
    plane_paths = glob.glob(osp.join(stack_path, '*.jpg'))
    plane_paths = sorted(plane_paths,
                         key=lambda x: float(
                             osp.splitext(osp.basename(x))[0][1:]
                         ))
    # Load them into images
    plane_imgs = [io.imread(plane_path, as_gray=True)
                  for plane_path in plane_paths]
    plane_imgs = [img[50:-50, 50:-50] for img in plane_imgs]
    plane_imgs = [resize(img, plane_size) for img in plane_imgs]
    # Mask out the edges
    rr, cc = disk((plane_size[0]/2, plane_size[1]/2), plane_size[0]/2)
    circle_mask = np.zeros(plane_size)
    circle_mask[rr, cc] = 1
    plane_imgs = [img * circle_mask for img in plane_imgs]
    # Apply normalisation
    plane_imgs = normalise_stack(plane_imgs)
    # Bring it all together into a single multichannel image
    combined_img = np.zeros(
        (plane_size[0], plane_size[1], len(plane_imgs))
    )
    for i in range(len(plane_imgs)):
        combined_img[:, :, i] = plane_imgs[i]
    return combined_img


def predict(x, model, nms=None):
    """
    Predicts segmentation for a single frame

    Args:
        x (np.ndarray): Input image
        model (torch.nn.Module): Segmentation model
        nms ((np.ndarray, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> List[int], optional): 
            Function to call NMS model. Defaults to None.
    Returns:
        dict
    """

    image_tensor = torch.from_numpy(
        x).cuda().float().permute(2, 0, 1).unsqueeze(0)
    pred = model.forward(image_tensor)[0]
    # Extract data
    masks = pred['masks']
    boxes = pred['boxes'].int()
    depths = pred['labels']
    scores = pred['scores']
    # Do NMS if needed
    if nms is None:
        keep = [1 for _ in range(len(boxes))]
    else:
        try:
            keep = nms(image_tensor, masks, boxes, scores, depths)
            # keep = [1 for _ in range(len(boxes))]
        except Exception as e:
            keep = [1 for _ in range(len(boxes))]
        pred['masks'] = [mask for i, mask in enumerate(masks) if keep[i] == 1]
        pred['boxes'] = [box for i, box in enumerate(boxes) if keep[i] == 1]
        pred['labels'] = [depth for i,
                          depth in enumerate(depths) if keep[i] == 1]
        pred['scores'] = [score for i,
                          score in enumerate(scores) if keep[i] == 1]
    return pred


def serialise_prediction(prediction):
    lines = []
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


def find_overlapping_boxes(boxes, iou_threshold=0.7):
    # Calculate IOUs (in an upper triangular matrix)
    ious = torch.triu(ops.box_iou(boxes, boxes), diagonal=1)   
    # Keep IOUs greater than the threshold
    ious_above_thresh = (ious > iou_threshold).float() * ious
    # Compute pairs of indexs of boxes that exceed the IOU threshold
    overlapping_idxs = torch.nonzero(ious_above_thresh)
    # Split them into groups
    groups = {j: set() for j in range(len(boxes))}    
    for k, v in overlapping_idxs:
        groups[k.item()].add(v.item())
    # Convert groups from dict of sets into list of sets by "absorbing" the keys
    groups = [v.union({k}) for k, v in groups.items()]
    # Merge groups subsets into sets
    subsets_to_remove = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if groups[i].union(groups[j]) == groups[i]:
                subsets_to_remove.append(groups[j])
            elif groups[j].union(groups[i]) == groups[j]:
                subsets_to_remove.append(groups[i])
    for subset in subsets_to_remove:
        try:
            groups.remove(subset)
        except ValueError:
            continue
    return groups

def simple_nms(image_tensor, masks, boxes, scores, depths, conf_threshold=0.7, iou_threshold=0.9):
    """
    Applies Simple NMS
    """
    return [1 if x > conf_threshold else 0 for x in scores]
def spicy_nms(image_tensor, masks, boxes, scores, depths, conf_threshold=0.7, iou_threshold=0.9):
    overlapping_boxes = find_overlapping_boxes(boxes, iou_threshold=iou_threshold)
    # Remove low conf boxes
    boxes_under_thres = [i for i, x in enumerate(scores) if x <= conf_threshold]
    for i, group in enumerate(overlapping_boxes):
        overlapping_boxes[i].difference_update(set(boxes_under_thres))
    overlapping_boxes = [x for x in overlapping_boxes if len(x) > 0]
    # If boxes are literally on top of each other (with 1 FP distance), yeet the one with lowest conf
    for group in overlapping_boxes:
        if len(group) > 1:
            group_depth_and_conf = [(x, depths[x], scores[x]) for x in group]
            for i, current_depth, current_confidence in group_depth_and_conf:
                for j, depth, conf in [x for x in group_depth_and_conf if abs(current_depth - x[1]) <= 1]:
                    if conf < current_confidence:
                        try:
                            group.remove(j)
                        except KeyError:
                            # Probably already removed it
                            continue
    idxs_to_keep = [x for group in overlapping_boxes for x in group]
    return [1 if x in idxs_to_keep else 0 for x in range(len(scores))]
def global_nms(image_tensor, masks, boxes, scores, depths, conf_threshold=0.7, iou_threshold=0.9):
    overlapping_boxes = find_overlapping_boxes(boxes, iou_threshold=iou_threshold)
    # Remove low conf boxes
    boxes_under_thres = [i for i, x in enumerate(scores) if x <= conf_threshold]
    for i, group in enumerate(overlapping_boxes):
        overlapping_boxes[i].difference_update(set(boxes_under_thres))
    overlapping_boxes = [x for x in overlapping_boxes if len(x) > 0]
    # Pick the max from each group
    for group in overlapping_boxes:
        while (len(group) > 1):
            group.remove(min(group))
    idxs_to_keep = [x for group in overlapping_boxes for x in group]
    return [1 if x in idxs_to_keep else 0 for x in range(len(scores))]

if __name__ == '__main__':
    paths = glob.glob('exprs/MICCAI 2024/3D/*')
    # Init model
    model = maskrcnn2d(
        12).cuda() if torch.cuda.is_available() else maskrcnn2d(12)
    model.load_state_dict(
        torch.load(
            'fold_4_model_2000_new_data.ckpt'
        )
    )
    model.eval()
    # Do inference
    for stack_path in paths:
        x = load_stack(stack_path)
        pred = predict(x, model, spicy_nms)
        serialised_pred = serialise_prediction(pred)
        with open(osp.join('exprs/MICCAI 2024/Networks', osp.basename(stack_path) + '.txt'), 'w') as f:
            f.write(serialised_pred)
