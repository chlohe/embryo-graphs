import math
import torch
import torch.nn as nn
import torchvision.models
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads, paste_masks_in_image, fastrcnn_loss, maskrcnn_inference, maskrcnn_loss
from torchvision.models.detection.transform import resize_boxes, resize_keypoints

from .depthnet1 import DepthNet1, IdentityNMS

class CustomTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, images, targets):
        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images, size_divisible=32)
        image_sizes_list = []
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))
        return ImageList(images, image_sizes_list), targets

    def batch_images(self, images, size_divisible=32):

        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def max_by_axis(self, the_list):
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def postprocess(self,
                    result,
                    image_shapes,
                    original_image_sizes
                    ):
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks
            if "keypoints" in pred:
                keypoints = pred["keypoints"]
                keypoints = resize_keypoints(keypoints, im_s, o_im_s)
                result[i]["keypoints"] = keypoints
        return result


class CustomRoIHeads(RoIHeads):
    def __init__(self, original_roi_heads, depth_head):
        super().__init__(
            original_roi_heads.box_roi_pool,
            original_roi_heads.box_head,
            original_roi_heads.box_predictor,
            original_roi_heads.proposal_matcher.high_threshold,
            original_roi_heads.proposal_matcher.low_threshold,
            original_roi_heads.fg_bg_sampler.batch_size_per_image,
            original_roi_heads.fg_bg_sampler.positive_fraction,
            original_roi_heads.box_coder.weights,
            original_roi_heads.score_thresh,
            original_roi_heads.nms_thresh,
            original_roi_heads.detections_per_img,
            original_roi_heads.mask_roi_pool,
            original_roi_heads.mask_head,
            original_roi_heads.mask_predictor,
            original_roi_heads.keypoint_roi_pool,
            original_roi_heads.keypoint_head,
            original_roi_heads.keypoint_predictor
        )
        self.depth_head = depth_head

    def forward(self,
                features,      # type: Dict[str, Tensor]
                proposals,     # type: List[Tensor]
                image_shapes,  # type: List[Tuple[int, int]]
                targets=None   # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(
                proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result = []
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            # Depth regression
            boxes, scores, labels = self.depth_head(boxes, scores, labels)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                assert matched_idxs is not None
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(
                    features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert mask_logits is not None

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals,
                    gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {
                    "loss_mask": rcnn_loss_mask
                }
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)
        return result, losses


def maskrcnn2d(num_classes):
    # load an instance segmentation model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn()
    model.backbone.body.conv1 = nn.Conv2d(
        11, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)
    )
    model.transform = CustomTransform()

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    # create custom instance of RoIHeads
    model.roi_heads = CustomRoIHeads(model.roi_heads, IdentityNMS())
    return model
