"""module for inference on icevision mask r-cnn models trained on CV2"""
import torch
import numpy as np
from ceruleanml.utils import class_dict
from itertools import compress


def load_tracing_model(savepath):
    tracing_model = torch.jit.load(savepath)
    return tracing_model


def test_tracing_model_one_batch(dls, tracing_model):
    x, _ = dls.one_batch()
    out_batch_logits = tracing_model(x)
    return out_batch_logits


def logits_to_classes(out_batch_logits):
    """returns the confidence scores of the max confident classes
    and an array of max confident class ids.
    """
    probs = torch.nn.functional.softmax(out_batch_logits.squeeze(), dim=0)
    conf, classes = torch.max(probs, 0)
    return conf, classes


def apply_conf_threshold(conf, classes, conf_threshold):
    """Apply a confidence threshold to the output of logits_to_classes for a tile.
    Args:
        conf (np.ndarray): an array of shape [H, W] of max confidence scores for each pixel
        classes (np.ndarray): an array of shape [H, W] of class integers for the max confidence scores for each pixel
        conf_threshold (float): the threshold to use to determine whether a pixel is background or maximally confident category
    Returns:
        torch.Tensor: An array of shape [H,W] with the class ids that satisfy the confidence threshold. This can be vectorized.
    """
    high_conf_mask = torch.any(torch.where(conf > conf_threshold, 1, 0), axis=0)
    return torch.where(high_conf_mask, classes, 0)


def apply_conf_threshold_instances(pred_dict, bbox_conf_threshold):
    """Apply a confidence threshold to the output of logits_to_classes for a tile.
    Args:
        pred_dict (dict): a dict with (for example):

        {'boxes': tensor([[  0.00000,  14.11488, 206.41418, 210.23907],
          [ 66.99806, 119.41994, 107.67549, 224.00000],
          [ 47.37723,  41.04019, 122.53947, 224.00000]], grad_fn=<StackBackward0>),
        'labels': tensor([2, 2, 2]),
        'scores': tensor([0.99992, 0.99763, 0.22231], grad_fn=<IndexBackward0>),
        'masks': tensor([[[[0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    ...,
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.]]],


                [[[0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    ...,
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.]]],


                [[[0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    ...,
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.]]]], grad_fn=<UnsqueezeBackward0>)}
        bbox_conf_threshold (float): the threshold to use to determine whether a pixel is background or maximally confident category
    Returns:
        dict: The confidence thresholded dict result, using the bbox conf threshold. Value sof dict are now list sinstead of tensors: {'boxes':[], 'labels':[], 'scores':[], 'masks':[]}
    """
    new_dict = {"boxes": [], "labels": [], "scores": [], "masks": []}
    for i, score in enumerate(pred_dict["scores"]):
        if score > bbox_conf_threshold:
            new_dict["boxes"].append(pred_dict["boxes"][i])
            new_dict["labels"].append(pred_dict["labels"][i])
            new_dict["scores"].append(pred_dict["scores"][i])
            new_dict["masks"].append(pred_dict["masks"][i])
    return new_dict


def apply_conf_threshold_masks(pred_dict, mask_conf_threshold, size):
    """Apply a confidence threshold to the output of apply_conf_threshold_instances on the masks to get class masks.
    Args:
        pred_dict (dict): a dict with {'boxes':[], 'labels':[], 'scores':[], 'masks':[]}
        classes (np.ndarray): an array of shape [H, W] of class integers for the max confidence scores for each pixel
        conf_threshold (float): the threshold to use to determine whether a pixel is background or maximally confident category
    Returns:
        torch.Tensor: An array of shape [H,W] with the class ids that satisfy the confidence threshold. This can be vectorized.
    """
    high_conf_classes = []
    if len(pred_dict["masks"]) > 0:
        for i, mask in enumerate(pred_dict["masks"]):
            classes = torch.ones_like(mask) * pred_dict["labels"][i]
            classes = classes.long().squeeze()
            high_conf_class_mask = torch.where(mask > mask_conf_threshold, 1, 0)
            high_conf_class_mask = torch.where(high_conf_class_mask.bool(), classes, 0)
            high_conf_classes.append(high_conf_class_mask.squeeze())
        if len(high_conf_classes) > 1:
            # TODO flag as to be addressed in stitching solution
            # this gets more complicated with 6 class model
            # TODO changed this from stacking from past func def to plot masks individually
            # could have downstream impacts on other eval code?
            return high_conf_classes
        else:
            return [high_conf_class_mask.squeeze()]
    else:
        return [torch.zeros(size, size).long()]


def mrcnn_3_class_inference(
    list_chnnl_first_norm_tensors,
    scripted_model,
    bbox_conf_threshold,
    mask_conf_threshold,
    input_size,
    interclass_nms_threshold=None,
):
    scripted_model.eval()
    with torch.no_grad():
        losses, pred_list = scripted_model(list_chnnl_first_norm_tensors)
    pred_list[0]["masks"] = np.squeeze(
        pred_list[0]["masks"]
    )  # we modify this to match expected shape for plotting
    selected_classes_3_class_model = [
        "background",
        "infra_slick",
        "natural_seep",
        "coincident_vessel",
    ]
    selected_classes = [key for key in selected_classes_3_class_model if key in class_dict]
    pred_dict = apply_conf_threshold_instances(
        pred_list[0], bbox_conf_threshold=bbox_conf_threshold
    )
    if interclass_nms_threshold:
        pred_dict = apply_interclass_mask_nms(pred_dict, nms_threshold=interclass_nms_threshold)
    high_conf_class_arrs = apply_conf_threshold_masks(
        pred_dict, mask_conf_threshold=mask_conf_threshold, size=input_size
    )
    is_not_empty = [torch.any(mask) for mask in high_conf_class_arrs]
    high_conf_class_arrs = list(compress(high_conf_class_arrs, is_not_empty))
    pred_dict["labels"] = list(compress(pred_dict["labels"], is_not_empty))
    pred_dict["scores"] = list(compress(pred_dict["scores"], is_not_empty))
    pred_dict["boxes"] = list(compress(pred_dict["boxes"], is_not_empty))
    # necessary for torchmetrics
    pred_dict_thresholded = {}
    pred_dict_thresholded["masks"] = torch.stack(high_conf_class_arrs).to(dtype=torch.uint8)
    pred_dict_thresholded["scores"] = torch.stack(pred_dict["scores"])
    pred_dict_thresholded["labels"] = torch.stack(pred_dict["labels"])
    pred_dict_thresholded["boxes"] = torch.stack(pred_dict["boxes"])
    return pred_dict_thresholded, pred_dict


def mask_similarity(u, v):
    """
    Takes two pixel-confidence masks, and calculates how similar they are to each other
    Returns a value between 0 (no overlap) and 1 (identical)
    Utilizes an IoU style construction
    Can be used as NMS across classes for mutually-exclusive classifications
    """
    return 2 * torch.sum(torch.sqrt(torch.mul(u, v))) / (torch.sum(u + v))


def apply_interclass_mask_nms(pred_dict, nms_threshold):
    """Apply a nms threshold to the output of logits_to_classes for a tile.
    Args:
        pred_dict (dict): a dict with (for example):
        {'boxes': tensor([[  0.00000,  14.11488, 206.41418, 210.23907],
          [ 66.99806, 119.41994, 107.67549, 224.00000],
          [ 47.37723,  41.04019, 122.53947, 224.00000]], grad_fn=<StackBackward0>),
        'labels': tensor([2, 2, 2]),
        'scores': tensor([0.99992, 0.99763, 0.22231], grad_fn=<IndexBackward0>),
        'masks': tensor([[[[0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    ...,
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.]]],
                [[[0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    ...,
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.]]],
                [[[0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    ...,
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.]]]], grad_fn=<UnsqueezeBackward0>)}
        threshold (float): the threshold above which two predictions are considered overlapping
    Returns:
        dict: The confidence thresholded dict result, using the nms threshold. Values of dict are now lists instead of tensors: {'boxes':[], 'labels':[], 'scores':[], 'masks':[]}
    """
    masks = pred_dict["masks"]
    res = torch.tensor([True] * len(masks)).to(device="cuda")

    new_dict = {"boxes": [], "labels": [], "scores": [], "masks": []}
    for i, i_mask in enumerate(masks):
        if res[i]:
            downstream = [torch.tensor(True)] * (i + 1) + [
                mask_similarity(i_mask, j_mask) <= nms_threshold for j_mask in masks[i + 1 :]
            ]
            res = torch.logical_and(res, torch.stack(downstream).to(device="cuda"))

            new_dict["boxes"].append(pred_dict["boxes"][i])
            new_dict["labels"].append(pred_dict["labels"][i])
            new_dict["scores"].append(pred_dict["scores"][i])
            new_dict["masks"].append(pred_dict["masks"][i])
    return new_dict
