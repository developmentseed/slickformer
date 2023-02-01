"""module for inference on icevision mask r-cnn models trained on CV2"""
import torch

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
            stacked_arr = torch.dstack(high_conf_classes)
            # TODO flag as to be addressed in stitching solution
            # this gets more complicated with 6 class model
            return torch.max(stacked_arr, axis=2)[0]  # we only want the value array
        else:
            return high_conf_class_mask.squeeze()
    else:
        return torch.zeros(size, size).long()
