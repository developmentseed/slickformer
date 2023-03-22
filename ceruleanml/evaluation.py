import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm

from ceruleanml.inference import (
    apply_conf_threshold,
    apply_conf_threshold_instances,
    apply_conf_threshold_masks,
    logits_to_classes,
)

mpl.rcParams["axes.grid"] = False
mpl.rcParams["figure.figsize"] = (12, 12)

def match_instances(pred_masks, gt_masks, class_dict, iou_threshold=0.5):
    """
    Computes the matching prediction and groundtruth instances for each category based on IoU scores.

    Args:
        pred_masks (list of 2D numpy arrays): The list of predicted instance masks.
        gt_masks (list of 2D numpy arrays): The list of groundtruth instance masks.
        class_dict (dict): A dictionary mapping category names to category IDs.
        iou_threshold (float): The IoU threshold for matching instances (default: 0.5).

    Returns:
        A dictionary with the following keys:
        - 'true_positives': A dictionary with category IDs as keys and the number of true positives as values.
          Each value is a list of tuples (gt_category, pred_category) representing the matching groundtruth
          and predicted categories for each true positive.
        - 'false_positives': A dictionary with category IDs as keys and the number of false positives as values.
          Each value is a list of tuples (gt_category, pred_category) representing the groundtruth category and
          the predicted category for each false positive.
        - 'false_negatives': A dictionary with category IDs as keys and the number of false negatives as values.
          Each value is a list of tuples (gt_category, pred_category) representing the groundtruth category and
          the predicted category for each false negative.
    """
    # Get the list of class IDs from the class dictionary
    class_ids = list(class_dict.values())

    # Create dictionaries to store the true positives, false positives, and false negatives for each category
    true_positives = {class_id: [] for class_id in class_ids}
    false_positives = {class_id: [] for class_id in class_ids}
    false_negatives = {class_id: [] for class_id in class_ids}

    # Loop over the groundtruth masks and find the matching predictions
    for gt_idx, gt_mask in enumerate(gt_masks):
        gt_class_id = np.unique(gt_mask)[-1]
        if gt_class_id == 0:
            continue  # Skip the background class
        iou_max = -1
        pred_max_idx = -1
        for pred_idx, pred_mask in enumerate(pred_masks):
            pred_class_id = np.unique(pred_mask)[-1]
            if pred_class_id != gt_class_id:
                continue
            iou = compute_iou(gt_mask, pred_mask)
            if iou > iou_max:
                iou_max = iou
                pred_max_idx = pred_idx
        if iou_max >= iou_threshold:
            true_positives[gt_class_id].append((gt_class_id, gt_class_id))
            pred_masks.pop(pred_max_idx)  # Remove the matching prediction
        else:
            false_negatives[gt_class_id].append((gt_class_id, -1))

    # The remaining predictions are false positives
    for pred_mask in pred_masks:
        pred_class_id = np.unique(pred_mask)[-1]
        if pred_class_id == 0:
            continue  # Skip the background class
        false_positives[pred_class_id].append((-1, pred_class_id))

    # Convert the lists of tuples to counts
    true_positives = {k: len(v) for k, v in true_positives.items()}
    false_positives = {k: len(v) for k, v in false_positives.items()}
    false_negatives = {k: len(v) for k, v in false_negatives.items()}

    return {'true_positives': true_positives, 'false_positives': false_positives, 'false_negatives': false_negatives}

def compute_iou(mask1, mask2):
    """
    Computes the IoU score between two binary masks.

    Args:
        mask1 (2D numpy array): The first binary mask.
        mask2 (2D numpy array): The second binary mask.

    Returns:
        The IoU score between the two masks.
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def cm_f1(
    arrays_gt,
    arrays_pred,
    num_classes,
    save_dir,
    normalize=None,
    class_names=None,
    title="Confusion Matrix",
):
    """Takes paired arrays for ground truth and predicition masks, as well
        as the number of target classes and a directory to save the
        normalized confusion matrix plot to.
    Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    Args:
        arrays_gt (np.ndarray): The ground truth mask array from a label tile.
            Can have 1 channel shaped like (H, W, Channels).
        arrays_pred (np.ndarray): The prediction mask array from a validation tile
            having undergone inference.
            Can have 1 channel shaped like (H, W, Channels).
        num_classes (integer): The number of target classes.
        save_dir (string): The output directory to write the normalized confusion matrix plot to.
    Returns:
        F1 score (float): Evaluation metric. Harmonic mean of precision and recall.
        Normalized confusion matrix (table): The confusion matrix table.
    """
    # flatten our mask arrays and use scikit-learn to create a confusion matrix
    flat_preds = np.concatenate(arrays_pred).flatten()
    flat_truth = np.concatenate(arrays_gt).flatten()
    OUTPUT_CHANNELS = num_classes
    cm = confusion_matrix(
        flat_truth, flat_preds, labels=list(range(OUTPUT_CHANNELS)), normalize=normalize
    )

    classes = list(range(0, num_classes))

    # cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    if class_names is None:
        xticklabels = list(range(OUTPUT_CHANNELS))
        yticklabels = list(range(OUTPUT_CHANNELS))
    else:
        assert len(class_names) > 1
        xticklabels = class_names
        yticklabels = class_names
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f"  # 'd' # if normalize else 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
    ax.set_ylim(len(classes) - 0.5, -0.5)
    if normalize == "true":
        cm_name = os.path.join(f"{save_dir}", "cm_normed_true.png")
    elif normalize == "pred":
        cm_name = os.path.join(f"{save_dir}", "cm_normed_pred.png")
    elif normalize is None:
        cm_name = os.path.join(f"{save_dir}", "cm_count.png")
    else:
        raise ValueError(
            "normalize is not pred, true or None, check cm docs for sklearn."
        )

    plt.savefig(cm_name)
    print(f"Confusion matrix saved at {cm_name}")
    # compute f1 score
    f1 = f1_score(flat_truth, flat_preds, average="macro")
    print("f1_score", f1)

    return cm, f1



def get_cm_for_torchscript_model_mrcnn(
    coco_json_path,
    model,
    save_path,
    mask_conf_threshold,
    bbox_conf_threshold,
    num_classes,
    normalize=None,
    class_names=None,
    title="Confusion Matrix",
):
    """
    Calculates the confusion matrix from an icevision dataloader by converting mrcnn results to semantic masks. 
    Icevision's cm class needs to be used to create cm on instance basis but needs to be updated to account for
    false positives and negative samples, see PR on Skytruth repo (private currently).

    the torchscript model when it is loaded operates on batches, not individual images
    this doesn't support eval on negative samples if they are in the dls,
    since val masks don't exist with neg samples. need to be constructed with np.zeros

    returns cm and f1 score
    """
    with open(coco_json, 'r') as f:
        annotations = json.load(f)
    
    for im_record in annotations['images']:
        img_input = [torch.Tensor(np.moveaxis(record.img, 2, 0)) / 255]
        _, pred_list = model(img_input)
        masks_gt = []
        for anno in annotations['annotations']:
            if im_record['id'] == anno['image_id']:
                segment = anno['segmentation']
                # from the source, order is height then width https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/_mask.pyx#L288
                rle = maskUtils.frPyObjects(segment, anno['height'], anno['width'])
                masks_gt.append(maskUtils.decode(rle)*anno['category_id'])

        semantic_masks_gt = np.max(np.stack(masks_gt), axis=0)

        _, pred_list = model([torch.Tensor(np.moveaxis(record.img, 2, 0)) / 255])
        pred_dict = apply_conf_threshold_instances(
            pred_list[0], bbox_conf_threshold=bbox_conf_threshold
        )
        classes = apply_conf_threshold_masks(
            pred_dict,
            mask_conf_threshold=mask_conf_threshold,
            size=semantic_masks_gt.shape[0],
        )
        classes = classes.cpu().detach().numpy()
        val_arrs.append(semantic_masks_gt)
        class_preds.append(classes)
    return cm_f1(
        val_arrs, class_preds, num_classes, save_path, normalize, class_names, title
    )
