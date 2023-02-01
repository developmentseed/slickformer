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


def get_cm_for_torchscript_model_unet(
    dls,
    model,
    save_path,
    semantic_mask_conf_thresh,
    num_classes,
    normalize=None,
    class_names=None,
    title="Confusion Matrix",
):
    """
    the torchscript model when it is loaded operates on batches, not individual images
    this doesn't support eval on negative samples if they are in the dls,
    since val masks don't exist with neg samples. need to be constructed with np.zeros

    returns cm and f1 score
    """
    val_arrs = []
    class_preds = []
    for batch_tuple in tqdm(dls.valid):
        semantic_masks_batch = batch_tuple[1].cpu().detach().numpy()
        class_pred_batch = model(batch_tuple[0].cpu())
        probs, classes = logits_to_classes(class_pred_batch)
        t = apply_conf_threshold(probs, classes, semantic_mask_conf_thresh)
        class_pred_batch = t.cpu().detach().numpy()
        val_arrs.extend(semantic_masks_batch)
        class_preds.append(class_pred_batch)
    return cm_f1(
        val_arrs,
        class_preds,
        num_classes,
        save_path,
        normalize,
        class_names,
        title=title,
    )


def get_cm_for_torchscript_model_mrcnn(
    valid_ds,
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
    val_arrs = []
    class_preds = []
    for record in tqdm(valid_ds):
        masks_gt = []
        for i, label_id in enumerate(record.detection.label_ids):
            masks_gt.append(record.detection.mask_array.data[i] * label_id)
        # for 6 class model this needs to be mapping process TODO
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
