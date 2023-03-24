import torch
from itertools import compress
from ceruleanml.data_pipeline import stack_to_tensors
def three_class_remap(class_id):
    """
    Remap 6 classes from the class_dict to more simple categories. So we go from this

    [{'supercategory': 'slick', 'id': 0, 'name': 'background'}, # TODO this is a quantitatively inconsequential bug that this is include din the coco metadata
    {'supercategory': 'slick', 'id': 1, 'name': 'infra_slick'},
    {'supercategory': 'slick', 'id': 2, 'name': 'natural_seep'},
    {'supercategory': 'slick', 'id': 3, 'name': 'coincident_vessel'},
    {'supercategory': 'slick', 'id': 4, 'name': 'recent_vessel'},
    {'supercategory': 'slick', 'id': 5, 'name': 'old_vessel'},
    {'supercategory': 'slick', 'id': 6, 'name': 'ambiguous'}]

    to this

    [{'supercategory': 'slick', 'id': 0, 'name': 'background'},
    {'supercategory': 'slick', 'id': 1, 'name': 'infra_slick'},
    {'supercategory': 'slick', 'id': 2, 'name': 'natural_seep'},
    {'supercategory': 'slick', 'id': 3, 'name': 'coincident_vessel'}]
    """
    assert class_id != 0 # annotations should not be background class
    if class_id == 1 or class_id ==2:
        return class_id
    elif class_id in [3,4,5]:
        return 3
    elif class_id == 6:
        return None # denotes that this should be removed. we only keep ambiguous for evaluation.
    else:
        raise ValueError("Class IDs for CV2 groundtruth should be between 1 and 6.")

def remap_gt_dict(gt_dict):
    """Removes the ambgiuous class"""
    remapped_labels = torch.Tensor([three_class_remap(l) for l in gt_dict['labels']])
    is_not_none = [torch.logical_not(torch.isnan(l)) for l in remapped_labels]
    gt_dict['labels'] = list(compress(gt_dict['labels'], is_not_none))
    gt_dict['masks'] = list(compress(gt_dict['masks'], is_not_none))
    gt_dict['boxes'] = list(compress(gt_dict['boxes'], is_not_none))
    return stack_to_tensors(gt_dict)