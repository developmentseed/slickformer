from pycocotools import mask as maskUtils
import os
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
import skimage.io as skio
import albumentations as A
from itertools import compress
import numpy as np
from typing import Dict, List
import torch
from pathlib import Path
import json
from transformers import AutoImageProcessor
from itertools import compress


def is_int_like(var):
    int_types = (int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
    torch_int_types = (torch.int8, torch.int16, torch.int32, torch.int64)

    if isinstance(var, int_types):
        return True
    elif torch.is_tensor(var) and var.dtype in torch_int_types:
        return True
    raise ValueError(f"{var} of type {type(var)} is not an int and can't be used as a class label.")


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
    assert is_int_like(class_id)
    assert class_id != 0 # annotations should not be background class
    if class_id == 1 or class_id ==2:
        return class_id
    elif class_id in [3,4,5]:
        return 3
    elif class_id == 6:
        return np.nan # denotes that this should be removed. we only keep ambiguous for evaluation.
    else:
        raise ValueError("Class IDs for CV2 groundtruth should be between 1 and 6.")

def stack_boxes(gt_sample):
    gt_sample['boxes'] = torch.stack([torch.tensor(arr) for arr in gt_sample['boxes']])
    return gt_sample

def get_src_pths_annotations(data_pth):
    data_dir = Path(data_pth)
    l = data_dir/"tiled_images"
    imgs = list(l.glob("*"))
    with open(data_dir/"instances_CeruleanCOCO.json", 'r') as f:
        annotations = json.load(f)
    return imgs, annotations

def remap_class_dict(class_dict: Dict[str, Dict[str, object]], new_class_list: List[str]) -> Dict[str, Dict[str, object]]:
    """
    Remap a dictionary of class names and their attributes to a new dictionary with
    a different set of class names and color codes.

    Parameters:
        class_dict (dict): A dictionary of class names and their attributes.
            Each key is a string representing the class name, and each value is
            a dictionary with keys "hr" (human readable text) and "cc" (color code).
        new_class_list (list): A list of new class names to use in the remapped dictionary,
            using the original keys of the `data_creation.class_dict`.

    Returns:
        A dictionary with the same key-value pairs as the original `class_dict`,
        but with the new class names and color codes.

    Raises:
        ValueError: If a new class name is not found in the original `class_dict`.
    """
    if new_class_list == list(class_dict.keys()):
        return class_dict
    color_list = [class_dict[name]["cc"] for name in class_dict.keys()]
    if len(color_list) < len(new_class_list):
        raise ValueError("Not enough colors in original class dict.")
    new_class_dict = {}
    for new_class_name, color in zip(new_class_list, color_list):
        if new_class_name in class_dict:
            old_class_dict = class_dict[new_class_name]
            new_class_dict[new_class_name] = {"hr": old_class_dict["hr"], "cc": color}
        else:
            raise ValueError(f"New class name '{new_class_name}' not found in original class dict.")
    return new_class_dict

def stack_tensors(gdict):
    gdict['masks'] = torch.stack([torch.tensor(arr) for arr in gdict['masks']]).to(dtype=torch.uint8)
    gdict['labels'] = torch.stack([torch.tensor(arr) for arr in gdict['labels']])
    gdict['boxes'] = torch.stack([torch.tensor(arr) for arr in gdict['boxes']])
    return gdict

def put_image_in_dict(image):
    pdict = {}
    pdict['image'] = image
    return pdict


@functional_datapipe("get_scene_paths")
class GetScenePaths(IterDataPipe):
    def __init__(self, source_dp, annotations_dir, **kwargs):
        self.source_dp = source_dp
        self.annotations_dir = annotations_dir
        self.kwargs = kwargs
        self.annotations = next(iter(self.source_dp))

    def __iter__(self):
        for record in self.annotations['images']:
            yield os.path.join(self.annotations_dir, "tiled_images", record['file_name'])
    def __len__(self):
        return len(self.annotations['images'])

@functional_datapipe("read_tiff")
class ReadTiff(IterDataPipe):
    def __init__(self, source_dp, **kwargs):
        self.src_img_paths = source_dp
        self.kwargs = kwargs
    def __iter__(self):
        for src_img_path in self.src_img_paths:
            yield skio.imread(src_img_path)
    def __len__(self):
        return len(self.src_img_paths)

@functional_datapipe("decode_masks")
class DecodeMasks(IterDataPipe):
    """decodes mask based on scene and annotation metadata.

    Args:
        scene_id (dict): A scene_id in annotations['images']. assumes working from full S1 imagery.
        annotations (dict): the COCO JSON dict

    Returns:
        tuple: Tuple of list of mask arrays and annotation metadata, including RLE compressed masks
    """
    def __init__(self, label_dp, **kwargs):
        self.label_dp = label_dp
        self.kwargs = kwargs

    def __iter__(self):
        for annotations in self.label_dp:
            for im_record in annotations['images']:
                mask_arrs = []
                labels = []
                annos = {}
                for anno in annotations['annotations']:
                    if im_record['id'] == anno['image_id']:
                        segment = anno['segmentation']
                        # from the source, order is height then width https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/_mask.pyx#L288
                        rle = maskUtils.frPyObjects(segment, anno['height'], anno['width'])
                        mask_arrs.append(maskUtils.decode(rle)*anno['category_id'])
                        labels.append(anno['category_id'])
                        annos.update({'masks':mask_arrs, 'labels':labels, "image_name": anno['big_image_original_fname']})
                yield annos

@functional_datapipe("random_crop_mask_if_exists")
class RandomCropByMasks(IterDataPipe):
    #https://albumentations.ai/docs/getting_started/mask_augmentation/
    def __init__(self, img_masks_tuple, crop_w, crop_h, **kwargs):
        self.tups =  img_masks_tuple
        self.kwargs = kwargs
        self.crop_w = crop_w
        self.crop_h = crop_h
    def __iter__(self):
        transform = A.Compose(
            [A.CropNonEmptyMaskIfExists(width=self.crop_w, height=self.crop_h)],
        )
        for src_img, mask_dict in self.tups:
            t = transform(image = src_img, masks=mask_dict['masks'], category_ids=mask_dict['labels'])
            is_not_empty = [np.any(mask) for mask in t['masks']]
            t['masks'] = list(compress(t['masks'], is_not_empty))
            # albumentations uses category ids, coco and mrcnn model use labels.
            t['labels'] = list(compress(t['category_ids'], is_not_empty))
            t.pop('category_ids')
            t['boxes'] = [extract_bounding_box(mask) for mask in t['masks'] ]
            t['image_name'] = mask_dict['image_name']
            yield t

@functional_datapipe("channel_first_norm_to_tensor")
class ChannelFirstNormToTensor(IterDataPipe):
    def __init__(self, source_dp):
        self.source_dp = source_dp

    def __iter__(self):
        for gdict in self.source_dp:
            gdict.update({"image": torch.Tensor(np.moveaxis(gdict['image'], 2, 0) / 255)})
            yield gdict

    def __len__(self):
        return len(self.source_dp)

@functional_datapipe("combine_src_label_dicts")
class CombineDicts(IterDataPipe):
    #https://albumentations.ai/docs/getting_started/mask_augmentation/
    def __init__(self, img_masks_tuple, **kwargs):
        self.tups =  img_masks_tuple
        self.kwargs = kwargs
    def __iter__(self):
        for src_img, mask_dict in self.tups:
            mask_dict['boxes'] = [extract_bounding_box(mask) for mask in mask_dict['masks'] ]
            mask_dict['image'] = src_img
            yield mask_dict


@functional_datapipe("remap_remove")
class RemapAndRemoveAmbiguous(IterDataPipe):
    #https://albumentations.ai/docs/getting_started/mask_augmentation/
    def __init__(self, sample_dicts, **kwargs):
        self.sample_dicts =  sample_dicts
        self.kwargs = kwargs
    def __iter__(self):
        """Removes samples with the ambgiuous class and empty samples"""
        for gt_dict in self.sample_dicts:
            if gt_dict['image_name'] == 'S1A_IW_GRDH_1SDV_20210827T001554_20210827T001619_039408_04A7B9_21A5.tif':
                print("this has empty masks but shouldn't at some point before masks_to...")
            # TODO we are filtering out negative samples here, but good to experiment to include
            remapped_labels = np.array([three_class_remap(l) for l in gt_dict['labels']])
            is_not_none = [np.logical_not(np.isnan(l)) for l in remapped_labels]
            if np.any(is_not_none):
                gt_dict['labels'] = list(compress(remapped_labels, is_not_none))
                gt_dict['labels'] = np.array(gt_dict['labels'], dtype=np.int8)
                gt_dict['masks'] = list(compress(gt_dict['masks'], is_not_none))
                gt_dict['boxes'] = list(compress(gt_dict['boxes'], is_not_none))
                gt_dict['masks'] = [(mask > 0) * gt_dict['labels'][i] for i, mask in enumerate(gt_dict['masks'])]
                yield gt_dict

@functional_datapipe("stack_to_tensor")
class StackConvertLabelsToTensor(IterDataPipe):
    #https://albumentations.ai/docs/getting_started/mask_augmentation/
    def __init__(self, sample_dicts, **kwargs):
        self.sample_dicts =  sample_dicts
        self.kwargs = kwargs
    def __iter__(self):
        for gt_dict in self.sample_dicts:
            assert gt_dict['labels'] is not []
            assert gt_dict['masks'] is not []
            assert gt_dict['boxes'] is not []
            gt_dict = stack_tensors(gt_dict)
            yield gt_dict

def curried_amax(instance_masks):
    """
    squashes the instance masks to semantic masks, prioritizing vessel, then natural, then infra
    """
    return torch.amax(instance_masks, axis=0)

def all_arrays_equal(mask_arrays):
    # Check if there's at least two arrays in the list
    if len(mask_arrays) < 2:
        return False

    # Compare each array to the first one
    first_array = mask_arrays[0]
    for array in mask_arrays[1:]:
        if np.array_equal(first_array, array):
            return True

    return False

def masks_to_instance_mask_and_dict(mask_arrays, class_ids, starting_instance_id=1):
    """Instead of squashing instance masks, combines instance masks into an instance mask
    and return it plus dict mapping instance ids to class ids.

    Args:
        mask_arrays (_type_): _description_
        class_ids (_type_): _description_
        starting_instance_id (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    instance_mask = torch.zeros_like(mask_arrays[0], dtype=torch.int32)
    instance_id_to_semantic_id = {}

    for i, (mask, class_id) in enumerate(zip(mask_arrays, class_ids)):
        # this is 1 if there's only 1 instance mask, then 2, etc. count resets per sample.
        instance_mask[mask > 0] = starting_instance_id + i
        #TODO is that correct? TODO subsequent instances overwrite others
        instance_id_to_semantic_id[starting_instance_id + i] = class_id # hack to deal with processor quirk
    if len(np.unique(instance_mask)) <= 1:
        print("The mask is empty but shouldn't be")
    if all_arrays_equal(mask_arrays):
        raise ValueError("Masks should not be identical for the same image!")

    return instance_mask, instance_id_to_semantic_id


@functional_datapipe("m2fprocessor")
class Mask2FormerProcessorDP(IterDataPipe):
    #https://albumentations.ai/docs/getting_started/mask_augmentation/
    def __init__(self, sample_dicts, config_path, **kwargs):
        self.sample_dicts =  sample_dicts
        # TODO doesn't appear to be any way to use lcoal configs with this class, it tries to pull from repo and use class info from repo.
        # TODO https://pyimagesearch.com/2023/03/13/train-a-maskformer-segmentation-model-with-hugging-face-transformers/
        self.processor = AutoImageProcessor.from_pretrained(config_path, ignore_mismatched_sizes=True)
        self.kwargs = kwargs
    def __iter__(self):
        for sample_dict in self.sample_dicts:
            # https://pyimagesearch.com/2023/03/13/train-a-maskformer-segmentation-model-with-hugging-face-transformers/
            # we use pixel wise class annotations as input
            # need to use instance masks
            # if sample_dict['image_name'] == 'S1A_IW_GRDH_1SDV_20200905T143719_20200905T143744_034225_03FA12_7661.tif':
            #     print("the masks to isntance mask function used to? break this and sets values below 2 to ")
            if sample_dict['image_name'] == 'S1A_IW_GRDH_1SDV_20210827T001554_20210827T001619_039408_04A7B9_21A5.tif':
                print("this has empty masks but shouldn't at some point before masks_to...")
            sample_dict['masks'] = [mask for mask in sample_dict['masks']]
            sample_dict['boxes'] = [mask for mask in sample_dict['boxes']]
            if all_arrays_equal(sample_dict['masks']): #TODO hack to get rid of duplicate masks
                sample_dict['masks'] = sample_dict['masks'][0].unsqueeze(0)
                sample_dict['labels'] = sample_dict['labels'][0].unsqueeze(0)
            instance_mask, instance_id_to_semantic_id = masks_to_instance_mask_and_dict(sample_dict['masks'], sample_dict['labels'])
            if len(np.unique(instance_mask)) <= 1:
                raise ValueError("The mask is empty but shouldn't be")
            inputs = self.processor(images=[sample_dict['image']], segmentation_maps=[instance_mask], instance_id_to_semantic_id= instance_id_to_semantic_id, task_inputs=["panoptic"], reduce_labels=False, return_tensors="pt")
            inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()}
            yield inputs

#potentially just use processor to modify the outputs to pass to model? just get text encodings
# but yield instance masks and image since encode process converts semantic masks to list of masks anyway
def extract_bounding_box(mask) -> np.ndarray:
    """Extract the bounding box of a mask.
    :param mask: HxW numpy array
    :return: bounding box
    """
    pos = np.where(mask)

    if not (pos[0].size or pos[1].size):
        return np.array([0, 0, 0, 0])

    xmin = np.min(pos[1])
    xmax = np.max(pos[1]) + 1
    ymin = np.min(pos[0])
    ymax = np.max(pos[0]) + 1
    return np.array([xmin, ymin, xmax, ymax])