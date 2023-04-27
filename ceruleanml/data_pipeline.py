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
from transformers import OneFormerProcessor
from transformers import OneFormerForUniversalSegmentation

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

def channel_first_norm_to_tensor(gdict):
    # channel first needs to happen after pil crop
    # norm is faster if applied post pil crop by about 1 second
    gdict.update({"image": torch.Tensor(np.moveaxis(gdict['image'],2,0) / 255)})
    # gdict = stack_tensors(gdict)
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

def curried_amax(instance_masks):
    """
    squashes the instance masks to semantic masks, prioritizing vessel, then natural, then infra
    TODO replace with https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/oneformer/image_processing_oneformer.py#L875
    to treat inputs as instances
    requires using oneformer image processor instead of OneFormerProcessor
    """
    return torch.amax(instance_masks, axis=0)

@functional_datapipe("ofprocessor")
class OneFormerProcessorDP(IterDataPipe):
    #https://albumentations.ai/docs/getting_started/mask_augmentation/
    def __init__(self, sample_dicts, **kwargs):
        self.sample_dicts =  sample_dicts
        # todo figure out how to inspect config, the config class repr?
        # todo this drastically slows everything down
        dummy_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large",  num_labels=3, ignore_mismatched_sizes=True)
        self.processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large",  num_labels=3, ignore_mismatched_sizes=True)
        # weird design forces us to have to load model here to define num_text 
        # https://github.com/praeclarumjj3/transformers/blob/b723fbb1713be397d71c6bb56f693a277196d02d/tests/models/oneformer/test_modeling_oneformer.py#L519
        self.processor.image_processor.num_text = dummy_model.config.num_queries - dummy_model.config.text_encoder_n_ctx
        self.kwargs = kwargs
    def __iter__(self):
        for sample_dict in self.sample_dicts:
            #todo should this be done upfront for all iamges and masks in the coco dataset? seems like a chore and maybe text embeddings not worth
            results = self.processor.encode_inputs(images=[sample_dict['image']], segmentation_maps=[curried_amax(sample_dict['masks'])], task_inputs=["panoptic"], return_tensors="pt")
            results['mask_labels'] = results['mask_labels'][0].long()
            # transformers docs are a bit whack, encode_inputs dict has key for class_labels but this is not a tensor with 
            # correct shape for the model
            results['class_labels'] = results['class_labels'][0].long()
            yield results

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