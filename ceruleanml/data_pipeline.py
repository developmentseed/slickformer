from pycocotools import mask as maskUtils
import os
from albumentations.augmentations.geometric.resize import LongestMaxSize, SmallestMaxSize

def decode_masks(scene_id: str, annotations:dict):
    """decodes mask based on scene and annotation metadata.

    Args:
        scene_id (dict): A scene_id in annotations['images']. assumes working from full S1 imagery.
        annotations (dict): the COCO JSON dict

    Returns:
        _type_: _description_
    """
    scene_coco_record = [im_record for im_record in annotations['images'] if im_record['big_image_original_fname'] == scene_id+".tif"][0]
    mask_arrs = []
    annos = []
    for anno in annotations['annotations']:
        if scene_coco_record['id'] == anno['image_id']:
            segment = anno['segmentation']
            # from the source, order is height then width https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/_mask.pyx#L288
            rle = maskUtils.frPyObjects(segment, anno['height'], anno['width'])
            mask_arrs.append(maskUtils.decode(rle)*anno['category_id'])
            annos.append(anno)
    return annos, mask_arrs

@functional_datapipe("get_scene_paths")
class GetScenePaths(IterDataPipe):
    def __init__(self, source_dp, annotations, annotations_dir, **kwargs):
        self.source_dp = source_dp
        self.annotations = annotations
        self.annotations_dir = annotations_dir
        self.kwargs = kwargs

    def __iter__(self):
        #might need to sort by key here to align with labels dp TODO
        for record in self.annotations['images']:
            yield os.path.join(self.annotations_dir, "tiled_images", record['file_name'])

@functional_datapipe("longest_max_size_pad_labels")
class PadToLongestMaxSizeLabels(IterDataPipe):
    def __init__(self, source_dp, decoded_masks, longest_max_size, **kwargs):
        self.source_dp = source_dp
        self.decoded_masks = decoded_masks
        self.longest_max_size= longest_max_size
        self.kwargs = kwargs
    def __iter__(self):
        for decoded_mask in self.decoded_masks:
            yield from LongestMaxSize(decoded_mask, max_size = longest_max_size, always_apply=False)

# @functional_datapipe("decode_masks")
# class DecodeMasks(IterDataPipe):
#     """decodes mask based on scene and annotation metadata.

#     Args:
#         scene_id (dict): A scene_id in annotations['images']. assumes working from full S1 imagery.
#         annotations (dict): the COCO JSON dict

#     Returns:
#         _type_: _description_
#     """
#     def __init__(self, source_dp, annotations, scene_ids, **kwargs):
#         self.source_dp = source_dp
#         self.annotations = annotations
#         self.annotations_dir = annotations_dir
#         self.kwargs = kwargs

#     def __iter__(self):
#         for record in self.annotations['images']:
#             yield os.path.join(self.annotations_dir, "tiled_images", record['file_name'])

#     scene_coco_record = [im_record for im_record in annotations['images'] if im_record['big_image_original_fname'] == scene_id+".tif"][0]
#     mask_arrs = []
#     annos = []
#     for anno in annotations['annotations']:
#         if scene_coco_record['id'] == anno['image_id']:
#             segment = anno['segmentation']
#             # from the source, order is height then width https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/_mask.pyx#L288
#             rle = maskUtils.frPyObjects(segment, anno['height'], anno['width'])
#             mask_arrs.append(maskUtils.decode(rle)*anno['category_id'])
#             annos.append(anno)
#     return annos, mask_arrs