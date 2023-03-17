from pycocotools import mask as maskUtils
import os
from albumentations.augmentations.geometric.resize import LongestMaxSize, SmallestMaxSize
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
import skimage.io as skio
import albumentations as A

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
                for anno in annotations['annotations']:
                    if im_record['id'] == anno['image_id']:
                        segment = anno['segmentation']
                        # from the source, order is height then width https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/_mask.pyx#L288
                        rle = maskUtils.frPyObjects(segment, anno['height'], anno['width'])
                        mask_arrs.append(maskUtils.decode(rle)*anno['category_id'])
                yield mask_arrs, anno

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
        for src_img, mask_arrs in self.tups:
            t = transform(image = src_img, masks=mask_arrs)
            yield t