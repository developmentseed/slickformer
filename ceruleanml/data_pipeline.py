from pycocotools import mask as maskUtils

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