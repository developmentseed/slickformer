# ### Icevision Inference and Evalutation and Comparison to FastAI

import torch
import torchvision  # noqa
from fastai.data.block import DataBlock
from fastai.data.transforms import IndexSplitter
from fastai.vision.augment import Resize
from fastai.vision.data import ImageBlock, MaskBlock
from icevision import Dataset, tfms
from icevision.metrics.confusion_matrix import SimpleConfusionMatrix
from icevision.models.checkpoint import model_from_checkpoint

from ceruleanml import data, preprocess
from ceruleanml.coco_load_fastai import (
    get_image_path,
    record_collection_to_record_ids,
    record_to_mask,
)
from ceruleanml.data import class_mapping_coco
from ceruleanml.evaluation import (
    get_cm_for_torchscript_model_mrcnn,
    get_cm_for_torchscript_model_unet,
)

# model files
# 6 class model
# fastai_unet_experiment_dir = "/root/data/experiments/cv2/29_Jun_2022_06_36_38_fastai_unet"
# tracing_model_cpu_pth = (
#     f"{fastai_unet_experiment_dir}/tracing_cpu_224_120__512_36__4_34_0.0003_0.436.pt"
# )
# fastai_unet_cm_title="Fastai Unet Confusion Matrix: 29_Jun_2022_06_36_38"
# 3 class model
fastai_unet_experiment_dir = "/root/experiments/cv2/26_Jul_2022_21_57_24_fastai_unet"
tracing_model_cpu_pth = f"{fastai_unet_experiment_dir}/tracing_cpu_test_32_34_224_0.824_30.pt"
fastai_unet_cm_title = "Fastai Unet Confusion Matrix: 26_Jul_2022_21_57_24"
icevision_experiment_dir = "/root/data/experiments/cv2/20_Jul_2022_00_14_15_icevision_maskrcnn"
scripted_model = torch.jit.load(f"{icevision_experiment_dir}/scripting_cpu_test_28_34_224_58.pt")
icevision_mrcnn_cm_title = "Icevision MRCNN Confusion Matrix: 20_Jul_2022_00_14_15"
checkpoint_path = "/root/data/experiments/cv2/20_Jul_2022_00_14_15_icevision_maskrcnn/state_dict_test_28_34_224_58.pt"
class_names = [
    "background",
    "infra_slick",
    "recent_vessel",
]  # if remapping changes from vessel and slick, change sneed to be made
## loading icevision validation set
size = 224
negative_sample_count = 0
negative_sample_count_val = 0
area_thresh = 0

data_path = "/root/"
mount_path = "/root/data"

val_set = "val-with-context-512"
tiled_images_folder_val = "tiled_images"
json_name_val = "instances_TiledCeruleanDatasetV2.json"
coco_json_path_val = f"{mount_path}/partitions/{val_set}/{json_name_val}"
tiled_images_folder_val = f"{mount_path}/partitions/{val_set}/{tiled_images_folder_val}"
remove_list = ["ambiguous", "natural_seep"]
class_names_to_keep = [
    "background",
    "infra_slick",
    "recent_vessel",
]
remap_dict = {  # only remaps coincident and old to recent
    3: 4,
    5: 4,
}

# since we remove ambiguous and natural seep and remap all vessels to 1 and include background
num_classes = 3
num_classes_mrcnn = 3
num_classes_fastai = 3
class_map = {v: k for k, v in data.class_mapping_coco_inv.items()}
class_ints = list(range(1, len(list(class_map.keys())[:-1]) + 1))

record_collection_with_negative_small_filtered_val = preprocess.load_set_record_collection(
    coco_json_path_val,
    tiled_images_folder_val,
    area_thresh,
    negative_sample_count_val,
    preprocess=True,
    class_names_to_keep=class_names_to_keep,
    remap_dict=remap_dict,
    remove_list=remove_list,
)
record_ids_val = record_collection_to_record_ids(record_collection_with_negative_small_filtered_val)

valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=size)])
valid_ds = Dataset(record_collection_with_negative_small_filtered_val, valid_tfms)
# valid_ds = Dataset(
#     record_collection_with_negative_small_filtered_val[0:10]
#     + record_collection_with_negative_small_filtered_val[100:102]
#     + record_collection_with_negative_small_filtered_val[-2:],
#     valid_tfms,
# )


# ### Loading Fastai Validation Set
# for fastai we need the train set to parse the val set with fastai dls
# mount_path = "/root/"
# train_set = "train-with-context-512"
# tiled_images_folder_train = "tiled_images"
# json_name_train = "instances_TiledCeruleanDatasetV2.json"

# coco_json_path_train = f"{mount_path}/partitions/{train_set}/{json_name_train}"
# tiled_images_folder_train = f"{mount_path}/partitions/{train_set}/{tiled_images_folder_train}"
# record_collection_with_negative_small_filtered_train = preprocess.load_set_record_collection(
#     coco_json_path_train,
#     tiled_images_folder_train,
#     area_thresh,
#     negative_sample_count_val,
#     preprocess=False,
#     class_names_to_keep=class_names_to_keep,
#     remap_dict=remap_dict,
#     remove_list=remove_list,
# )
# record_ids_train = record_collection_to_record_ids(
#     record_collection_with_negative_small_filtered_train
# )

# record_ids_val = record_collection_to_record_ids(record_collection_with_negative_small_filtered_val)

# assert len(set(record_ids_train)) + len(set(record_ids_val)) == len(record_ids_train) + len(
#     record_ids_val
# )

# train_val_record_ids = record_ids_train + record_ids_val
# combined_record_collection = (
#     record_collection_with_negative_small_filtered_train
#     + record_collection_with_negative_small_filtered_val
# )


# def get_val_indices(combined_ids, val_ids):
#     return list(range(len(combined_ids)))[-len(val_ids) :]


# def get_image_by_record_id(record_id):
#     return get_image_path(combined_record_collection, record_id)


# def get_mask_by_record_id(record_id):
#     return record_to_mask(combined_record_collection, record_id)


# # Constructing a FastAI DataBlock that uses parsed COCO Dataset from icevision parser. aug_transforms can only be used with_context=True
# val_indices = get_val_indices(train_val_record_ids, record_ids_val)

# coco_seg_dblock = DataBlock(
#     blocks=(ImageBlock, MaskBlock(codes=class_ints)),  # ImageBlock is RGB by default, uses PIL
#     get_x=get_image_by_record_id,
#     splitter=IndexSplitter(val_indices),
#     get_y=get_mask_by_record_id,
#     # batch_tfms=batch_transfms,
#     item_tfms=Resize(512),
#     n_inp=1,
# )

# model = torch.jit.load(tracing_model_cpu_pth)
# dls = coco_seg_dblock.dataloaders(source=train_val_record_ids, batch_size=1)

# cm_unet, f1_unet = get_cm_for_torchscript_model_unet(
#     dls,
#     model,
#     fastai_unet_experiment_dir,
#     semantic_mask_conf_thresh=0.2,
#     num_classes=num_classes_fastai,
#     class_names=class_names,
#     normalize=None,
#     title=fastai_unet_cm_title,
# )

# cm_unet, f1_unet = get_cm_for_torchscript_model_unet(
#     dls,
#     model,
#     fastai_unet_experiment_dir,
#     semantic_mask_conf_thresh=0.2,
#     num_classes=num_classes_fastai,
#     class_names=class_names,
#     normalize="true",
#     title=fastai_unet_cm_title,
# )

# cm_unet, f1_unet = get_cm_for_torchscript_model_unet(
#     dls,
#     model,
#     fastai_unet_experiment_dir,
#     semantic_mask_conf_thresh=0.2,
#     num_classes=num_classes_fastai,
#     class_names=class_names,
#     normalize="pred",
#     title=fastai_unet_cm_title,
# )


cm_mrcnn, f1_mrcnn = get_cm_for_torchscript_model_mrcnn(
    valid_ds,
    scripted_model,
    save_path=icevision_experiment_dir,
    mask_conf_threshold=0.01,
    bbox_conf_threshold=0.7,
    num_classes=num_classes_mrcnn,
    normalize=None,
    class_names=class_names,
    title=icevision_mrcnn_cm_title,
)


# cm_mrcnn, f1_mrcnn = get_cm_for_torchscript_model_mrcnn(
#     valid_ds,
#     scripted_model,
#     save_path=icevision_experiment_dir,
#     mask_conf_threshold=0.01,
#     bbox_conf_threshold=0.7,
#     num_classes=num_classes_mrcnn,
#     normalize="true",
#     class_names=class_names,
#     title=icevision_mrcnn_cm_title,
# )

# cm_mrcnn, f1_mrcnn = get_cm_for_torchscript_model_mrcnn(
#     valid_ds,
#     scripted_model,
#     save_path=icevision_experiment_dir,
#     mask_conf_threshold=0.01,
#     bbox_conf_threshold=0.7,
#     num_classes=num_classes_mrcnn,
#     normalize="pred",
#     class_names=class_names,
#     title=icevision_mrcnn_cm_title,
# )

class_names = ["Background", "Infrastructure", "Recent Vessel"]

checkpoint_and_model = model_from_checkpoint(
    checkpoint_path,
    model_name="torchvision.mask_rcnn",
    backbone_name="resnet34_fpn",
    img_size=224,
    classes=[
        "background",
        "infra_slick",
        "recent_vessel",
    ],
    is_coco=False,
)

model = checkpoint_and_model["model"]
model_type = checkpoint_and_model["model_type"]
backbone = checkpoint_and_model["backbone"]
class_map = checkpoint_and_model["class_map"]
img_size = checkpoint_and_model["img_size"]
model_type, backbone, class_map, img_size

infer_dl = model_type.infer_dl(valid_ds, batch_size=1, shuffle=False)

preds = model_type.predict_from_dl(model, infer_dl, keep_images=True, detection_threshold=0.5)

cm = SimpleConfusionMatrix()
cm.accumulate(preds)

_ = cm.finalize()

ax = cm.plot(figsize=15, normalize=None, values_size=24)

import matplotlib.pyplot as plt

instance_cm_pth = (
    f"{icevision_experiment_dir}/{icevision_mrcnn_cm_title.replace(' ', '').lower()}"
    + "_instance.png"
)
plt.tight_layout()
ax.savefig(instance_cm_pth)

print(f"Saved instance cm to: {instance_cm_pth}")
