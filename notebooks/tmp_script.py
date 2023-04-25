# %%
import sys
import random
import warnings
import numpy as np
import torch
import torch.nn as nn  # PyTorch Lightning NN (neural network) module
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, default_collate
import torch.optim as optim 
from torchdata.dataloader2 import DataLoader2
import torchdata
import pytorch_lightning as pl
from pytorch_lightning.callbacks import DeviceStatsMonitor
from ceruleanml import plot
from ceruleanml import evaluation
from ceruleanml.data_pipeline import put_image_in_dict, get_src_pths_annotations, channel_first_norm_to_tensor, stack_tensors
from transformers import AutoModelForUniversalSegmentation, OneFormerConfig
import skimage.io as skio

if not sys.warnoptions:
    warnings.simplefilter("ignore")
# Set the random seed
seed=0 # we need to set this for torch datapipe separately
random.seed(seed)

# %% [markdown]
# Loading the train and validation set

# %%
train_dir = "../data/partitions/train_tiles_context_0/"
train_imgs, train_annotations = get_src_pths_annotations(train_dir)
val_dir = "../data/partitions/val_tiles_context_0/"
val_imgs, val_annotations = get_src_pths_annotations(val_dir)
test_dir = "../data/partitions/test_tiles_context_0/"

# %% [markdown]
# Setting up the datapipes

# %%
train_i_coco_pipe = torchdata.datapipes.iter.IterableWrapper(iterable=[train_annotations])
train_l_coco_pipe = torchdata.datapipes.iter.IterableWrapper(iterable=[train_annotations])
train_source_pipe_processed = (
    train_i_coco_pipe.get_scene_paths(train_dir)  # get source items from the collection
    .read_tiff()
)

# %%
train_labels_pipe_processed = (
    train_l_coco_pipe.decode_masks()
)

# %% [markdown]
# We'll train on random crops of masks to focus on the most informative parts of scene for more efficient training.

# %%
train_dp = (
    train_source_pipe_processed.zip(train_labels_pipe_processed)
    .random_crop_mask_if_exists(2000,2000)
    .map(channel_first_norm_to_tensor)
    .map(stack_tensors)
)

# %%
torchdata.datapipes.utils.to_graph(dp=train_dp)

# %% [markdown]
# Putting datapipes in a pytorch-lightning DataModule

# %%
import matplotlib.pyplot as plt

class OneFormerDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, test_dir, batch_size, num_workers, crop_size=2000):
        super().__init__()
        self.train_dir, self.val_dir, self.test_dir = train_dir, val_dir, test_dir
        self.bs = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size

    def setup(self, stage):
        if stage is not None:  # train/val/test/predict
            train_imgs, train_annotations = get_src_pths_annotations(self.train_dir)
            train_i_coco_pipe = torchdata.datapipes.iter.IterableWrapper(iterable=[train_annotations])
            train_l_coco_pipe = torchdata.datapipes.iter.IterableWrapper(iterable=[train_annotations])
            train_source_pipe_processed = (
                train_i_coco_pipe.get_scene_paths(self.train_dir)  # get source items from the collection
                .read_tiff()
            )
            train_labels_pipe_processed = (
                train_l_coco_pipe.decode_masks()
            )
            self.train_dp = (
                train_source_pipe_processed.zip(train_labels_pipe_processed)
                .random_crop_mask_if_exists(self.crop_size, self.crop_size)
                .map(channel_first_norm_to_tensor)
                .map(stack_tensors)
            )
            # TODO if val processing mirrors train processing, this could be factored out to a func
            val_imgs, val_annotations = get_src_pths_annotations(self.val_dir)
            val_i_coco_pipe = torchdata.datapipes.iter.IterableWrapper(iterable=[val_annotations])
            val_l_coco_pipe = torchdata.datapipes.iter.IterableWrapper(iterable=[val_annotations])
            val_source_pipe_processed = (
                val_i_coco_pipe.get_scene_paths(self.val_dir) # get source items from the collection
                .  read_tiff()
            )
            val_labels_pipe_processed = (
                val_l_coco_pipe.decode_masks()
            )
            self.val_dp = (
                val_source_pipe_processed.zip(val_labels_pipe_processed)
                .random_crop_mask_if_exists(self.crop_size,self.crop_size)
                .map(channel_first_norm_to_tensor)
                .map(stack_tensors)
            )

            test_imgs, test_annotations = get_src_pths_annotations(self.test_dir)
            test_i_coco_pipe = torchdata.datapipes.iter.IterableWrapper(iterable=[test_annotations])
            test_l_coco_pipe = torchdata.datapipes.iter.IterableWrapper(iterable=[test_annotations])
            test_source_pipe_processed = (
            test_i_coco_pipe.get_scene_paths(self.test_dir) # get source items from the collection
                .read_tiff()
                .map(put_image_in_dict)
                .map(channel_first_norm_to_tensor)
            )
            test_labels_pipe_processed = (
                test_l_coco_pipe.decode_masks()
            )
            self.test_dp = (
            test_source_pipe_processed.zip(test_labels_pipe_processed)
            .combine_src_label_dicts() # we don't crop for the test set TODO, do we also not crop for validation?
            .map(channel_first_norm_to_tensor)
            .map(stack_tensors)
            )

    def graph_dp(self):
        return torchdata.datapipes.utils.to_graph(dp=self.train_dp)

    def show_batch(self, channel=0):
        """
        channel 0 - vv radar
        channel 1 infra distance
        channel 2 historical vessel traffic

        """
        assert channel in [0,1,2]

        def closest_factors(n):
            factor1 = int(n**0.5)
            factor2 = n // factor1
            while factor1 * factor2 != n:
                factor1 -= 1
                factor2 = n // factor1
            return factor1, factor2

        nrows, ncols = closest_factors(self.bs)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
        
        for item, ax in zip(self.train_dp, fig.axes):
            if channel == 0:
                vv_db_units = np.log10(np.array(item["image"][channel,:,:])) * 10
            else:
                vv_db_units = np.array(item["image"][channel,:,:])
            vv_db_units[vv_db_units == -np.inf] = np.nan
            min_value = np.nanpercentile(vv_db_units, 5)
            max_value = np.nanpercentile(vv_db_units, 95)
            rescaled = (vv_db_units - min_value) / (max_value - min_value)
            im = ax.imshow(rescaled)
            
            # Create an individual colorbar for the current image
            cbar = fig.colorbar(im, ax=ax, shrink=0.7)
            
            # Set the number of ticks on the colorbar
            cbar.locator = MaxNLocator(nbins=5)
            cbar.update_ticks()

            # Format the tick labels
            tick_formatter = FuncFormatter(lambda x, pos: f'{min_value + x * (max_value - min_value):.2f}')
            cbar.ax.yaxis.set_major_formatter(tick_formatter)

        plt.tight_layout()

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dp.ofprocessor().batch(self.bs), batch_size=None)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dp.ofprocessor().batch(self.bs), batch_size=None)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dp.ofprocessor().batch(self.bs), batch_size=None)

    def predict_dataloader(self):
        return DataLoader(dataset=self.test_dp.ofprocessor().batch(self.bs), batch_size=None)

# %%
onef_dm = OneFormerDataModule(train_dir, val_dir, test_dir, batch_size=4, num_workers=1, crop_size=2000)

# %%
onef_dm.setup(stage="train") #what's the purpose of stage?

# %%
from matplotlib.ticker import MaxNLocator, FuncFormatter

onef_dm.show_batch(0)

# %%
for i in onef_dm.train_dataloader():
    i
    break

# %%
i[3].pixel_values.shape

# %%
i[3]['mask_labels']

# %%
i[0]['pixel_mask'].unique()

# %%
i[0].keys()

# %%
i[0]['mask_labels'][0].shape

# %%
from transformers import AdamW
class Backbone(nn.Module):
    def __init__(
        self, model_name, in_chans, num_classes, pretrained, global_pool, drop_rate
    ):
        super().__init__()
        # loads from huggingface if not downloaded
        
        #by default the above method sets eval mode, set to training
        self.backbone.train()

    def forward(self, xb):
        return self.backbone(xb)

class OneFormerLightningModel(pl.LightningModule):
    def __init__(
        self,
        num_classes=3,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()  # saves all hparams as self.hparams
        # this allows num_classes to differ from coco
        # https://github.com/huggingface/transformers/pull/17257/files
        self.config = OneFormerConfig.from_pretrained("shi-labs/oneformer_coco_swin_large", num_labels=num_classes, ignore_mismatched_sizes=True) 
        # can try other universal segmentation models: https://github.com/huggingface/transformers/pull/20766/files#r1050493186
        self.model = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large", config = self.config, ignore_mismatched_sizes=True)
        #by default the above method sets eval mode, set to training
        self.model.train()

    def forward(self, xb):
        return self.model(xb)

    def one_step(self, batch):
        # potential edge case, we squash instance masks to semantic masks. it's possible to lose 
        # mask label 3 from mask but not class labels since we don't edit class labels
        loss, output = self(pixel_values=batch["pixel_values"], mask_labels=batch["mask_labels"], class_labels=batch['class_labels'], text_inputs=batch['text_inputs'], task_inputs=batch['task_inputs'])
        return loss
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        loss = self.one_step(batch)
        self.log("trn_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.one_step(batch)
        self.log("val_loss", loss, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, score = self.one_step(batch)
        self.log("tst_loss", loss, prog_bar=True, logger=True)

# %%
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything

# %%
trainer = Trainer(
    max_epochs = 1,
    accelerator="auto",
    devices = 1 if torch.cuda.is_available else None,
)

# %%
model = OneFormerLightningModel()

# %% [markdown]
# need to run this in terminal because for some reason created dirs are owned by root even though docker container built for user 1000

# %%


trainer.fit(model, datamodule=onef_dm)

# %%


# %%
from transformers import OneFormerProcessor
processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_tiny",  num_labels=3)

panoptic_inputs = processor(images=data['image'], segmentation_maps=data['masks'], task_inputs=["panoptic"], return_tensors="pt")
for k,v in panoptic_inputs.items():
  print(k,v.shape)

# %%
from transformers import  AutoImageProcessor, MaskFormerForInstanceSegmentation
image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco", num_labels=3)
instance_inputs = image_processor(images=data['image'], return_tensors="pt")
for k,v in instance_inputs.items():
  print(k,v.shape)

# %%
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")

# %%
v

# %%
model = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large")

# %%
type(model)

# %%
dir(model.model)

# %%
import torch

# forward pass
with torch.no_grad():
    outputs = model(**panoptic_inputs)

# %%

panoptic_segmentation = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
print(panoptic_segmentation.keys())

# %%

from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches

def draw_panoptic_segmentation(segmentation, segments_info):
    # get the used color map
    viridis = cm.get_cmap('viridis', torch.max(segmentation))
    fig, ax = plt.subplots()
    ax.imshow(segmentation)
    instances_counter = defaultdict(int)
    handles = []
    # for each segment, draw its legend
    for segment in segments_info:
        segment_id = segment['id']
        segment_label_id = segment['label_id']
        segment_label = model.config.id2label[segment_label_id]
        label = f"{segment_label}-{instances_counter[segment_label_id]}"
        instances_counter[segment_label_id] += 1
        color = viridis(segment_id)
        handles.append(mpatches.Patch(color=color, label=label))
        
    ax.legend(handles=handles)
    plt.savefig('cats_panoptic.png')

draw_panoptic_segmentation(**panoptic_segmentation)

# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# Groundtruth datapipe with non cropped images. We will use these for inference with the trained model.

# %%
gt_train_dp = (train_dp
                    .map(evaluation.remap_gt_dict)
                    .map(evaluation.stack_boxes)
)

# %%
from torchmetrics import detection

m = detection.mean_ap.MeanAveragePrecision(box_format='xyxy', iou_type='bbox', iou_thresholds=[.5], rec_thresholds=None, max_detection_thresholds=None, class_metrics=True)

m.update(preds=[pred_dict_thresholded_nms], target=[test_sample])

from pprint import pprint
pprint(m.compute())


