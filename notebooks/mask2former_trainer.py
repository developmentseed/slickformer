import sys
import random
import warnings
import numpy as np
import torch
import torch.nn as nn  # PyTorch Lightning NN (neural network) module
import torchvision
from torch.utils.data import DataLoader
import torchdata
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
import lightning as L
from ceruleanml.data_pipeline import put_image_in_dict, get_src_pths_annotations
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
import os
#for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
if not sys.warnoptions:
    warnings.simplefilter("ignore")
# Set the random seed
seed=0 # we need to set this for torch datapipe separately
random.seed(seed)
torch.set_float32_matmul_precision('medium') # if you have tensor cores

# Loading the train and validation set

train_dir = "/home/work/slickformer/data/partitions/train_tiles_context_0/"
train_imgs, train_annotations = get_src_pths_annotations(train_dir)
val_dir = "/home/work/slickformer/data/partitions/val_tiles_context_0/"
val_imgs, val_annotations = get_src_pths_annotations(val_dir)
test_dir = "/home/work/slickformer/data/partitions/test_tiles_context_0/"

# Setting up the datapipes

train_i_coco_pipe = torchdata.datapipes.iter.IterableWrapper(iterable=[train_annotations])
train_l_coco_pipe = torchdata.datapipes.iter.IterableWrapper(iterable=[train_annotations])

# We'll train on random crops of masks to focus on the most informative parts of scene for more efficient training.


# Putting datapipes in a pytorch-lightning DataModule

# +
import matplotlib.pyplot as plt

class Mask2FormerDataModule(L.LightningDataModule):
    def __init__(self, config_path, train_dir, val_dir, test_dir, batch_size, num_workers, crop_size=512):
        super().__init__()
        self.train_dir, self.val_dir, self.test_dir = train_dir, val_dir, test_dir
        self.bs = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size
        self.config_path = config_path

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
                .channel_first_norm_to_tensor()
                .remap_remove()
                .stack_to_tensor()
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
                .channel_first_norm_to_tensor()
                .remap_remove()
                .stack_to_tensor()
            )

            test_imgs, test_annotations = get_src_pths_annotations(self.test_dir)
            test_i_coco_pipe = torchdata.datapipes.iter.IterableWrapper(iterable=[test_annotations])
            test_l_coco_pipe = torchdata.datapipes.iter.IterableWrapper(iterable=[test_annotations])
            test_source_pipe_processed = (
            test_i_coco_pipe.get_scene_paths(self.test_dir) # get source items from the collection
                .read_tiff()
                .map(put_image_in_dict)
            )
            test_labels_pipe_processed = (
                test_l_coco_pipe.decode_masks()
            )
            self.test_dp = (
            test_source_pipe_processed.zip(test_labels_pipe_processed)
            .combine_src_label_dicts() # we don't crop for the test set TODO, do we also not crop for validation?
            .channel_first_norm_to_tensor()
            .remap_remove()
            .stack_to_tensor()
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
        train_dp = self.train_dp.m2fprocessor(self.config_path)
        return DataLoader(num_workers= self.num_workers, pin_memory=True, dataset=train_dp, batch_size=self.bs)

    def val_dataloader(self):
        val_dp  = self.val_dp.m2fprocessor(self.config_path)
        return DataLoader(num_workers= self.num_workers, pin_memory=True, dataset=val_dp, batch_size=self.bs)

    def test_dataloader(self):
        test_dp = self.test_dp.m2fprocessor(self.config_path)
        return DataLoader(num_workers= self.num_workers, pin_memory=True, dataset=test_dp, batch_size=self.bs)

    def predict_dataloader(self):
        test_dp = self.test_dp.m2fprocessor(self.config_path)
        return DataLoader(num_workers= self.num_workers, pin_memory=True, dataset=test_dp, batch_size=self.bs)


# +
data_config_path= "/home/work/slickformer/custom_processors/preprocessor_config.json"
onef_dm = Mask2FormerDataModule(data_config_path, train_dir, val_dir, test_dir, batch_size=1, num_workers=os.cpu_count() - 1, crop_size=512)
onef_dm.setup(stage="train") #what's the purpose of stage?
# 10 is limit for 24 Gb gpu memory and 512 crop size

onef_dm.setup(stage="train") #what's the purpose of stage?

# +
from matplotlib.ticker import MaxNLocator, FuncFormatter

onef_dm.show_batch(0)

# +
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

class Mask2FormerLightningModel(L.LightningModule):
    def __init__(
        self,
        config_path,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()  # saves all hparams as self.hparams
        # can try other universal segmentation models: https://github.com/huggingface/transformers/pull/20766/files#r1050493186
        config = Mask2FormerConfig.from_pretrained(config_path, ignore_mismatched_sizes=True) 
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-cityscapes-panoptic", config=config, cache_dir="./model_cache", ignore_mismatched_sizes=True)
        #by default the above method sets eval mode, set to training
        self.model.train()
        # Move the model to the default CUDA device (if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def forward(self, xb):
        return self.model(xb)

    def one_step(self, batch):
        # potential edge case, we squash instance masks to semantic masks. it's possible to lose 
        # mask label 3 from mask but not class labels since we don't edit class labels
        #self.model.train() # for some reason this needs to be set here, not picked up in init
        outputs = self.model(pixel_values=batch["pixel_values"], mask_labels=batch["mask_labels"], class_labels=batch['class_labels'])
        return outputs.loss
    
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

# +
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

profiler = SimpleProfiler(dirpath=".", filename="perf_logs")
logger = TensorBoardLogger("tb_logs", name="my_model")

trainer = L.Trainer(
    max_epochs = 5,
    accelerator="auto",
    devices = 1 if torch.cuda.is_available else None,
    logger=logger,
    profiler=profiler,
    fast_dev_run=True
    # deterministic=True # can't set this when using cuda
)
# -

model_config_path= "/home/work/slickformer/custom_models/config.json"
model = Mask2FormerLightningModel(model_config_path)

# need to run this in terminal because for some reason created dirs are owned by root even though docker container built for user 1000

trainer.fit(model, datamodule=onef_dm)
print(profiler.summary())


