# %%
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig, AutoImageProcessor

config = Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-small-cityscapes-panoptic", cache_dir="../config_cache", ignore_mismatched_sizes=True) 


# %%
config.id2label = {
        "1": "Infrastructure Slick",
        "2": "Natural Seep",
        "3": "Vessel Slick"
    }

config.label2id = {
        "Infrastructure Slick": 1,
        "Natural Seep" : 2,
        "Vessel Slick" : 3
    }

config.backbone_config.id2label = {
        "1": "Infrastructure Slick",
        "2": "Natural Seep",
        "3": "Vessel Slick"
    }

config.backbone_config.label2id = {
        "Infrastructure Slick": 1,
        "Natural Seep" : 2,
        "Vessel Slick" : 3
    }
# %%
config.save_pretrained("../config/")
#%%
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-cityscapes-panoptic", config=config, cache_dir="../model_cache", ignore_mismatched_sizes=True)

#%%
model.save_pretrained("../custom_models")


#%%
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-panoptic", cache_dir="../processor_cache", ignore_mismatched_sizes=True)

# %%
processor.do_normalize= False
processor.do_reduce_labels= True
processor.ignore_index= 0
processor.do_rescale= False
processor.do_resize= False
processor.num_labels= 3

processor.save_pretrained("../custom_processors")
# %%
