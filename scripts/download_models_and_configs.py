from transformers import AutoModelForUniversalSegmentation, OneFormerForUniversalSegmentation, OneFormerConfig, OneFormerProcessor


config = OneFormerConfig.from_pretrained("shi-labs/oneformer_coco_swin_large", num_labels=3, ignore_mismatched_sizes=True) 
model = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large", config=config, ignore_mismatched_sizes=True)
model2 = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large", config=config, ignore_mismatched_sizes=True)
processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large", config=config, ignore_mismatched_sizes=True)