import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from matplotlib.patches import Patch
from ceruleanml import data_creation
from typing import Union
import torch
from torchvision import transforms
import pathlib

def plot_scene_annotations(img: Union[str, np.ndarray, torch.Tensor], mask_arrs: list, mask_cat_ids: dict, outline_only: bool = True):
    """Plots a scene with a simple legend based on the 6 class COCO json.

    Args:
        img_path (str): The path to the scene
        mask_arrs (list): A list of decoded masks from decode_masks.
        mask_cat_ids (dict): The category ids for mask arrs.
        outline_only (bool): If True, only draw polygon outlines (default False).
    """
    if isinstance(img, str) or isinstance(img, pathlib.PosixPath): 
        img_pil_channels = Image.open(img).split()
        img_pil = img_pil_channels[0]
    elif isinstance(img, torch.Tensor) or isinstance(img, np.ndarray):
        img_pil = transforms.ToPILImage()(img)
    else:
        raise TypeError("img must be a string, torch tensor, or numpy array")
    # Create the figure
    fig, ax = plt.subplots()
    draw_img = Image.new('RGB', mask_arrs[0].shape, (255, 255, 255))
    # Draw the image and polygons
    draw = ImageDraw.Draw(draw_img)
    draw.text((10, 10), 'Image', fill=(0, 0, 0))
    draw_img.paste(img_pil)
    for i, poly in enumerate(mask_arrs):
        rows, cols = np.where(poly)
        # Combine the row and column indices into (x, y) coordinate tuples
        coords = list(zip(cols, rows))
        label = data_creation.class_list[mask_cat_ids[i]]
        color = data_creation.class_dict[label]['cc']
        if outline_only:
            draw.polygon(coords, outline=color)
        else:
            draw.polygon(coords, outline=color, fill=color)
    # Show the image with polygons
    plt.imshow(np.array(draw_img))
    legend_elements = []
    for k, v in data_creation.class_dict.items():
        legend_elements.append(Patch(facecolor=[x/255 for x in v['cc']], edgecolor='r', label=k))

    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()