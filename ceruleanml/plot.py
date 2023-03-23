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
from ceruleanml.data_pipeline import remap_class_dict
import ceruleanml.data_creation

def plot_instance_masks(img: Union[str, np.ndarray, torch.Tensor], mask_arrs: list, mask_cat_ids: dict, class_list: list = data_creation.class_list, outline_only: bool = True):
    """Plots a scene with a simple legend based on the 6 class COCO json or remapped classes.

    Args:
        img_path (str): The path to the scene
        mask_arrs (list): A list of decoded masks from decode_masks.
        mask_cat_ids (dict): The category ids for mask arrs.
        class_list (list): The list of category keys that map to data_creation.class_dict. 
            Used to remap plot colors to actual mask ids. User needs to know how the prediction 
            or mask IDs map to the class dict. For the 3 class model, it should currently be
            1: coincident vessel 2: infra_slick, 3 natural seep.
        outline_only (bool): If True, only draw polygon outlines (default False).
    """
    new_class_dict = remap_class_dict(data_creation.class_dict, class_list)
    new_class_list = list(new_class_dict.keys())
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
        #TODO this is a quick and dirty way to find contours. couldn't find
        # a good way to plot the result of find_contours from skimage quickly
        rows, cols = np.where(poly)
        # Combine the row and column indices into (x, y) coordinate tuples
        coords = list(zip(cols, rows))
        label = new_class_list[mask_cat_ids[i]]
        color = new_class_dict[label]['cc']
        try:
            if outline_only:
                draw.polygon(coords, outline=color)
            else:
                draw.polygon(coords, outline=color, fill=color)
        except TypeError: # coord less than 2 issue TODO figure otu why this happens with thresholding sometimes. poly too small?
            pass
    # Show the image with polygons
    plt.imshow(np.array(draw_img))
    legend_elements = []
    for k, v in new_class_dict.items():
        legend_elements.append(Patch(facecolor=[x/255 for x in v['cc']], edgecolor='r', label=k))

    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()