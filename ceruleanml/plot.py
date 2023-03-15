import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from matplotlib.patches import Patch
from ceruleanml import data_creation


def plot_scene_wth_annotations(img_path: str, mask_arrs: list, annos: dict):
    """Plots a scene with a simple legend based on the 6 class COCO json.

    Args:
        img_path (str): The path to the scene
        mask_arrs (list): A list of decoded masks from decode_masks.
        annos (dict): The COCO json
    """
    # Create the figure
    fig, ax = plt.subplots()
    draw_img = Image.new('RGB', mask_arrs[0].shape, (255, 255, 255))
    # Draw the image and polygons
    img_pil_channels = Image.open(img_path).split()
    img_pil = img_pil_channels[0]
    draw = ImageDraw.Draw(draw_img)
    draw.text((10, 10), 'Image', fill=(0, 0, 0))
    draw_img.paste(img_pil)
    for i, poly in enumerate(mask_arrs):
        rows, cols = np.where(poly)
        # Combine the row and column indices into (x, y) coordinate tuples
        coords = list(zip(cols, rows))
        label = data_creation.class_list[annos[i]['category_id']]
        color = data_creation.class_dict[label]['cc']
        draw.polygon(coords, outline=color)
    # Show the image with polygons
    plt.imshow(np.array(draw_img))
    legend_elements = []
    for k, v in data_creation.class_dict.items():
        legend_elements.append(Patch(facecolor=[x/255 for x in v['cc']], edgecolor='r', label=k))

    ax.legend(handles=legend_elements, loc='lower left')

    plt.show()