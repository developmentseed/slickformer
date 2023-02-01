"""Console script for ceruleanml."""
import os
import time
from datetime import date
from pathlib import Path
from typing import List

import click
import dask
from dask.distributed import Client, progress

import ceruleanml.data as data


@click.group()
def main():
    """CeruleanML CLI scripts for data processing."""


# TODO class list associated with single source of truth
# set empty list as default for class_list and make an arg to cli funcs
def make_coco_metadata(
    name="Cerulean Dataset V2",
    description: str = "Cerulean Dataset V2",
    version: str = "1.0",
    class_list: List[str] = [
        "infra_slick",
        "natural_seep",
        "coincident_vessel",
        "recent_vessel",
        "old_vessel",
        "ambiguous",
    ],
):
    """Creates COCO Metadata

    Args:
        dname (str, optional): The name of the dataset. Defaults to "Cerulean Dataset V2".
        description (str, optional): A string description fo the coco dataset. Defaults to "Cerulean Dataset V2".
        version (str, optional): Defaults to "1.0".
        class_list (List, optional): The lsit of classes. Ordering of list must match integer ids used during annotation. Defaults to [ "infra_slick", "natural_seep", "coincident_vessel", "recent_vessel", "old_vessel", "ambiguous", ].

    Returns:
        dict : A COCO metadata dictionary.
    """
    info = {
        "description": description,
        "url": "none",
        "version": version,
        "year": 2022,
        "contributor": "Skytruth",
        "date_created": str(date.today()),
    }
    licenses = [{"url": "none", "id": 1, "name": name}]
    assert len(class_list) > 0
    categories = [
        {"supercategory": "slick", "id": i + 1, "name": cname}
        for i, cname in enumerate(class_list)
    ]  # order matters, check that this matches the ids used when annotating if you get a data loading error
    return {
        "info": info,
        "licenses": licenses,
        "images": [],
        "annotations": [],
        "categories": categories,
    }


@main.command()
@click.argument("partition_list", nargs=1)
@click.argument("aux_data_path", nargs=1)
@click.argument("coco_outdir", nargs=1)
@click.argument("tile_length", type=int, nargs=1)
def make_coco_dataset_with_tiles(
    partition_list: str,
    aux_data_path: str,
    coco_outdir: str,
    name: str = "TiledCeruleanDatasetV2",
    tile_length: int = 512,
):
    """Create the dataset with tiles and context files (ship density and infra distance).

    Args:
        partition_list (str): the path to the text file containing the scene ids for a given partition "train_scenes.txt", "val_scenes.txt", or "test_scenes.txt"
        aux_data_path (str): the path to the folder containing the aux
            files TODO update (currently only infra_locations.json). This is located on ceruleanml bucket under aux_datasets.
        coco_outdir (str): the folder path to save the coco json and the folder
            of tiled images.
        name (str): the output name of the coco json file. also stored in the coco json metadata to tag the dataset.
        tile_length (str): square length size of the tiles in the tile grid. this can be set to any value. Default is 512.

    Example:
    make_coco_dataset_with_tiles(
        partition_list ="/root/data/partitions/test_scenes.txt",
        aux_data_path ="/root/data/aux_datasets",
        coco_outdir ="/root/partitions/test_tiles_context_512",
        tile_length = 512)
    """
    start = time.time()
    os.makedirs(coco_outdir, exist_ok=True)
    os.makedirs(os.path.join(coco_outdir, "tiled_images"), exist_ok=True)
    coco_tiler = data.COCOtiler(os.path.join(coco_outdir, "tiled_images"))

    aux_datasets = [
        os.path.join(aux_data_path, "infra_locations_01_cogeo.tiff"),
        "ship_density",
    ]
    with Client() as client:  # this needs to be commented out to use single threaded for profiling
        print("Dask client dashboard link: ", client.dashboard_link)
        coco_outputs = []
        with open(partition_list) as file:
            for scene_index, line in enumerate(file):
                scene_path = Path(line.rstrip())
                layer_dirs = list(map(str, scene_path.glob("*png")))
                scene_data_tuple = dask.delayed(coco_tiler.save_background_img_tiles)(
                    scene_id=scene_path.name,
                    layer_paths=layer_dirs,
                    aux_datasets=aux_datasets,
                    aux_resample_ratio=8,  # TODO call out if this is the one place to change resample ratio for all three bands or not
                    tile_length=tile_length,
                )
                coco_output = dask.delayed(coco_tiler.create_coco_from_photopea_layers)(
                    scene_index, scene_data_tuple, layer_dirs, tile_length=tile_length
                )
                coco_outputs.append(coco_output)
        final_coco_output = make_coco_metadata(name=name)
        # when we create a distributed client
        coco_outputs = client.persist(
            coco_outputs
        )  # start computation in the background
        progress(coco_outputs)  # watch progress
        coco_outputs = client.compute(coco_outputs, sync=True)
        # coco_outputs = dask.compute(
        #     *coco_outputs, scheduler="single-threaded"
        # )  # convert to final result when done
        for co in coco_outputs:
            final_coco_output["images"].extend(co["images"])
            final_coco_output["annotations"].extend(co["annotations"])
        for i, ann in enumerate(final_coco_output["annotations"]):
            ann["id"] = i
        coco_tiler.save_coco_output(
            final_coco_output,
            os.path.join(coco_outdir, f"./instances_{name.replace(' ', '')}.json"),
        )
        num_images = len(final_coco_output["images"])
        print(f"Number of seconds for {num_images} images: {time.time() - start}")
        print(f"Images and COCO JSON have been saved in {coco_outdir}.")
    print("Now, sync images to destination bucket on gcp")


@main.command()
@click.argument("partition_list", nargs=1)
@click.argument("coco_outdir", nargs=1)
def make_coco_dataset_no_tiles(
    partition_list: str,
    coco_outdir: str,
    name="UntiledCeruleanDatasetV2NoContextFiles",
):
    """Create the dataset without tiling and without context files.

    Args:
        partition_list (str): the path to the text file containing the scene ids for a given partition "train_scenes.txt", "val_scenes.txt", or "test_scenes.txt"
        coco_outdir (str): the path to save the coco json and the folder
            of tiled images.
        name (str): the output name of the coco json file. also stored in the coco json metadata to tag the dataset.
    """
    os.makedirs(coco_outdir, exist_ok=True)
    os.makedirs(os.path.join(coco_outdir, "untiled_images"), exist_ok=True)
    coco_tiler = data.COCOtiler(os.path.join(coco_outdir, "untiled_images"))
    with Client() as client:  # this needs to be commented out to use single threaded for profiling
        print("Dask client dashboard link: ", client.dashboard_link)
        coco_outputs = []
        with open(partition_list) as file:
            for scene_index, line in enumerate(file):
                scene_path = Path(line.rstrip())
                layer_dirs = list(map(str, scene_path.glob("*png")))
                delayed_tuple = dask.delayed(
                    data.fetch_sentinel1_reprojection_parameters
                )(scene_path.name)
                scene_data_tuple = (
                    delayed_tuple[1],
                    delayed_tuple[2],
                    delayed_tuple[3],
                )
                coco_output = dask.delayed(
                    coco_tiler.create_coco_from_photopea_layers_no_tile
                )(
                    scene_index,
                    scene_data_tuple,
                    layer_dirs,
                )
                coco_outputs.append(coco_output)
        final_coco_output = make_coco_metadata(name=name)
        # when we create a distributed client
        coco_outputs = dask.persist(coco_outputs)
        # start computation in the background
        progress(coco_outputs)  # watch progress
        coco_outputs = client.compute(coco_outputs, sync=True)
        for co in coco_outputs:
            final_coco_output["images"].extend(co["images"])
            final_coco_output["annotations"].extend(co["annotations"])
        for i, ann in enumerate(final_coco_output["annotations"]):
            ann["id"] = i
        coco_tiler.save_coco_output(
            final_coco_output,
            os.path.join(coco_outdir, f"./instances_{name.replace(' ', '')}.json"),
        )
    print(f"Images and COCO JSON have been saved in {coco_outdir}.")
    print("Now, sync images to destination bucket on gcp")


@main.command()
@click.argument("partition_list", nargs=1)
@click.argument("coco_outdir", nargs=1)
@click.argument("tile_length", type=int, nargs=1)
def make_coco_dataset_no_context(
    partition_list: str,
    coco_outdir: str,
    name="TiledCeruleanDatasetV2NoContextFiles",
    tile_length: int = 512,
):
    """Create the dataset with tiles but without context files (ship density and infra distance).

    Args:
        partition_list (str): the path to the text file containing the scene ids for a given partition "train_scenes.txt", "val_scenes.txt", or "test_scenes.txt"
        coco_outdir (str): the path to save the coco json and the folder
            of tiled images.
        name (str): the output name of the coco json file. also stored in the coco json metadata to tag the dataset.
        tile_length (str): square length size of the tiles. this can be set to any value. Default is 512.
    """
    start = time.time()
    with Client() as client:  # this needs to be commented out to use single threaded for profiling
        print("Dask client dashboard link: ", client.dashboard_link)
        os.makedirs(coco_outdir, exist_ok=True)
        os.makedirs(os.path.join(coco_outdir, "tiled_images_no_context"), exist_ok=True)
        coco_output = make_coco_metadata(name=name)
        coco_tiler = data.COCOtiler(
            os.path.join(coco_outdir, "tiled_images_no_context")
        )
        coco_outputs = []

        with open(partition_list) as file:
            for scene_index, line in enumerate(file):
                scene_path = Path(line.rstrip())
                layer_dirs = list(map(str, scene_path.glob("*png")))
                scene_data_tuple = dask.delayed(coco_tiler.save_background_img_tiles)(
                    scene_id=scene_path.name,
                    layer_paths=layer_dirs,
                    aux_datasets=[],
                    aux_resample_ratio=8,  # TODO call out if this is the one place to change resample ratio for all three bands or not
                    tile_length=tile_length,
                )
                coco_output = dask.delayed(coco_tiler.create_coco_from_photopea_layers)(
                    scene_index, scene_data_tuple, layer_dirs, tile_length=tile_length
                )
                coco_outputs.append(coco_output)
        final_coco_output = make_coco_metadata(name=name)
        # when we create a distributed client, dask.compute uses that isntead of thread scheduler by default
        coco_outputs = client.persist(
            coco_outputs
        )  # start computation in the background
        progress(coco_outputs)  # watch progress
        coco_outputs = client.compute(coco_outputs, sync=True)
        # coco_outputs = dask.compute(*coco_outputs, scheduler="processes")
        for co in coco_outputs:
            final_coco_output["images"].extend(co["images"])
            final_coco_output["annotations"].extend(co["annotations"])
        for i, ann in enumerate(final_coco_output["annotations"]):
            ann["id"] = i
        coco_tiler.save_coco_output(
            final_coco_output,
            os.path.join(coco_outdir, f"./instances_{name.replace(' ', '')}.json"),
        )
        num_images = len(final_coco_output["images"])
        print(f"Number of seconds for {num_images} images: {time.time() - start}")
        print(f"Images and COCO JSON have been saved in {coco_outdir}.")
    print("Now, sync images to destination bucket on gcp")


if __name__ == "__main__":
    main()
