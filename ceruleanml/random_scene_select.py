import os
import random

source_path = "/root/data-cv2/"
dest_path = "/root/data/partitions/"

val_frac = 0.2
test_frac = 0.1


def partition_scenes(source_path, val_frac, test_frac):
    """Applies random selection of scenes for mutually exclusive partitions (train, validation, test).
    Partition fractions must range from 0.0 to 1.0.
    Args:
        source_path (str): Parent folder with class-name subfolders.
        val_frac (float): Percent of items to allocate to validation.
        test_frac (float): Percent of items to allocate to testing.
    Returns:
        train_scenes (list): List of train items.
        val_scenes (list): List of validation items.
        test_scenes (list): List of test items.
    """
    class_folders = [f.path for f in os.scandir(source_path) if f.is_dir()]
    train_scenes = []
    val_scenes = []
    test_scenes = []

    for class_folder in class_folders:
        print("Selecting from class: ", class_folder)
        class_scenes = [f.path for f in os.scandir(class_folder) if f.is_dir()]
        random.seed(4)
        random.shuffle(class_scenes)
        num_val = int(len(class_scenes) * val_frac)
        num_test = int(len(class_scenes) * test_frac)

        val_scenes += class_scenes[0:num_val]
        test_scenes += class_scenes[num_val : num_val + num_test]
        train_scenes += class_scenes[num_val + num_test :]

    # Check for mutual exclusivity
    assert len(train_scenes + val_scenes + test_scenes) > 0
    assert len(set(train_scenes + val_scenes + test_scenes)) == len(
        train_scenes + val_scenes + test_scenes
    )
    return train_scenes, val_scenes, test_scenes


train_scenes, val_scenes, test_scenes = partition_scenes(
    source_path, val_frac, test_frac
)

partitions = (
    ["train_scenes.txt", train_scenes],
    ["val_scenes.txt", val_scenes],
    ["test_scenes.txt", test_scenes],
)

for fname, scenes in partitions:
    with open(os.path.join(dest_path, fname), "w") as f:
        for item in scenes:
            f.write("%s\n" % item)
