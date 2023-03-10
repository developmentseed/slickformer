# # this is best run on a 32 core machine to make the train dataset in under 30 minutes
# python3.9 /slickformer/work/ceruleanml/random_scene_select.py
# gsutil cp /slickformer/data/partitions/test_scenes.txt gs://ceruleanml/partitions/
# gsutil cp /slickformer/data/partitions/val_scenes.txt gs://ceruleanml/partitions/
# gsutil cp /slickformer/data/partitions/train_scenes.txt gs://ceruleanml/partitions/

MEMTILE_SIZE=0 # Setting MEMTILE_SIZE=0 generates a coco dataset with the full scenes, instead of tiling them first

# # dataset with vv imagery + aux
ceruleanml make-coco-dataset /slickformer/data/partitions/test_scenes.txt /slickformer/data/aux_datasets /slickformer/data/partitions/test_tiles_context_$MEMTILE_SIZE/ $MEMTILE_SIZE

ceruleanml make-coco-dataset /slickformer/data/partitions/val_scenes.txt /slickformer/data/aux_datasets /slickformer/data/partitions/val_tiles_context_$MEMTILE_SIZE/ $MEMTILE_SIZE

ceruleanml make-coco-dataset /slickformer/data/partitions/train_scenes.txt /slickformer/data/aux_datasets /slickformer/data/partitions/train_tiles_context_$MEMTILE_SIZE/ $MEMTILE_SIZE
