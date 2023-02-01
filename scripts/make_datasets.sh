# this is best run on a 32 core machine to make the train dataset in under 30 minutes
python3.9 /root/work/ceruleanml/random_scene_select.py
gsutil cp /root/data/partitions/test_scenes.txt gs://ceruleanml/partitions/
gsutil cp /root/data/partitions/val_scenes.txt gs://ceruleanml/partitions/
gsutil cp /root/data/partitions/train_scenes.txt gs://ceruleanml/partitions/

TILE_SIZE=1024

# # dataset with vv imagery + aux
ceruleanml make-coco-dataset-with-tiles /root/data/partitions/test_scenes.txt /root/data/aux_datasets /root/partitions/test_tiles_context_$TILE_SIZE/ $TILE_SIZE
gsutil -m rsync -r /root/partitions/test_tiles_context_$TILE_SIZE/ gs://ceruleanml/partitions/test_tiles_context_$TILE_SIZE/

ceruleanml make-coco-dataset-with-tiles /root/data/partitions/val_scenes.txt /root/data/aux_datasets /root/partitions/val_tiles_context_$TILE_SIZE/ $TILE_SIZE
gsutil -m rsync -r /root/partitions/val_tiles_context_$TILE_SIZE/ gs://ceruleanml/partitions/val_tiles_context_$TILE_SIZE/

ceruleanml make-coco-dataset-with-tiles /root/data/partitions/train_scenes.txt /root/data/aux_datasets /root/partitions/train_tiles_context_$TILE_SIZE/ $TILE_SIZE
gsutil -m rsync -r /root/partitions/train_tiles_context_$TILE_SIZE/ gs://ceruleanml/partitions/train_tiles_context_$TILE_SIZE/
