TILE_SIZE=1024
mkdir -p /root/experiments/cv2

mkdir -p /root/partitions/test_tiles_context_$TILE_SIZE/
gsutil -m rsync -r gs://ceruleanml/partitions/test_tiles_context_$TILE_SIZE/ /root/partitions/test_tiles_context_$TILE_SIZE/

mkdir -p /root/partitions/val_tiles_context_$TILE_SIZE/
gsutil -m rsync -r gs://ceruleanml/partitions/val_tiles_context_$TILE_SIZE/ /root/partitions/val_tiles_context_$TILE_SIZE/

mkdir -p /root/partitions/train_tiles_context_$TILE_SIZE/
gsutil -m rsync -r gs://ceruleanml/partitions/train_tiles_context_$TILE_SIZE/ /root/partitions/train_tiles_context_$TILE_SIZE/
