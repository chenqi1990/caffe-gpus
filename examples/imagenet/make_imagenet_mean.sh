#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

./build/tools/compute_image_mean /media/chenqi/C2CE0C08CE0BF407/ILSVRC2012/ilsvrc12_train_lmdb \
  imagenet_mean.binaryproto

echo "Done."
