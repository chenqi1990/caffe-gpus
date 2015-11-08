#!/usr/bin/env sh

./build/tools/caffe_multi_gpu train \
	--gpu_start=0 \
	--gpu_end=1 \
	--solver=examples/imagenet/solver.prototxt
