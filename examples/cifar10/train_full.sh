#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe_multi_gpu train \
    --gpu_start=0 \
    --gpu_end=1 \
    --solver=examples/cifar10/cifar10_full_solver.prototxt

