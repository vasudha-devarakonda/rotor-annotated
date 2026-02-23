#!/bin/bash

set -e

GPU_DEVICE=0
BATCH_SIZE=256
EPOCHS=2

cleanup() {
    rm -f test.txt
}

for BUDGET in 716.80 1003.52 1361.92
do
    cleanup
    python train.py -gpu -net resnet18 -b $BATCH_SIZE -e $EPOCHS \
        -iter-limit -1 -no-eval -budget $BUDGET -gpu-device $GPU_DEVICE
done


for BUDGET in 3072.00 4710.40 6758.40
do
    cleanup
    python train.py -gpu -net resnet50 -b $BATCH_SIZE -e $EPOCHS \
        -iter-limit -1 -no-eval -budget $BUDGET -gpu-device $GPU_DEVICE
done


for BUDGET in 7168.00 10035.20 13619.20
do
    cleanup
    python train.py -gpu -net resnet152 -b $BATCH_SIZE -e $EPOCHS \
        -iter-limit -1 -no-eval -budget $BUDGET -gpu-device $GPU_DEVICE
done

for BUDGET in 5939.20 8396.80 10854.40
do
    cleanup
    python train.py -gpu -net inceptionv3 -b $BATCH_SIZE -e $EPOCHS \
        -iter-limit -1 -no-eval -budget $BUDGET -gpu-device $GPU_DEVICE
done

for BUDGET in 1536.00 2355.20 3379.20
do
    cleanup
    python train.py -gpu -net googlenet -b $BATCH_SIZE -e $EPOCHS \
        -iter-limit -1 -no-eval -budget $BUDGET -gpu-device $GPU_DEVICE
    
    cleanup
    sleep 30
done