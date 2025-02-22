#!/usr/bin/env bash
export NCCL_LL_THRESHOLD=0

ARCH=$1
GPUS=$2
PORT=${PORT:-29501}

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    --use_env main.py --model $ARCH --batch-size 16 --epochs 300 --data-path /nws/user/lirenda621/Data/imagenet_1k \
     ${@:3}