#!/bin/bash

LOGDIR="/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_mnist_uncond"
VERSION=0

RDMAV_FORK_SAFE=1 poetry run mnist_train \
    --beta-schedule cosine \
    --train-timesteps 1000 \
    --max-epochs 1000 \
    --batch-size 1800 \
    --device-id -1 \
    --batch-size 512 \
    --layers-per-block 2 \
    --dim-mults 1 2 1 \
    --learning-rate 1e-4 \
    --dim 32 \
    --logdir $LOGDIR \
    --val-freq 10 \
    --dropout 0 \
    --blocks Block AttnBlock Block


# poetry run mnist_test \
#     --logdir $LOGDIR \
#     --version $VERSION \
#     --batch-size 1800
