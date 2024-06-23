#!/bin/bash

LOGDIR="/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_mnist_guided"
# VERSION=0

# RDMAV_FORK_SAFE=1 

poetry run mnist_train_guided \
    --beta-schedule cosine \
    --train-timesteps 1000 \
    --max-epochs 2000 \
    --batch-size 2048 \
    --device-id 1 \
    --layers-per-block 2 \
    --dim-mults 1 2 1 \
    --learning-rate 1e-4 \
    --dim 32 \
    --logdir $LOGDIR \
    --val-freq 10 \
    --dropout 0 \
    --blocks Block CrossAttnBlock Block \
    --uncond-prob 0.2 \
    --use-cross-attn \
    --val-freq 50 \
    --num-workers 11


# poetry run mnist_test \
#     --logdir $LOGDIR \
#     --version $VERSION \
#     --batch-size 1800
