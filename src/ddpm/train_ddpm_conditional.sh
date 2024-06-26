#!/bin/bash

LOGDIR="/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_mnist_guided"
# VERSION=0

# CUDA_VISIBLE_DEVICES=0,1 RDMAV_FORK_SAFE=1 

poetry run mnist_train_guided \
    --beta-schedule cosine \
    --train-timesteps 1000 \
    --max-epochs 10000 \
    --batch-size 1200 \
    --device-id 1 \
    --layers-per-block 2 \
    --dim-mults 1 2 2 \
    --learning-rate 1e-3 \
    --dim 32 \
    --logdir $LOGDIR \
    --val-freq 50 \
    --dropout 0 \
    --blocks Block CrossAttnBlock CrossAttnBlock \
    --uncond-prob 0.1 \
    --use-cross-attn \
    --num-workers 10 \
    --num-sanity-val-steps 1


# poetry run mnist_test \
#     --logdir $LOGDIR \
#     --version $VERSION \
#     --batch-size 1800
