#!/bin/bash

LOGDIR="/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_mnist_guided"
CLS_LOGDIR="/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_mnist_classifier/"
poetry run mnist_editing_metrics --logdir $LOGDIR --version 45 --cls-logdir $CLS_LOGDIR --cls-version 47
