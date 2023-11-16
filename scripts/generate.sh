#!/bin/bash

RDMAV_FORK_SAFE=1 OPENAI_LOGDIR=/vol/biomedic3/rrr2417/.tmp poetry run image_sample -- \
    --model_path /vol/biomedic3/rrr2417/.tmp/model004400.pt \
    --num_samples 100
