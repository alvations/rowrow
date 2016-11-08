#!/bin/bash

set -e

PATH_TO_MODEL=/home/ltan/Paddle/demo/rowrow/waddlewaddle/model

paddle train \
--config='train.conf' \
--save_dir=${PATH_TO_MODEL} \
--use_gpu=true \
--num_passes=100 \
--show_parameter_stats_period=1000 \
--trainer_count=4 \
--log_period=10 \
--dot_period=5 \
2>&1 | tee 'train.log'
