#!/bin/bash

base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=mil_ct_deepspeed
OUTPUT_DIR=${base_dir}/model_outputs

mkdir -p $OUTPUT_DIR

deepspeed --hostfile host --include localhost:5,6,7,8 \
${base_dir}/deepspeed_train.py \
--deepspeed \
--deepspeed_config ${base_dir}/ds_config.json \