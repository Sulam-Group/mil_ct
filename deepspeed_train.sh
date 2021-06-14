#!/bin/bash

base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=mil_ct_deepspeed
OUTPUT_DIR=${base_dir}/model_outputs

mkdir -p $OUTPUT_DIR

deepspeed --hostfile host --include localhost:0,1 \
${base_dir}/deepspeed_train.py \
--deepspeed \
--cf ${base_dir}/hemorrhage_detection_config.json \
--output_dir model_outputs \
--deepspeed_config ${base_dir}/ds_config.json \
--job_name $JOB_NAME