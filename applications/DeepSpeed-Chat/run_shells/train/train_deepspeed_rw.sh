#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../

base_dir="/mnt/cephfs/hjh/train_record/nlp/deepSpeedExamples"
out_dir="${base_dir}/outs"

#-----------------
# reward model
#-----------------
cd training/step2_reward_model_finetuning

#sigle gpu
#CUDA_VISIBLE_DEVICES=0 \
#bash training_scripts/single_gpu/run_350m.sh \
#${out_dir}/reward-models/350m

##multi gpu
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
bash training_scripts/single_node/run_350m.sh \
${out_dir}/reward-models/350m