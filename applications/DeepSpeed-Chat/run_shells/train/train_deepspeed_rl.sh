#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../

base_dir="/mnt/cephfs/hjh/train_record/nlp/deepSpeedExamples"
out_dir="${base_dir}/outs"

sft_ckp_dir="${out_dir}/actor-models/1.3b"
rw_ckp_dir="${out_dir}/reward-models/350m"
actor_zero_stage=1
reward_zero_stage=2

#-----------------
# reward model
#-----------------
cd training/step3_rlhf_finetuning

##multi gpu
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
  bash training_scripts/single_node/run_1.3b.sh \
  ${sft_ckp_dir} \
  ${rw_ckp_dir} \
  ${actor_zero_stage} \
  ${reward_zero_stage} \
  ${out_dir}/rl-models/1.3b-350m
