#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../

base_dir="/mnt/cephfs/hjh/train_record/nlp/deepSpeedExamples"
out_dir="${base_dir}/outs"

#-----------------
# step 1
#-----------------

python train.py \
  --num-gpus 8 \
  --step 2 \
  --actor-model 1.3b \
  --reward-model 350m \
  --output-dir ${out_dir}
