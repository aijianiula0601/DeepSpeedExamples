#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../../

base_dir="/mnt/cephfs/hjh/train_record/nlp/deepSpeedExamples/llama7b"
out_dir="${base_dir}/outs"
mkdir -p $out_dir

llama_model_name='huggyllama/llama-7b'
data_path="Dahoas/rm-static"
ZERO_STAGE=2

#-----------------
# step 1
#-----------------
cd training/step1_supervised_finetuning

CUDA_VISIBLE_DEVICES=4,5,6,7 \
deepspeed main.py \
   --data_path ${data_path} \
   --data_split 2,4,4 \
   --model_name_or_path ${llama_model_name} \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 16 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir ${out_dir}
