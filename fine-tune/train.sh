#!/bin/bash

hostfile=""
deepspeed --hostfile=$hostfile fine-tune.py  \
    --report_to "none" \
    --data_path "data/belle_chat_ramdon_10k.json" \
    --model_name_or_path "/root/autodl-tmp/base_model/baichuan-inc/Baichuan2-7B-Base" \
    --output_dir "output/baichuan2-7B-sft-1e5-0426" \
    --model_max_length 512 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.0 \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --deepspeed ds_config.json \
    --bf16 True \
    --tf32 True