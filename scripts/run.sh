#!/bin/bash
# Run the full pipeline for one setting of Table 1 on a single node with 8 GPUs:
# SFT -> response pairs -> preference labels -> proxy RM -> SignCert-PO -> win rate.
#
# Usage: bash scripts/run.sh <setting> [seed]
#   <setting>: tldr_pythia1b
# Labeling with GPT-4.1 Nano reads the API key from OPENAI_API_KEY.
set -euo pipefail
cd "$(dirname "$0")/.."

SETTING=$1
SEED=${2:-1}
OUT=${OUT:-outputs}/$SETTING/seed$SEED
LAUNCH="accelerate launch --config_file configs/deepspeed_zero2.yaml --num_processes 8"

TLDR_PYTHIA=vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144
TLDR_QWEN=johannesack/summarize_from_feedback_tldr_3_filtered_oai_preprocessingQWEN_1748920372
ALPACA_QWEN=shinnosukeono/alpaca_farm_preprocessing_1771395192

# PAIRS: response pairs per prompt; RM_EPOCHS: passes over them in RM training;
# RM_BATCH: per-device RM batch size (total batch size 128).
case $SETTING in
tldr_pythia1b)
    MODEL=EleutherAI/pythia-1b-deduped DATA=$TLDR_PYTHIA TASK=tldr LABELER=skywork
    PAIRS=1 RM_EPOCHS=1 RM_BATCH=16 TOKENS=53 BETA=0.01 Q=0.4 STEPS=2000
    ;;
*)
    echo "unknown setting: $SETTING" >&2
    exit 1
    ;;
esac

$LAUNCH sft.py --base_model $MODEL --dataset $DATA --output_dir $OUT/sft --seed $SEED

python sample.py --sft_model $OUT/sft --dataset $DATA --output $OUT/pairs \
    --num_epochs $PAIRS --max_new_tokens $TOKENS --seed $SEED

python label.py --input $OUT/pairs --output $OUT/labeled --labeler $LABELER --task $TASK

$LAUNCH train_rm.py --sft_model $OUT/sft --dataset $OUT/labeled --output_dir $OUT/rm \
    --num_train_epochs $RM_EPOCHS --per_device_batch_size $RM_BATCH --seed $SEED

# 8 GPUs x 2 completions x 16 accumulation steps = 256 completions (32 prompts x K=8) per step.
$LAUNCH train.py --sft_model $OUT/sft --reward_model $OUT/rm --dataset $DATA \
    --output_dir $OUT/policy --seed $SEED \
    --quantile $Q --beta $BETA --max_steps $STEPS \
    --learning_rate 1e-6 --lr_scheduler_type constant \
    --num_generations 8 --per_device_train_batch_size 2 --gradient_accumulation_steps 16 \
    --max_prompt_length 512 --max_completion_length $TOKENS --temperature 0.7 \
    --scale_rewards False --loss_type dr_grpo \
    --use_vllm --vllm_mode colocate --vllm_gpu_memory_utilization 0.28 \
    --bf16 --disable_dropout --logging_steps 10 --save_strategy no --report_to none

python evaluate.py --model $OUT/policy --dataset $DATA --max_new_tokens $TOKENS \
    --output $OUT/eval.json
