#!/usr/bin/env bash

# Evaluation
python3 -m verl.trainer.main_eval \
    data.path=data/qwen3_1.7b/gen/test_64.parquet \
    data.prompt_key=prompt \
    data.response_key=responses \
    custom_reward_function.path=recipe/open_math_reasoning/compute_score.py \
    custom_reward_function.name=compute_score_data_source
