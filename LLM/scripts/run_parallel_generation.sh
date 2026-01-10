MODEL_PATH=models/Qwen3-1.7B
DATA_PATH=data
OUTPUT_PATH=data/qwen3_1.7b/gen

# Generation
python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$DATA_PATH/test.parquet \
    data.prompt_key=prompt \
    data.batch_size=1024 \
    data.n_samples=64 \
    data.output_path=$OUTPUT_PATH/test_64.parquet \
    model.path=$MODEL_PATH \
    rollout.temperature=1.0 \
    rollout.top_p=0.95 \
    rollout.prompt_length=1024 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.9 \
    rollout.max_num_batched_tokens=65536
