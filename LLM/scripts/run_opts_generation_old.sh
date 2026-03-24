MODEL_PATH=models/Qwen3-1.7B
DATA_PATH=data
OUTPUT_PATH=data/qwen3_1.7b/opts_gen

# Generation
python3 -m verl.trainer.main_opts_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$DATA_PATH/test.parquet \
    data.prompt_key=prompt \
    data.batch_size=512 \
    data.n_samples=2 \
    data.n_rounds=32 \
    data.output_path=$OUTPUT_PATH/test_64.parquet \
    model.path=$MODEL_PATH \
    critic.model.path=$MODEL_PATH \
    critic.forward_micro_batch_size_per_gpu=8 \
    rollout.root_tuct=0.1 \
    rollout.temperature=1.0 \
    rollout.top_p=0.95 \
    rollout.prompt_length=1024 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.9 \
    rollout.max_num_batched_tokens=65536 \
    algorithm.lam=0.995 \
