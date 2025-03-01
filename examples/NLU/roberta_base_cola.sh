export num_gpus=2
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=5
export output_dir="./cola/"
CUDA_VISIBLE_DEVICES='0,1' python -m torch.distributed.launch --master_port 103 --nproc_per_node=$num_gpus \
--use-env examples/text-classification/run_glue.py \
--model_name_or_path roberta-base \
--task_name cola \
--do_train \
--do_eval \
--max_seq_length 512 \
--per_device_train_batch_size 32 \
--learning_rate 4e-4 \
--num_train_epochs 80 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0.06 \
--apply_lora \
--lora_r 8 \
--lora_alpha 16 \
--seed 5 \
--weight_decay 0.1 \
--project_name project_name \
--job_name job_name \
--use_learn_weight 0 \
--use_gate \
--use_load_balance \
--weight_main_init 0.95 \
--weight_other_init 0.05 \
--ddp_find_unused_parameters False 