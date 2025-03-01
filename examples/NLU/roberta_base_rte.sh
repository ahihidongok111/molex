export num_gpus=8
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=10
export output_dir="./rte/"
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -m torch.distributed.launch --master_port 5 --nproc_per_node=$num_gpus \
--use_env examples/text-classification/run_glue.py \
--model_name_or_path roberta-base \
--task_name rte \
--do_train \
--do_eval \
--use_indv_gate \
--max_seq_length 512 \
--per_device_train_batch_size 32 \
--learning_rate 5e-4 \
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
--seed 10 \
--weight_decay 0.1 \
--use_gate \
--use_load_balance \
--gate_lr 0.1 \
--gate_weight_decay 0.1 \
--layers_to_use 12 \
--ddp_find_unused_parameters False \
--project_name project_name \
--job_name job_name \
--g_balance 0.001 