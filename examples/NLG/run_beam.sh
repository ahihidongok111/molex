
var="GPT2_M"

CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --master_port 139 --nproc_per_node=1 \
    --use_env src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M/${var}/model.26290.pt \
    --platform local \
    --lora_dim 4 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M/${var} \
    --output_file predict.${var}.jsonl \
    --use_gate \
    --use_learn_weight 0 \
    --weight_main_init 0.95 \
    --weight_other_init 0.05 \
    --layers_to_use 12