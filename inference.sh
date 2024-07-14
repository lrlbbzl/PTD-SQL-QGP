export CUDA_VISIBLE_DEVICES=0
python inference.py --data-path './test_data/test_data_list.json' \
    --history-length 10 \
    --inference-direction 'right' \
    --partial-num 0 \
    --data-augment \
    --base-model-path './models/modelscope' \
    --base-model Llama-2-7b-ms \
    --output-dir './outputs/llama2_7b' \
    --lora-weights-path './outputs/llama2_7b'