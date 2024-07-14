export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
torchrun --nproc_per_node 1 --master_port 20133 main.py --data-path '/apdcephfs_cq8/share_300043402/rileyrlluo/codes/sql_classification/train_data/ft_data.json' \
    --n-global-epoch 100 \
    --hidden-size 500 \
    --do-finetune \
    --batch-size 4 \
    --sm-batch-size 1 \
    --lora-rank 8 \
    --history-length 20 \
    --lora-dropout 0.1 \
    --n-ft-epoch 2 \
    --inference-direction 'bi' \
    --data-aug \
    --useid \
    --base-model-path /apdcephfs_cq8/share_300043402/rainbowlin/base_data/base_model/LLM_base_model/ \
    --base-model LLAMA2-7B \
    --run-name sql_llama2_7b \
    --output-dir '/apdcephfs_cq8/share_300043402/rileyrlluo/codes/sql_classification' \
