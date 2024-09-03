# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_llama_mt_flatten_hpp_dpp.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_llama_mt_onset_hpp_ddp.py