import os
import torch
import torch.nn.functional as F
import time
import random
import librosa
import numpy as np
import soundfile
import matplotlib.pyplot as plt
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from data.maestro import MaestroMultiTask
from data.collate import collate_fn
from data.io import events_to_notes
# from models.crnn import CRnn
from tqdm import tqdm
import museval
import argparse
import wandb

from data.tokenizers import Tokenizer
from models.enc_dec import EncDecConfig, EncDecPos

batch_size = 5
num_workers = 8
evaluate_step_frequency = 10000
save_step_frequency = 1000
training_steps = 1000000
debug = False
filename = Path(__file__).stem
segment_seconds = 10.
lr = 1e-4
frames_num = 1001
max_token_len = 1536
# wandb_log = True if rank == 0 else False

model_name = "AudioLlama"
checkpoints_dir = Path("./checkpoints", filename, model_name)

root = "/root/autodl-tmp/maestro-v3.0.0"

# if wandb_log:
#     wandb.init(
#         project="mini_piano_transcription",
#         name=filename
#     )

tokenizer = Tokenizer()

# Dataset
train_dataset = MaestroMultiTask(
    root=root,
    split="train",
    segment_seconds=segment_seconds,
    tokenizer=tokenizer,
    max_token_len=max_token_len,
    task="onset"
)

test_dataset = MaestroMultiTask(
    root=root,
    split="validation",
    segment_seconds=segment_seconds,
    tokenizer=tokenizer,
    max_token_len=max_token_len,
    task="onset"
)

# # Sampler
# train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
# eval_train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False)
# eval_test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

# enc_model_name = "HPPNet"
# enc_model = get_model(enc_model_name)
model_file = "/root/autodl-tmp/hpp-10secondsInput-120000.pt"
enc_model = torch.load(model_file)
# enc_model.to(device)

config = EncDecConfig(
    block_size=max_token_len + 1, 
    vocab_size=tokenizer.vocab_size, 
    padded_vocab_size=tokenizer.vocab_size, 
    n_layer=6,
    n_head=12,
    n_embd=768,
    audio_n_embd=88*4
)


model = EncDecPos(config)
# get number of params in M
print("# Params:", sum([p.numel() for p in model.parameters()]) / 1e6, "M")