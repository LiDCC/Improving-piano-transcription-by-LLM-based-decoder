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

def cleanup():
    dist.destroy_process_group()

def train(args):
    dist.init_process_group(backend="nccl", init_method="env://")
    # rank = dist.get_rank()
    # world_size = dist.get_world_size()
    # torch.cuda.set_device(args.local_rank)
    # device = torch.device(f'cuda:{args.local_rank}')
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # local_rank = int(os.environ["LOCAL_RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    print(f"Local rank: {local_rank}")
    
    # Default parameters
    batch_size = 8
    num_workers = 16
    evaluate_step_frequency = 10000 // world_size
    save_step_frequency = 1000 // world_size
    training_steps = 1000000 // world_size
    debug = False
    filename = Path(__file__).stem
    segment_seconds = 10.
    lr = 1e-4
    frames_num = 1001
    max_token_len = 1536
    wandb_log = True if rank == 0 else False

    model_name = "AudioLlama"
    model_setting = "tiny"
    checkpoints_dir = Path("./checkpoints", model_setting, filename, model_name)
    
    root = "/root/autodl-tmp/maestro-v3.0.0"

    if wandb_log:
        wandb.init(
            project="mini_piano_transcription",
            name=model_setting + "_" + filename
        )

    tokenizer = Tokenizer()
    
    # Dataset
    train_dataset = MaestroMultiTask(
        root=root,
        split="train",
        segment_seconds=segment_seconds,
        tokenizer=tokenizer,
        max_token_len=max_token_len,
        task="offset"
    )

    test_dataset = MaestroMultiTask(
        root=root,
        split="validation",
        segment_seconds=segment_seconds,
        tokenizer=tokenizer,
        max_token_len=max_token_len,
        task="offset"
    )

    # Sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    eval_train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    eval_test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=True,
        # shuffle=True,
        pin_memory=True
    )

    eval_train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=eval_train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=True,
        # shuffle=False,
        pin_memory=True
    )

    eval_test_dataloader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        sampler=eval_test_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers, 
        drop_last=True,
        # shuffle=False,
        pin_memory=True
    )

    enc_model_name = "HPPNet"
    enc_model = get_model(enc_model_name)
    enc_model.to(device)
    enc_model = DDP(enc_model, device_ids=[rank])

    config = EncDecConfig(
        block_size=max_token_len + 1, 
        vocab_size=tokenizer.vocab_size, 
        padded_vocab_size=tokenizer.vocab_size, 
        n_layer=4, 
        n_head=8, 
        n_embd=512,
        audio_n_embd=88*4
    )
    

    model = EncDecPos(config)
    model.to(device)
    model = DDP(model, device_ids=[rank])

    # Optimizer
    optimizer = optim.AdamW(list(enc_model.parameters()) + list(model.parameters()), lr=lr)

    # Create checkpoints directory
    if rank == 0:
        Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # Train
    # for step, data in tqdm(enumerate(train_dataloader), total=training_steps, desc="Training"):
        
    #     audio = data["audio"].to(device)
    #     input_token = data["token"][:, 0 : -1].to(device)
    #     target_token = data["token"][:, 1 :].to(device)
    #     target_mask = data["mask"][:, 1 :].to(device)

    #     optimizer.zero_grad()

    #     enc_model.train()
    #     model.train()
    #     audio_emb = enc_model.module.run_on_batch(audio)
    #     audio_emb = audio_emb.permute(0, 2, 3, 1).reshape(batch_size, 501, 4 * 88)

    #     logits, loss = model(audio_emb=audio_emb, idx=input_token, target=target_token, target_mask=target_mask)
        
    #     loss.backward()
    #     optimizer.step()
        
    #     if rank == 0 and step % evaluate_step_frequency == 0:
    #         print("Evaluating ...")
    #         train_loss = validate(enc_model, model, eval_train_dataloader, rank)
    #         test_loss = validate(enc_model, model, eval_test_dataloader, rank)
    #         print("--- step: {} ---".format(step))
    #         print("Train loss: {:.4f}".format(train_loss))
    #         print("Test loss: {:.4f}".format(test_loss))

    #         if wandb_log:
    #             wandb.log({
    #                 "train loss": train_loss,
    #                 "test loss": test_loss
    #             })

    #     if rank == 0 and step % save_step_frequency == 0:
    #         checkpoint_path = Path(checkpoints_dir, "step={}.pth".format(step))
    #         torch.save(model.module.state_dict(), checkpoint_path)
    #         print("Save model to {}".format(checkpoint_path))

    #         checkpoint_path = Path(checkpoints_dir, "latest.pth")
    #         torch.save(model.module.state_dict(), Path(checkpoint_path))
    #         print("Save model to {}".format(checkpoint_path))

    #         checkpoint_path = Path(checkpoints_dir, "step={}_encoder.pth".format(step))
    #         torch.save(enc_model.module.state_dict(), checkpoint_path)
    #         print("Save model to {}".format(checkpoint_path))

    #     if step == training_steps:
    #         break
    step = 0
    with tqdm(total=training_steps, desc="Training") as pbar:
        while step < training_steps:
            for data in train_dataloader:
                audio = data["audio"].to(device)
                input_token = data["token"][:, 0 : -1].to(device)
                target_token = data["token"][:, 1 :].to(device)
                target_mask = data["mask"][:, 1 :].to(device)

                optimizer.zero_grad()

                enc_model.train()
                model.train()
                audio_emb = enc_model.module.run_on_batch(audio)
                audio_emb = audio_emb.permute(0, 2, 3, 1).reshape(batch_size, 501, 4 * 88)

                logits, loss = model(audio_emb=audio_emb, idx=input_token, target=target_token, target_mask=target_mask)
                
                loss.backward()
                optimizer.step()

                if rank == 0 and step % evaluate_step_frequency == 0:
                    print("Evaluating ...")
                    train_loss = validate(enc_model, model, eval_train_dataloader, rank)
                    test_loss = validate(enc_model, model, eval_test_dataloader, rank)
                    print("--- step: {} ---".format(step))
                    print("Train loss: {:.4f}".format(train_loss))
                    print("Val loss: {:.4f}".format(test_loss))

                    if wandb_log:
                        wandb.log({
                            "train loss": train_loss,
                            "val loss": test_loss
                        })

                if rank == 0 and step % save_step_frequency == 0:
                    checkpoint_path = Path(checkpoints_dir, "step={}.pth".format(step))
                    torch.save(model.module.state_dict(), checkpoint_path)
                    print("Save model to {}".format(checkpoint_path))

                    checkpoint_path = Path(checkpoints_dir, "latest.pth")
                    torch.save(model.module.state_dict(), Path(checkpoint_path))
                    print("Save model to {}".format(checkpoint_path))

                    checkpoint_path = Path(checkpoints_dir, "step={}_encoder.pth".format(step))
                    torch.save(enc_model.module.state_dict(), checkpoint_path)
                    print("Save model to {}".format(checkpoint_path))

                step += 1
                pbar.update(1)

                if step >= training_steps:
                    break

def validate(enc_model, model, dataloader, rank): 

    device = torch.device(f"cuda:{rank}")
    losses = []
    
    len_dataset = len(dataloader.dataset)
    len_dataloader = len_dataset // dataloader.batch_size

    for step, data in tqdm(enumerate(dataloader), total=len_dataloader, desc="Validating"):
        if step == 5:
            break

        audio = data["audio"].to(device)
        input_token = data["token"][:, 0 : -1].to(device)
        target_token = data["token"][:, 1 :].to(device)
        target_mask = data["mask"][:, 1 :].to(device)

        with torch.no_grad():
            enc_model.eval()
            audio_emb = enc_model.module.run_on_batch(audio)
            # print(audio_emb.shape)
            audio_emb = audio_emb.permute(0, 2, 3, 1).reshape(8, 501, 4 * 88)

            model.eval()
            logits, loss = model(audio_emb=audio_emb, idx=input_token, target=target_token, target_mask=target_mask)

        losses.append(loss.item())

    return np.mean(losses)

def get_model(model_name):
    if model_name == "HPPNet":
        model_file = "/root/autodl-tmp/Improving-piano-transcription-by-LLM-based-decoder/hpp-10secondsInput-120000.pt"
        model = torch.load(model_file)
        return model
    else:
        raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    train(args)