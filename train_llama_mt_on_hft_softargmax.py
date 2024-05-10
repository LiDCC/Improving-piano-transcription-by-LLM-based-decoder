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
from data.maestro import MaestroMultiTask_hft
from data.collate import collate_fn
from data.io import events_to_notes
from models.crnn import CRnn
from tqdm import tqdm
import museval
import argparse
import wandb
import os
import pickle
import json
import torchaudio

from torch.optim.lr_scheduler import StepLR
from data.tokenizers import Tokenizer
from models.enc_dec import EncDecConfig, EncDecPos
from model.model_spec2midi import *

device = "cuda"
tr_mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000,
                                                          n_fft=2048,
                                                          win_length=2048,
                                                          hop_length=256,
                                                          pad_mode='constant',
                                                          n_mels=256, norm='slaney').to(device)

def train(args):
    # Arguments
    # model_name = args.model_name

    # Default parameters
    # batch_size = 16
    batch_size = 4
    # num_workers = 0
    num_workers = 32
    evaluate_step_frequency = 10000
    save_step_frequency = 10000
    training_steps = 1000000
    debug = False
    filename = Path(__file__).stem
    segment_seconds = 2.048
    lr = 1e-4
    frames_num = 1001
    max_token_len = 256
    wandb_log = True

    model_name = "AudioLlama_hft_embedding_B_freeze_+soft"
    checkpoints_dir = Path("./checkpoints", filename, model_name)

    # root = "/datasets/maestro-v2.0.0/maestro-v2.0.0"
    root = "/datasets/maestro-v3.0.0/maestro-v3.0.0"

    if wandb_log:
        wandb.init(
            project="mini_piano_transcription",
            name=filename + model_name
        )

    tokenizer = Tokenizer()

    # Dataset
    train_dataset = MaestroMultiTask_hft(
        root=root,
        split="train",
        segment_seconds=segment_seconds,
        tokenizer=tokenizer,
        max_token_len=max_token_len,
        task="onset"
    )

    test_dataset = MaestroMultiTask_hft(
        root=root,
        split="test",
        segment_seconds=segment_seconds,
        tokenizer=tokenizer,
        max_token_len=max_token_len,
        task="onset"
    )

    # Sampler
    train_sampler = Sampler(dataset_size=len(train_dataset))
    eval_train_sampler = Sampler(dataset_size=len(train_dataset))
    eval_test_sampler = Sampler(dataset_size=len(test_dataset))

    # Dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    eval_train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=eval_train_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    eval_test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        sampler=eval_test_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    # # read config file
    # with open("/datasets/maestro-v3.0.0/corpus/config.json", 'r', encoding='utf-8') as f:
    #     config = json.load(f)
    #
    # # AMT class
    # AMT = amt.AMT(config, None, None)

    # adsf
    # Load checkpoint
    enc_model_name = "hFT"
    # checkpoint_path = Path("checkpoints/train_llama_mt_on_hft/AudioLlama_hft_embedding_B_freeze/step=200000_encoder.pth")
    enc_model = get_model(enc_model_name)
    # enc_model.load_state_dict(torch.load(checkpoint_path))
    enc_model.to(device)

    for param in enc_model.parameters():
        param.requires_grad = False

    # vel_model = Vel_linear().to(device)

    config = EncDecConfig(
        block_size=max_token_len + 1,
        vocab_size=tokenizer.vocab_size,
        padded_vocab_size=tokenizer.vocab_size,
        n_layer=6,
        n_head=16,
        n_embd=1024,
        audio_n_embd=352
    )

    model = EncDecPos(config)
    # checkpoint_path = Path("checkpoints/train_llama_mt_on_hft/AudioLlama_hft_embedding_B_freeze/step=200000.pth")
    # model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    # Optimizer
    optimizer = optim.AdamW(list(model.parameters()), lr=lr)
    # optimizer = optim.AdamW(list(enc_model.parameters()) + list(vel_model.parameters()) + list(model.parameters()), lr=lr)
    # optimizer = optim.AdamW(list(vel_model.parameters()) + list(model.parameters()), lr=lr)
    # scheduler = StepLR(optimizer, step_size=10000, gamma=0.98)

    # Create checkpoints directory
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    tmp = []

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        audio = data["audio"].to(device)
        input_token = data["token"][:, 0: -1].to(device)
        target_token = data["token"][:, 1:].to(device)
        target_mask = data["mask"][:, 1:].to(device)

        # Play the audio.
        if debug:
            play_audio(mixture, target)

        optimizer.zero_grad()

        enc_model.train()
        # vel_model.train()
        model.train()
        #注意
        # mel = AMT.wav2feature(audio, device)[:, :, :-1]
        mel_spec = tr_mel(audio)
        mel = (torch.log(mel_spec + 1e-08))
        mel = mel[:, :, :-1]
        # print(mel.shape) #torch.Size([251, 256, 16])
        output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, attention, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B, midi_time, output_velocity_A_soft, output_velocity_B_soft = enc_model(mel)
        # vel_out_B = vel_model(output_velocity_B)
        audio_emb = torch.cat((output_onset_B, output_mpe_B, output_offset_B, output_velocity_B_soft), dim=-1)
        # print(audio_emb.shape)

        logits, loss = model(audio_emb=audio_emb, idx=input_token, target=target_token, target_mask=target_mask)

        loss.backward()

        optimizer.step()
        # scheduler.step()

        if step % evaluate_step_frequency == 0:
            print("Evaluating ...")
            train_loss = validate(enc_model, model, eval_train_dataloader)
            test_loss = validate(enc_model, model, eval_test_dataloader)
            print("--- step: {} ---".format(step))
            print("Train loss: {:.4f}".format(train_loss))
            print("Test loss: {:.4f}".format(test_loss))

            if wandb_log:
                wandb.log({
                    "train loss": train_loss,
                    "test loss": test_loss
                })

        # Save model
        if step % save_step_frequency == 0:
            checkpoint_path = Path(checkpoints_dir, "step={}.pth".format(step))
            torch.save(model.state_dict(), checkpoint_path)
            print("Save model to {}".format(checkpoint_path))

            # checkpoint_path = Path(checkpoints_dir, "latest.pth")
            # torch.save(model.state_dict(), Path(checkpoint_path))
            # print("Save model to {}".format(checkpoint_path))

            #
            checkpoint_path = Path(checkpoints_dir, "step={}_encoder.pth".format(step))
            torch.save(enc_model.state_dict(), checkpoint_path)
            print("Save model to {}".format(checkpoint_path))

            # checkpoint_path = Path(checkpoints_dir, "step={}_vel.pth".format(step))
            # torch.save(vel_model.state_dict(), checkpoint_path)
            # print("Save model to {}".format(checkpoint_path))

        # tmp.extend(data["answer_tokens_num"])
        # from IPython import embed; embed(using=False); os._exit(0)

        if step == training_steps:
            break

        # if step == 1000:
        #     from IPython import embed; embed(using=False); os._exit(0)
        #     hist, bin_edges = np.histogram(tmp)


def get_model(model_name):
    if model_name == "CRnn":
        return CRnn()
    elif model_name == "CRnn2":
        from models.crnn2 import CRnn2
        return CRnn2()
    elif model_name == "CRnn3":
        from models.crnn3 import CRnn3
        return CRnn3()
    elif model_name == "CRnn3_onset_offset_vel":
        from models.crnn3_onset_offset_vel import CRnn3_onset_offset_vel
        return CRnn3_onset_offset_vel()
    # elif model_name == "Note_pedal":
    #     from models_bd.models import Note_pedal
    #     return Note_pedal()
    elif model_name == "AudioLlamaQA":
        from models.audiollama_qa import AudioLlamaQA
    elif model_name == "HPPNet":
        model_file = "/home/dylan/Projects/HPPNet/runs/hppnet_10s/model-120000.pt"
        model = torch.load(model_file)
        return model
    elif model_name == "hFT":
        model_path = "/home/dylan/Projects/hFT-Transformer/checkpoint/MAESTRO-V3/model_016_003.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        raise NotImplementedError


class Sampler:
    def __init__(self, dataset_size):
        self.indexes = list(range(dataset_size))
        random.shuffle(self.indexes)

    def __iter__(self):

        pointer = 0

        while True:

            if pointer == len(self.indexes):
                random.shuffle(self.indexes)
                pointer = 0

            # print(pointer)
            index = self.indexes[pointer]
            pointer += 1

            yield index


def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)


def play_audio(mixture, target):
    soundfile.write(file="tmp_mixture.wav", data=mixture[0].cpu().numpy().T, samplerate=44100)
    soundfile.write(file="tmp_target.wav", data=target[0].cpu().numpy().T, samplerate=44100)
    from IPython import embed;
    embed(using=False);
    os._exit(0)


def validate(enc_model, model, dataloader):
    pred_ids = []
    target_ids = []
    device = next(model.parameters()).device
    losses = []

    for step, data in tqdm(enumerate(dataloader)):

        if step == 5:
            break

        audio = data["audio"].to(device)
        input_token = data["token"][:, 0: -1].to(device)
        target_token = data["token"][:, 1:].to(device)
        target_mask = data["mask"][:, 1:].to(device)
        mel_spec = tr_mel(audio)
        mel = (torch.log(mel_spec + 1e-08))
        mel = mel[:, :, :-1]

        with torch.no_grad():
            enc_model.eval()
            # vel_model.eval()
            output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, attention, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B, midi_time, output_velocity_A_max, output_velocity_B_max = enc_model(mel)
            # vel_out_B = vel_model(output_velocity_B)
            audio_emb = torch.cat((output_onset_B, output_mpe_B, output_offset_B, output_velocity_B_max), dim=-1)
            # output_velocity_A_max = torch.argmax(output_velocity_A, dim=-1) / 128.0
            # audio_emb = torch.cat((output_onset_A, output_mpe_A, output_offset_A, output_velocity_A_max), dim=-1)

        with torch.no_grad():
            model.eval()
            logits, loss = model(audio_emb=audio_emb, idx=input_token, target=target_token, target_mask=target_mask)

        losses.append(loss.item())

    return np.mean(losses)


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', type=str, default="AudioLlama")
    args = parser.parse_args()

    train(args)