import os
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse

from data.maestro import MaestroMultiTask
from data.collate import collate_fn
from data.tokenizers import Tokenizer
from models.crnn import Note_pedal
from models.enc_dec import EncDecConfig, EncDecPos

def get_model(model_name):
    if model_name == "CRnn":
        return Note_pedal()
    # Add other model types if needed
    else:
        raise NotImplementedError

def validate(enc_model, model, dataloader, device):
    losses = []
    
    for data in tqdm(dataloader, desc="Validating"):
        audio = data["audio"].to(device)
        input_token = data["token"][:, 0 : -1].to(device)
        target_token = data["token"][:, 1 :].to(device)
        target_mask = data["mask"][:, 1 :].to(device)

        with torch.no_grad():
            enc_model.eval()
            audio_emb = enc_model(audio)["onoffvel_emb_h"]

            model.eval()
            _, loss = model(audio_emb=audio_emb, idx=input_token, target=target_token, target_mask=target_mask)

        losses.append(loss.item())

    return np.mean(losses)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    tokenizer = Tokenizer()
    val_dataset = MaestroMultiTask(
        root=args.data_root,
        split="validation",
        segment_seconds=10.,
        tokenizer=tokenizer,
        max_token_len=1536,
        task="flatten"
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )

    # Load encoder model
    enc_model = get_model("CRnn")
    enc_model.to(device)

    # Initialize EncDecPos model
    config = EncDecConfig(
        block_size=1537,  # max_token_len + 1
        vocab_size=tokenizer.vocab_size,
        padded_vocab_size=tokenizer.vocab_size,
        n_layer=6,
        n_head=16,
        n_embd=1024,
        audio_n_embd=1536
    )
    model = EncDecPos(config)
    model.to(device)

    checkpoints_dir = Path(args.checkpoints_dir)
    best_loss = float('inf')
    best_step = None
    
    checkpoint_encoder_files = sorted(checkpoints_dir.glob("step=*_encoder.pth"))
    # sort by step
    checkpoint_encoder_files = sorted(checkpoint_encoder_files, key=lambda x: int(x.stem.split('=')[1].split('_')[0]))

    for checkpoint_file in checkpoint_encoder_files:
        step = int(checkpoint_file.stem.split('=')[1].split('_')[0])
        checkpoint = checkpoints_dir / f"step={step}.pth"
        encoder_checkpoint = checkpoints_dir / f"step={step}_encoder.pth"

        if not encoder_checkpoint.exists():
            print(f"Skipping step {step}: encoder checkpoint not found")
            continue

        print(f"Evaluating checkpoint: {checkpoint_file}")

        # Load encoder
        enc_model.load_state_dict(torch.load(encoder_checkpoint, map_location=device))

        # Load model
        model.load_state_dict(torch.load(checkpoint, map_location=device))

        # Validate
        loss = validate(enc_model, model, val_dataloader, device)

        print(f"Step {step} - Validation Loss: {loss:.4f}")
        with open("validation_loss.txt", "a") as f:
            f.write(f"{step},{loss}\n")

        if loss < best_loss:
            best_loss = loss
            best_step = step

    print(f"\nBest checkpoint: step={best_step} with validation loss: {best_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/maestro-v3.0.0", help="Path to the dataset root")
    parser.add_argument("--checkpoints_dir", type=str, default="/root/autodl-tmp/Improving-piano-transcription-by-LLM-based-decoder/checkpoints/train_llama_mt_flatten_crnn_ddp/AudioLlama", help="Directory containing checkpoints")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for validation")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    args = parser.parse_args()

    main(args)