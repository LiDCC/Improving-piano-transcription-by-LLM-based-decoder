import torch
import time
import pickle
import librosa
import numpy as np
import pandas as pd
import soundfile
import pretty_midi
from pathlib import Path
import argparse
from train_llama_mt_on_hft import get_model
import mir_eval
import re
import torchaudio

from data.tokenizers import Tokenizer
from models.enc_dec import EncDecConfig, EncDecPos
from data.maestro import MaestroStringProcessor
from data.io import events_to_notes, notes_to_midi, read_single_track_midi, write_notes_to_midi, fix_length
from model.model_spec2midi import *

device = "cuda"
tr_mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000,
                                                          n_fft=2048,
                                                          win_length=2048,
                                                          hop_length=256,
                                                          pad_mode='constant',
                                                          n_mels=256, norm='slaney').to(device)

def inference_in_batch(args):

    # Arguments
    # model_name = args.model_name
    filename = Path(__file__).stem

    # Default parameters
    segment_seconds = 2.048
    sample_rate = 16000
    top_k = 1
    batch_size = 4
    frames_num = 1001
    max_token_len = 500
    segment_samples = int(segment_seconds * sample_rate)

    tokenizer = Tokenizer()

    # Load checkpoint
    enc_model_name = "hFT"
    checkpoint_path = Path("checkpoints/train_llama_mt_vel_hft/AudioLlama_hft_embedding_B_freeze_finetune+vel_model/step=90000_encoder.pth")
    enc_model = get_model(enc_model_name)
    enc_model.load_state_dict(torch.load(checkpoint_path))
    enc_model.to(device)

    for param in enc_model.parameters():
        param.requires_grad = False

    # Load checkpoint
    checkpoint_path = Path("checkpoints/train_llama_mt_vel_hft/AudioLlama_hft_embedding_B_freeze_finetune+vel_model/step=90000.pth")
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
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    # Data
    root = "/datasets/maestro-v3.0.0/maestro-v3.0.0"
    meta_csv = Path(root, "maestro-v3.0.0.csv")
    meta_data = load_meta(meta_csv, split="test")
    audio_paths = [Path(root, name) for name in meta_data["audio_filename"]]
    midi_paths = [Path(root, name) for name in meta_data["midi_filename"]]

    onset_midis_dir = Path("pred_midis", "inference_llama_mt_on_hftAudioLlama_hft_embedding_B_314000")
    est_midis_dir = Path("pred_midis_vel", filename)
    Path(est_midis_dir).mkdir(parents=True, exist_ok=True)

    string_processor = MaestroStringProcessor(
        label=False,
        onset=True,
        offset=True,
        sustain=False,
        velocity=True,
        pedal_onset=False,
        pedal_offset=False,
        pedal_sustain=False,
    )

    precs = []
    recalls = []
    f1s = []
    vel_precs = []
    vel_recalls = []
    vel_f1s = []

    for audio_idx in range(len(audio_paths)):
    # for audio_idx in range(44, len(audio_paths)):
        print(audio_idx)
        t1 = time.time()

        audio_path = audio_paths[audio_idx]

        # est_midi_path = Path(est_midis_dir, "{}.mid".format(Path(audio_path).stem))
        # if est_midi_path.exists():
        #     continue

        # from IPython import embed; embed(using=False); os._exit(0) 

        audio, _ = librosa.load(path=audio_path, sr=sample_rate, mono=True)
        # (channels_num, audio_samples)

        audio_samples = audio.shape[-1]
        bgn = 0
        
        # 
        onset_midi_path = Path(onset_midis_dir, "{}.mid".format(Path(audio_path).stem))
        onset_midi_data = pretty_midi.PrettyMIDI(str(onset_midi_path))
        pred_onset_notes = onset_midi_data.instruments[0].notes

        #
        all_notes = []

        while bgn < audio_samples:

            segment = audio[bgn : bgn + segment_samples]
            segment = librosa.util.fix_length(data=segment, size=segment_samples, axis=-1)

            segments = librosa.util.frame(segment, frame_length=segment_samples, hop_length=segment_samples).T

            segments = np.concatenate((np.zeros([1, 8192]), segments, np.zeros([1, 8192])), axis=1)
            if bgn > 0:
                segments[0, :8192] = audio[bgn-8192:bgn]
            if not bgn + segment_samples >= audio_samples:
                tmp_audio = audio[bgn+segment_samples: bgn+segment_samples+8192]
                tmp_audio = librosa.util.fix_length(data=tmp_audio, size=8192, axis=-1)
                segments[0, -8192:] = tmp_audio

            bgn_sec = bgn / sample_rate
            print("Processing: {:.1f} s".format(bgn_sec))

            segments = torch.Tensor(segments).to(device)
            mel_spec = tr_mel(segments)
            mel = (torch.log(mel_spec + 1e-08))
            mel = mel[:, :, :-1]

            with torch.no_grad():
                enc_model.eval()
                output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, attention, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B, midi_time, output_velocity_A_max, output_velocity_B_max = enc_model(mel)
                audio_emb = torch.cat((output_onset_B, output_mpe_B, output_offset_B, output_velocity_B_max), dim=-1)

            #
            bgn_sec = bgn / sample_rate
            end_sec = bgn_sec + segment_seconds
            candidate_notes = []
            for note in pred_onset_notes:
                if bgn_sec <= note.start < end_sec:
                    candidate_notes.append(note)

            strings = [
                "<sos>",
                "task=velocity",
            ]
            tokens = tokenizer.strings_to_tokens(strings)
            
            # 
            for note in candidate_notes:
                token = tokenizer.stoi("time={:.2f}".format(note.start - bgn_sec))
                tokens.append(token)
                token = tokenizer.stoi("pitch={}".format(note.pitch))
                tokens.append(token)
                tokens = np.array(tokens)[None, :]
                tokens = torch.LongTensor(tokens).to(device)
            
                # 
                with torch.no_grad():
                    model.eval()
                    pred_tokens = model.generate_in_batch(
                        audio_emb=audio_emb, 
                        idx=tokens,
                        max_new_tokens=1,
                        end_token=tokenizer.stoi("<eos>")
                    ).data.cpu().numpy()
                    pred_token = pred_tokens[0][-1]
                
                tokens = tokens[0].tolist() + [pred_token]

                # append new notes
                string = tokenizer.itos(pred_token)
                if 'velocity' in string:
                    vel = int(re.search('velocity=(.*)', string).group(1))
                else:
                    vel = 100
                note.velocity = vel
                all_notes.append(note) 
                
            bgn += segment_samples
            
        notes_to_midi(all_notes, "_zz.mid")
        # soundfile.write(file="_zz.wav", data=audio, samplerate=16000)
        
        est_midi_path = Path(est_midis_dir, "{}.mid".format(Path(audio_path).stem))
        notes_to_midi(all_notes, str(est_midi_path))
        
        ref_midi_path = midi_paths[audio_idx]
        ref_intervals, ref_pitches, ref_vels = parse_midi(ref_midi_path)
        est_intervals, est_pitches, est_vels = parse_midi(est_midi_path) 

        note_precision, note_recall, note_f1, _ = \
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals=ref_intervals, 
            ref_pitches=ref_pitches, 
            est_intervals=est_intervals, 
            est_pitches=est_pitches, 
            onset_tolerance=0.05, 
            offset_ratio=None,)

        print("P: {:.4f}, R: {:.4f}, F1: {:.4f}, time: {:.4f} s".format(note_precision, note_recall, note_f1, time.time() - t1))
        precs.append(note_precision)
        recalls.append(note_recall)
        f1s.append(note_f1)
        
        # eval with vel
        note_precision, note_recall, note_f1, _ = \
           mir_eval.transcription_velocity.precision_recall_f1_overlap(
               ref_intervals=ref_intervals,
               ref_pitches=ref_pitches,
               ref_velocities=ref_vels,
               est_intervals=est_intervals,
               est_pitches=est_pitches,
               est_velocities=est_vels,
               offset_ratio=None,
               )

        print("        P: {:.4f}, R: {:.4f}, F1: {:.4f}, time: {:.4f} s".format(note_precision, note_recall, note_f1, time.time() - t1))
        vel_precs.append(note_precision)
        vel_recalls.append(note_recall)
        vel_f1s.append(note_f1)

    print("--- Onset -------")
    print("Avg Prec: {:.4f}".format(np.mean(precs)))
    print("Avg Recall: {:.4f}".format(np.mean(recalls)))
    print("Avg F1: {:.4f}".format(np.mean(f1s)))
    print("--- Onset + Vel -------")
    print("Avg Prec: {:.4f}".format(np.mean(vel_precs)))
    print("Avg Recall: {:.4f}".format(np.mean(vel_recalls)))
    print("Avg F1: {:.4f}".format(np.mean(vel_f1s)))


def load_meta(meta_csv, split):

    df = pd.read_csv(meta_csv, sep=',')

    indexes = df["split"].values == split

    midi_filenames = df["midi_filename"].values[indexes]
    audio_filenames = df["audio_filename"].values[indexes]

    meta_data = {
        "midi_filename": midi_filenames,
        "audio_filename": audio_filenames
    }

    return meta_data


def parse_midi(midi_path):

    midi_data = pretty_midi.PrettyMIDI(str(midi_path))

    notes = midi_data.instruments[0].notes

    intervals = []
    pitches = []
    velocities = []

    for note in notes:
        intervals.append([note.start, note.end])
        pitches.append(note.pitch)
        velocities.append(note.velocity)

    return np.array(intervals), np.array(pitches), np.array(velocities)


def deduplicate_array(array):

    new_array = []

    for pair in array:
        time = pair[0]
        pitch = pair[1]
        if (time - 1, pitch) not in new_array:
            new_array.append((time, pitch))

    return np.array(new_array)


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="AudioLlama")
    args = parser.parse_args()

    # inference(args)
    inference_in_batch(args)
