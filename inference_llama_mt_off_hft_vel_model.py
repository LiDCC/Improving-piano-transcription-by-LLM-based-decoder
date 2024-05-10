import torch
import time
import pickle
import librosa
import numpy as np
import pandas as pd
import soundfile
import pretty_midi
from pathlib import Path
import torch.optim as optim
from data.collate import collate_fn
from models.crnn import CRnn
from tqdm import tqdm
import museval
import argparse
import matplotlib.pyplot as plt
from train_llama_mt_off_hft import get_model
import mir_eval
import re
import os
import torchaudio
# from models_bd.models import Note_pedal, Regress_onset_offset_frame_velocity_CRNN

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
    max_token_len = 1000
    segment_samples = int(segment_seconds * sample_rate)

    tokenizer = Tokenizer()

    # Load checkpoint
    enc_model_name = "hFT"
    checkpoint_path = Path("checkpoints/train_llama_mt_off_hft/AudioLlama_hft_embedding_B_freeze_finetune+vel_model/step=110000_encoder.pth")
    enc_model = get_model(enc_model_name)
    enc_model.load_state_dict(torch.load(checkpoint_path))
    enc_model.to(device)

    for param in enc_model.parameters():
        param.requires_grad = False

    vel_model = Vel_linear()
    checkpoint_path = Path("checkpoints/train_llama_mt_off_hft/AudioLlama_hft_embedding_B_freeze_finetune+vel_model/step=110000_vel.pth")
    vel_model.load_state_dict(torch.load(checkpoint_path))
    vel_model.to(device)

    # Load checkpoint
    checkpoint_path = Path("checkpoints/train_llama_mt_off_hft/AudioLlama_hft_embedding_B_freeze_finetune+vel_model/step=110000.pth")
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

    onset_midis_dir = Path("pred_midis_vel", "inference_llama_mt_vel_hft_vel_mocel")
    est_midis_dir = Path("pred_midis_off(vel_model)", filename)
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
    off_precs = []
    off_recalls = []
    off_f1s = []
    off_vel_precs = []
    off_vel_recalls = []
    off_vel_f1s = []

    for audio_idx in range(len(audio_paths)):
    # for audio_idx in range(44, len(audio_paths)):

        print(audio_idx)
        t1 = time.time()

        audio_path = audio_paths[audio_idx]

        # est_midi_path = Path(est_midis_dir, "{}.mid".format(Path(audio_path).stem))
        #
        # if not est_midi_path.exists():

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
        # seg_notes = []
        all_notes = []
        sustain_notes = []
        still_sustain_notes = []

        while bgn < audio_samples:

            segment = audio[bgn : bgn + segment_samples]
            segment = librosa.util.fix_length(data=segment, size=segment_samples, axis=-1)

            segments = librosa.util.frame(segment, frame_length=segment_samples, hop_length=segment_samples).T

            segments = np.concatenate((np.zeros([1, 8192]), segments, np.zeros([1, 8192])), axis=1)

            if bgn > 0:
                segments[0, :8192] = audio[bgn - 8192:bgn]
            if not bgn + segment_samples >= audio_samples:
                tmp_audio = audio[bgn + segment_samples: bgn + segment_samples + 8192]
                tmp_audio = librosa.util.fix_length(data=tmp_audio, size=8192, axis=-1)
                segments[0, -8192:] = tmp_audio

            bgn_sec = bgn / sample_rate
            end_sec = bgn_sec + segment_seconds
            # print("Processing: {:.1f} s".format(bgn_sec))

            segments = torch.Tensor(segments).to(device)
            mel_spec = tr_mel(segments)
            mel = (torch.log(mel_spec + 1e-08))
            mel = mel[:, :, :-1]

            with torch.no_grad():
                enc_model.eval()
                vel_model.eval()
                output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, attention, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B, midi_time, output_velocity_A_max, output_velocity_B_max = enc_model(mel)
                vel_out_B = vel_model(output_velocity_B)
                audio_emb = torch.cat((output_onset_B, output_mpe_B, output_offset_B, vel_out_B), dim=-1)
                # output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, attention, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B, _, _, _ = enc_model(
                #     mel)
                # output_velocity_A_max = torch.argmax(output_velocity_A, dim=-1) / 128.0
                # audio_emb = torch.cat((output_onset_A, output_mpe_A, output_offset_A, output_velocity_A_max),
                #                       dim=-1)

            #
            strings = [
                "<sos>",
                "task=offset",
            ]
            tokens = tokenizer.strings_to_tokens(strings)

            # Process sustain notes
            sustain_notes = still_sustain_notes + sustain_notes
            still_sustain_notes = []
            for note in sustain_notes:
                token = tokenizer.stoi("name=note_sustain")
                tokens.append(token)
                token = tokenizer.stoi("pitch={}".format(note.pitch))
                tokens.append(token)
                tokens = np.array(tokens)[None, :]
                tokens = torch.LongTensor(tokens).to(device)

                with torch.no_grad():
                    model.eval()
                    pred_tokens = model.generate_in_batch(
                        audio_emb=audio_emb,
                        idx=tokens,
                        max_new_tokens=1,
                        end_token=tokenizer.stoi("<eos>")
                    ).data.cpu().numpy()
                    pred_token = pred_tokens[0][-1]

                string = tokenizer.itos(pred_token)
                if "time" in string:
                    offset_time = float(re.search('time=(.*)', string).group(1))
                    note.end = bgn_sec + offset_time
                    all_notes.append(note)

                elif "name=note_sustain" in string:
                    still_sustain_notes.append(note)
                    # pass
                    # from IPython import embed; embed(using=False); os._exit(0)

                tokens = tokens[0].tolist() + [pred_token]

            sustain_notes = []


            # Process notes
            strings = [
                "<sos>",
                "task=offset",
            ]
            tokens = tokenizer.strings_to_tokens(strings)

            candidate_notes = []
            for note in pred_onset_notes:
                if bgn_sec <= note.start < end_sec:
                    candidate_notes.append(note)

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

                string = tokenizer.itos(pred_token)
                if "time" in string:
                    offset_time = float(re.search('time=(.*)', string).group(1))
                    note.end = bgn_sec + offset_time
                    all_notes.append(note)
                elif "name=note_sustain" in string:
                    sustain_notes.append(note)

                tokens = tokens[0].tolist() + [pred_token]

            # if bgn_sec == 20:
            #     from IPython import embed; embed(using=False); os._exit(0)


            # tokens.append(tokenizer.stoi("<eos>"))
            # tokens = tokens[2:]
            # strings = tokenizer.tokens_to_strings(tokens)

            bgn += segment_samples

        all_notes.sort(key=lambda note: (note.start, note.pitch))

        # notes_to_midi(all_notes, "_zz.mid")
        # soundfile.write(file="_zz.wav", data=audio, samplerate=16000)
        # from IPython import embed; embed(using=False); os._exit(0)

        est_midi_path = Path(est_midis_dir, "{}.mid".format(Path(audio_path).stem))
        notes_to_midi(all_notes, str(est_midi_path))

        # Load with pedals GT
        ref_midi_path = midi_paths[audio_idx]
        notes, _ = read_single_track_midi(ref_midi_path, extend_pedal=True)
        write_notes_to_midi(notes, "_zz_gt.mid")

        ref_intervals, ref_pitches, ref_vels = parse_midi("_zz_gt.mid")
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

        # eval with offset
        note_precision, note_recall, note_f1, _ = \
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals=ref_intervals, 
            ref_pitches=ref_pitches, 
            est_intervals=est_intervals, 
            est_pitches=est_pitches, 
            onset_tolerance=0.05, 
            offset_ratio=0.2,

        )
        print("    P: {:.4f}, R: {:.4f}, F1: {:.4f}, time: {:.4f} s".format(note_precision, note_recall, note_f1, time.time() - t1))
        off_precs.append(note_precision)
        off_recalls.append(note_recall)
        off_f1s.append(note_f1)
        
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

        # eval with vel
        note_precision, note_recall, note_f1, _ = \
           mir_eval.transcription_velocity.precision_recall_f1_overlap(
               ref_intervals=ref_intervals,
               ref_pitches=ref_pitches,
               ref_velocities=ref_vels,
               est_intervals=est_intervals,
               est_pitches=est_pitches,
               est_velocities=est_vels,
               offset_ratio=0.2,
               )

        print("        P: {:.4f}, R: {:.4f}, F1: {:.4f}, time: {:.4f} s".format(note_precision, note_recall, note_f1, time.time() - t1))
        off_vel_precs.append(note_precision)
        off_vel_recalls.append(note_recall)
        off_vel_f1s.append(note_f1)

    print("--- Onset -------")
    print("Avg Prec: {:.4f}".format(np.mean(precs)))
    print("Avg Recall: {:.4f}".format(np.mean(recalls)))
    print("Avg F1: {:.4f}".format(np.mean(f1s)))
    print("--- Onset + Vel -------")
    print("Avg Prec: {:.4f}".format(np.mean(vel_precs)))
    print("Avg Recall: {:.4f}".format(np.mean(vel_recalls)))
    print("Avg F1: {:.4f}".format(np.mean(vel_f1s)))
    print("--- Onset + Off -------")
    print("Avg Prec: {:.4f}".format(np.mean(off_precs)))
    print("Avg Recall: {:.4f}".format(np.mean(off_recalls)))
    print("Avg F1: {:.4f}".format(np.mean(off_f1s)))
    print("--- Onset + Off + Vel -------")
    print("Avg Prec: {:.4f}".format(np.mean(off_vel_precs)))
    print("Avg Recall: {:.4f}".format(np.mean(off_vel_recalls)))
    print("Avg F1: {:.4f}".format(np.mean(off_vel_f1s)))


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


def onset_strings_to_events(strings):

    event = None
    events = []

    for w in strings:

        if "=" in w:
            key = re.search('(.*)=', w).group(1)
            value = re.search('{}=(.*)'.format(key), w).group(1)
            value = format_value(key, value)

            if key == "time":
                if event is not None:
                    events.append(event)
                event = {}

            event[key] = value

        if w == "<eos>" and event is not None:
            events.append(event)
            break

    new_events = []

    for e in events:

        if "time" in e.keys() and "pitch" in e.keys() and "velocity" in e.keys():
            e["name"] = "note_on"
            new_events.append(e)

            event = {
                "name": "note_off",
                "time": float(e["time"]) + 0.1,
                "pitch": e["pitch"]
            }
            new_events.append(event)
        
    new_events.sort(key=lambda e: (e["time"], e["name"], e["pitch"]))
    
    return new_events


def format_value(key, value): 
    if key in ["time"]:
        return float(value)

    elif key in ["pitch", "velocity"]:
        return int(value)

    else:
        return value



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="AudioLlama")
    args = parser.parse_args()

    # inference(args)
    inference_in_batch(args)
