import time
import librosa
import numpy as np
import pandas as pd
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
    # device = "cuda"
    sample_rate = 16000
    top_k = 1
    batch_size = 4
    max_token_len = 256
    segment_samples = int(segment_seconds * sample_rate)

    tokenizer = Tokenizer()

    # # read config file
    # with open("/datasets/maestro-v3.0.0/corpus/config.json", 'r', encoding='utf-8') as f:
    #     config = json.load(f)
    #
    # # AMT class
    # AMT = amt.AMT(config, None, None)

    # Load checkpoint
    enc_model_name = "hFT"
    checkpoint_path = Path("checkpoints/train_llama_mt_on_hft/AudioLlama_hft_embedding_B_freeze_finetune/step=314000_encoder.pth")
    enc_model = get_model(enc_model_name)
    enc_model.load_state_dict(torch.load(checkpoint_path))
    enc_model.to(device)

    for param in enc_model.parameters():
        param.requires_grad = False

    # Load checkpoint
    checkpoint_path = Path("checkpoints/train_llama_mt_on_hft/AudioLlama_hft_embedding_B_freeze_finetune/step=314000.pth")
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
    # meta_data = load_meta(meta_csv, split="train")
    audio_paths = [Path(root, name) for name in meta_data["audio_filename"]]
    midi_paths = [Path(root, name) for name in meta_data["midi_filename"]]

    onset_midis_dir = Path("pred_midis", filename + "AudioLlama_hft_embedding_B_314000")
    Path(onset_midis_dir).mkdir(parents=True, exist_ok=True)

    string_processor = MaestroStringProcessor(
        label=False,
        onset=True,
        offset=False,
        sustain=False,
        velocity=False,
        pedal_onset=False,
        pedal_offset=False,
        pedal_sustain=False,
    )

    precs = []
    recalls = []
    f1s = []

    # idx = tokenizer.stoi("<sos>")
    # idx = torch.LongTensor(idx * np.ones((batch_size, 1))).to(device)

    for audio_idx in range(len(audio_paths)):
    # for audio_idx in range(3, len(audio_paths)):

        print(audio_idx)
        t1 = time.time()

        audio_path = audio_paths[audio_idx]

        audio, _ = librosa.load(path=audio_path, sr=sample_rate, mono=True)
        # (channels_num, audio_samples)

        audio_samples = audio.shape[-1]
        bgn = 0
        segment_samples = int(segment_seconds * sample_rate)
        clip_samples = segment_samples * batch_size

        ##
        strings = [
            "<sos>",
            "task=onset",
        ]
        tokens = tokenizer.strings_to_tokens(strings)
        tokens = np.repeat(np.array(tokens)[None, :], repeats=batch_size, axis=0)
        tokens = torch.LongTensor(tokens).to(device)

        all_notes = []

        while bgn < audio_samples:

            clip = audio[bgn : bgn + clip_samples]
            clip = librosa.util.fix_length(data=clip, size=clip_samples, axis=-1)

            segments = librosa.util.frame(clip, frame_length=segment_samples, hop_length=segment_samples).T
            segments = np.concatenate((np.zeros([batch_size, 8192]), segments, np.zeros([batch_size, 8192])), axis=1)
            if not bgn == 0:
                segments[0, :8192] = audio[bgn-8192:bgn]
            if not bgn + clip_samples >= audio_samples:
                tmp_audio = audio[bgn+clip_samples: bgn+clip_samples+8192]
                tmp_audio = librosa.util.fix_length(data=tmp_audio, size=8192, axis=-1)
                segments[batch_size-1, -8192:] = tmp_audio
            for i in range(len(segments)):
                if not i == len(segments)-1:
                    segments[i, -8192:] = segments[i + 1, 8192:16384]
                if not i == 0:
                    segments[i, :8192] = segments[i-1, -16384:-8192]

            bgn_sec = bgn / sample_rate
            # print("Processing: {:.1f} s".format(bgn_sec))

            segments = torch.Tensor(segments).to(device)
            # mel = AMT.wav2feature(segments, device)[:, :, :-1]
            mel_spec = tr_mel(segments)
            mel = (torch.log(mel_spec + 1e-08))
            mel = mel[:, :, :-1]

            with torch.no_grad():
                enc_model.eval()
                output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, attention, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B, midi_time, output_velocity_A_max, output_velocity_B_max = enc_model(
                    mel)
                audio_emb = torch.cat((output_onset_B, output_mpe_B, output_offset_B, output_velocity_B_max), dim=-1)

                # output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, attention, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B, _ = enc_model(mel)
                # output_velocity_A_max = torch.argmax(output_velocity_A, dim=-1) / 128.0
                # audio_emb = torch.cat((output_onset_A, output_mpe_A, output_offset_A, output_velocity_A_max), dim=-1)
                # output_velocity_B_max = torch.argmax(output_velocity_B, dim=-1) / 128.0
                # audio_emb = torch.cat((output_onset_B, output_mpe_B, output_offset_B, output_velocity_B_max), dim=-1)
                # audio_emb = torch.cat((output_onset_B, output_mpe_B, output_offset_B), dim=-1)

            #
            with torch.no_grad():
                model.eval()
                pred_tokens = model.generate_in_batch(
                    audio_emb=audio_emb, 
                    idx=tokens, 
                    max_new_tokens=250,
                    end_token=tokenizer.stoi("<eos>")
                ).data.cpu().numpy()

                for k in range(pred_tokens.shape[0]):
                    for i, token in enumerate(pred_tokens[k]):
                        if token == tokenizer.stoi("<eos>"):
                            break                    

                    new_pred_tokens = pred_tokens[k, 1 : i + 1]
                    # from IPython import embed; embed(using=False); os._exit(0)
                    strings = tokenizer.tokens_to_strings(new_pred_tokens)
                    events = onset_strings_to_events(strings)
                    notes = events_to_notes(events)
                    
                    for note in notes:
                        note.start += bgn_sec + k * segment_seconds
                        note.end += bgn_sec + k * segment_seconds
                        if note.start > audio_samples / sample_rate:
                            notes.remove(note)
                            break

                    all_notes.extend(notes)

            bgn += clip_samples
            # from IPython import embed; embed(using=False); os._exit(0)

        # all_notes = list(set(all_notes))
        # notes_to_midi(all_notes, "_zz.mid")
        # soundfile.write(file="_zz.wav", data=audio, samplerate=16000)
        
        est_midi_path = Path(onset_midis_dir, "{}.mid".format(Path(audio_path).stem))
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
        # from IPython import embed; embed(using=False); os._exit(0)

    print("----------")
    print("Avg Prec: {:.4f}".format(np.mean(precs)))
    print("Avg Recall: {:.4f}".format(np.mean(recalls)))
    print("Avg F1: {:.4f}".format(np.mean(f1s)))


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

            if event is not None:
                event[key] = value

        if w == "<eos>" and event is not None:
            events.append(event)
            break

    new_events = []

    for e in events:

        if "time" in e.keys() and "pitch" in e.keys():
            e["name"] = "note_on"
            e["velocity"] = 100
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="AudioLlama")
    args = parser.parse_args()

    # inference(args)
    inference_in_batch(args)
