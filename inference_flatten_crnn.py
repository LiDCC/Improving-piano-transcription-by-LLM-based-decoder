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
import matplotlib.pyplot as plt
# from train_llama_mt_on_hpp import get_model
from models.crnn import Note_pedal
import mir_eval
import re

from data.tokenizers import Tokenizer
from models.enc_dec import EncDecConfig, EncDecPos
from data.maestro import MaestroStringProcessor
from data.io import events_to_notes, notes_to_midi, read_single_track_midi, write_notes_to_midi, fix_length


def inference_in_batch(args):

    # Arguments
    # model_name = args.model_name
    filename = Path(__file__).stem

    # Default parameters
    segment_seconds = 10.
    device = "cuda"
    sample_rate = 16000
    top_k = 1
    batch_size = 1
    frames_num = 1001
    max_token_len = 2000
    segment_samples = int(segment_seconds * sample_rate)

    tokenizer = Tokenizer()

    # Load checkpoint
    enc_model_name = "CRNN"
    checkpoint_path = Path("/root/autodl-tmp/Improving-piano-transcription-by-LLM-based-decoder/checkpoints/train_llama_mt_flatten_crnn_ddp/AudioLlama/step=17625_encoder.pth")
    enc_model = Note_pedal()
    enc_model.load_state_dict(torch.load(checkpoint_path))
    enc_model.to(device)

    for param in enc_model.parameters():
        param.requires_grad = False

    # Load checkpoint
    checkpoint_path = Path("/root/autodl-tmp/Improving-piano-transcription-by-LLM-based-decoder/checkpoints/train_llama_mt_flatten_crnn_ddp/AudioLlama/step=17625.pth")
    config = EncDecConfig(
        block_size=max_token_len + 1, 
        vocab_size=tokenizer.vocab_size, 
        padded_vocab_size=tokenizer.vocab_size, 
        n_layer=6, 
        n_head=16,
        n_embd=1024, 
        audio_n_embd=1536
    )
    model = EncDecPos(config)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    # Data
    root = "/root/autodl-tmp/maestro-v3.0.0"
    meta_csv = Path(root, "maestro-v3.0.0.csv")
    meta_data = load_meta(meta_csv, split="test") 
    # meta_data = load_meta(meta_csv, split="train")
    audio_paths = [Path(root, name) for name in meta_data["audio_filename"]]
    midi_paths = [Path(root, name) for name in meta_data["midi_filename"]]

    onset_midis_dir = Path("pred_midis", filename)
    Path(onset_midis_dir).mkdir(parents=True, exist_ok=True)

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
            "task=flatten",
        ]
        tokens = tokenizer.strings_to_tokens(strings)
        tokens = np.repeat(np.array(tokens)[None, :], repeats=batch_size, axis=0)
        tokens = torch.LongTensor(tokens).to(device)

        all_notes = []
        sustain_notes = []

        while bgn < audio_samples:

            clip = audio[bgn : bgn + clip_samples]
            clip = librosa.util.fix_length(data=clip, size=clip_samples, axis=-1)

            segments = librosa.util.frame(clip, frame_length=segment_samples, hop_length=segment_samples).T

            bgn_sec = bgn / sample_rate
            print("Processing: {:.1f} s".format(bgn_sec))

            segments = torch.Tensor(segments).to(device)
            
            with torch.no_grad():
                enc_model.eval()
                audio_emb = enc_model(segments)["onoffvel_emb_h"]
            # 
            with torch.no_grad():
                model.eval()
                pred_tokens = model.generate_in_batch(
                    audio_emb=audio_emb, 
                    idx=tokens, 
                    max_new_tokens=2000,
                    end_token=tokenizer.stoi("<eos>")
                ).data.cpu().numpy()

                for k in range(pred_tokens.shape[0]):
                    for i, token in enumerate(pred_tokens[k]):
                        if token == tokenizer.stoi("<eos>"):
                            break                    

                    new_pred_tokens = pred_tokens[k, 1 : i + 1]
                    strings = tokenizer.tokens_to_strings(new_pred_tokens)
                    for i, s in enumerate(strings):
                        if "time=" in s:
                            strings[i] = "time={:.2f}".format(bgn_sec + float(re.search('time=(.*)', s).group(1)))
                    note_event_groups = [strings[i:i+4] for i in range(0, len(strings), 4)]
                    
                    for note_events in note_event_groups:
                        if note_events == ["<eos>"]:
                            break
                        reformulated_note_events = []
                        note_events_type = []
                        for item in note_events:
                            if "=" in item:
                                key = re.search('(.*)=', item).group(1)
                                value = re.search('{}=(.*)'.format(key), item).group(1)
                                value = format_value(key, value)
                                reformulated_note_events.append(value)
                                note_events_type.append(key)
                        # print(reformulated_note_events, note_events_type)
                        # if event type is time, pitch, time, velocity
                        if note_events_type == ["time", "pitch", "time", "velocity"]:
                            note = {
                                "start": reformulated_note_events[0],
                                "end": reformulated_note_events[2],
                                "pitch": reformulated_note_events[1],
                                "velocity": reformulated_note_events[3]
                            }
                            all_notes.append(note)
                        elif note_events_type == ["name", "pitch", "time", "velocity"]:
                            # fetch onset from sustain notes
                            for i, sustain_note in enumerate(sustain_notes):
                                if sustain_note["pitch"] == reformulated_note_events[1]:
                                    onset = sustain_note["start"]
                                    velocity = sustain_note["velocity"]
                                    # delete the sustain note
                                    del sustain_notes[i]
                                    break
                            if i == len(sustain_notes) - 1: # if no sustain note found
                                continue
                            note = {
                                "start": onset,
                                "end": reformulated_note_events[2],
                                "pitch": reformulated_note_events[1],
                                "velocity": velocity
                            }
                            all_notes.append(note)
                        elif note_events_type == ["time", "pitch", "name", "velocity"]:
                            sustain_note = {
                                "start": reformulated_note_events[0],
                                "end": 0, # waiting to be filled
                                "pitch": reformulated_note_events[1],
                                "velocity": reformulated_note_events[3]
                            }
                            sustain_notes.append(sustain_note)
                        elif note_events_type == ["name", "pitch", "name", "velocity"]:
                            # this is still continued throughout this segment
                            # do nothing!
                            continue
                        else:
                            # raise ValueError("Invalid note event type: {}".format(note_events_type))
                            print("Invalid note event type: {}".format(note_events_type))
                            continue
        
            bgn += clip_samples
        
        est_midi_path = Path(onset_midis_dir, "{}.mid".format(Path(audio_path).stem))
        # notes_to_midi(all_notes, str(est_midi_path))
        midi_data = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        for note in all_notes:
            # check if anything is out of range, if so print
            if note["pitch"] < 21 or note["pitch"] > 108:
                print("Pitch out of range: {}".format(note["pitch"]))
            if note["velocity"] < 0 or note["velocity"] > 127:
                print("Velocity out of range: {}".format(note["velocity"]))
            if note["start"] < 0 or note["end"] < 0:
                print("Time out of range: {}, {}".format(note["start"], note["end"]))
            if note["start"] > note["end"]:
                print("Start time is greater than end time: {}, {}".format(note["start"], note["end"]))
            note = pretty_midi.Note(
                start=float(note["start"]),
                end=float(note["end"]),
                pitch=int(note["pitch"]),
                velocity=int(note["velocity"])
            )
            instrument.notes.append(note)
        midi_data.instruments.append(instrument)
        midi_data.write(str(est_midi_path))
        print("Written to: {}".format(est_midi_path))
        
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
    
    print(strings)
    return None

    for w in strings:

        if "=" in w:
            key = re.search('(.*)=', w).group(1)
            value = re.search('{}=(.*)'.format(key), w).group(1)
            value = format_value(key, value)

            if key == "onset":
                if event is not None:
                    events.append(event)
                event = {}

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
        
    new_events.sort(key=lambda e: (e["onset"], e["name"], e["pitch"]))
    
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
