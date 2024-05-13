
# Improving piano transcription by LLM-based decoder

This repo currently support:
Convert piano music from audio to MIDI.

## Motivation
- “Frame-level Automatic music Transcription (AMT)”: complicated rule-based post-processing step
- “Seq-to-seq AMT”: predicts onset, offset, pitch, and velocity simultaneously, resulting in a lengthy output token sequence.

## Contribution
- Integration of frame-level AMT pretrained encoder with large language model (LLM) decoder in a seq-to-seq AMT framework.
- Hierarchical inference: predicting onset and pitch for all notes initially, then utilizing these predictions as input for the decoders to predict velocity and offset of each note, respectively.
- Our AMT models, each consisting of an encoder and a decoder, outperform their respective encoder-only versions of direct piano-roll predictions in all metrics.

## Commandline Usage
0. Install dependencies

```bash
git clone https://github.com/qiuqiangkong/mini_piano_transcription

# Install Python environment
conda create --name piano_transcription python=3.8

conda activate piano_transcription

# Install Python packages dependencies.
sh env.sh

```

# Train
CUDA_VISIBLE_DEVICES=0 python train.py

# Inference
CUDA_VISIBLE_DEVICES=0 python inference.py

# Evaluate
python evaluate.py
