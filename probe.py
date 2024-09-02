import torch
from torch import nn

# Load the model
model = torch.load("/home/yongyizang/Improving-piano-transcription-by-LLM-based-decoder/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth")['model']

note_model = model['note_model']
pedal_model = model['pedal_model']

for key, value in note_model.items():
    print(key, value.shape)
    
for key, value in pedal_model.items():
    print(key, value.shape)