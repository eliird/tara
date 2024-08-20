import torch
from tara.model import TextAudioModel
from tara.constants import EMOTION2LABEL, LABEL2EMOTION
from transformers import (
    AutoTokenizer,
    Wav2Vec2Processor
)
import numpy as np
from typing import Dict

class EmotionRecognition:
    def __init__(self, weights_path: str, device: str = 'cuda') -> None:
        self.text_model_name = "distilbert-base-uncased"
        self.audio_model_name = "facebook/wav2vec2-base-960h"
        self.num_labesl = 7
        self.device = device
        self.samples = 48000
        
        self.text_processor = AutoTokenizer.from_pretrained(self.text_model_name)
        self.audio_processor = Wav2Vec2Processor.from_pretrained(self.audio_model_name)
        
        self.model = TextAudioModel(
            self.audio_model_name,
            self.text_model_name,
            self.num_labesl
        ) 
        
        self.model.load_state_dict(torch.load(weights_path))
        self.model.to(self.device)
        
    
    def detect(self, audio:np.ndarray, text: str)-> Dict:
        if len(audio.shape) > 1:
            # use 1st channel in case of two channels
            audio = audio[0]
        audio = self.pad_or_truncate(audio)
        audio = self.audio_processor(audio, sampling_rate=16000, return_tensors='pt').input_values.to(self.device)
        encoded_dict = self.text_processor.encode_plus(text, return_tensors='pt')
        input_ids = encoded_dict['input_ids'].to(self.device)
        att_mask = encoded_dict['attention_mask'].to(self.device)
        out = torch.softmax(self.model(audio, input_ids, att_mask)["logits"][0], dim=0).detach().cpu().numpy()        
        out_map = {LABEL2EMOTION[i]: out[i] for i in range(len(out))}       
        return out_map
        
             
    def pad_or_truncate(self, arr):
        if len(arr) > self.samples:
            arr = arr[:self.samples]
        else:
            arr = np.pad(arr, (0, self.samples - len(arr)), 'constant', constant_values=(0))
        return arr
        