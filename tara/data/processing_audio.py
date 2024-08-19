from tara.data import load_data
from torch.utils.data import Dataset, DataLoader
import librosa as lb
from typing import Any
import torch
from .processing import tensor_data_to_dataloader
import numpy as np
import warnings

class AudioDataset(Dataset):
    def __init__(self, df, audio_processor, sr=16_000, duration: float=3):
        self.df = df
        self.processor = audio_processor
        self.duration = duration
        self.sampling_rate = sr
        self.samples = self.duration * self.sampling_rate
    
    def __getitem__(self, index) -> Any:
        file_path = self.df.iloc[index]['filename']
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio_input ,sr = lb.load(file_path, sr=self.sampling_rate)
        audio_input = self.pad_or_truncate(audio_input)
        input_values = self.processor(audio_input, sampling_rate=sr, return_tensors='pt').input_values
        labels = torch.tensor(self.df.iloc[index]['labels'])
        
        return (input_values.squeeze(), labels)
    
    def __len__(self):
        return len(self.df)

    def pad_or_truncate(self, arr):
        if len(arr) > self.samples:
            arr = arr[:self.samples]
        else:
            arr = np.pad(arr, (0, self.samples - len(arr)), 'constant', constant_values=(0))
        return arr


def load_audio_dataloader(base_path: str, processor, 
                          sampling_rate: int=16000, duration: float=3, 
                          batch_size: int=16):
    train, test, val = load_data(base_path)
    train = tensor_data_to_dataloader(
        AudioDataset(
            train, 
            processor, sr=sampling_rate, duration=duration), 
            batch_size=batch_size
        )
    test = tensor_data_to_dataloader(
        AudioDataset(test, processor, sr=sampling_rate, duration=duration),
        batch_size=batch_size
        )
    val = tensor_data_to_dataloader(
        AudioDataset(val, processor, sr=sampling_rate, duration=duration), 
        batch_size=batch_size
        )
    
    return (train, test, val)
    