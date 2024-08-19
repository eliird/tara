from tara.data import load_data
from torch.utils.data import Dataset, DataLoader
import librosa as lb
from typing import Any
import torch
from .processing import tensor_data_to_dataloader, load_data_encoded
import numpy as np
import warnings



class TextAudioDataset(Dataset):
    def __init__(self, 
                 df, 
                 text_input_ids,
                 text_attention_masks,
                 audio_processor, 
                 sr=16_000, duration: float=3):
        
        self.df = df
        self.text_input_ids = text_input_ids
        self.text_attention_masks = text_attention_masks
        
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
        audio_input = self.processor(
            audio_input, sampling_rate=sr, return_tensors='pt')\
            .input_values.squeeze()
        
        text_input = self.text_input_ids[index]
        text_attention = self.text_attention_masks[index]
        
        labels = torch.tensor(self.df.iloc[index]['labels'])
        
        return (audio_input, text_input, text_attention, labels)
    
    def __len__(self):
        return len(self.df)

    def pad_or_truncate(self, arr):
        if len(arr) > self.samples:
            arr = arr[:self.samples]
        else:
            arr = np.pad(arr, (0, self.samples - len(arr)), 'constant', constant_values=(0))
        return arr
    

def load_text_audio_dataloader(base_path: str, processor, tokenizer, batch_size=16):
    
    train_df, test_df, val_df = load_data(base_path)
    train, test, val = load_data_encoded(
        base_path=base_path,
        tokenizer=tokenizer,
        return_tensordataset=False
    )
 
    train = TextAudioDataset(
        train_df,
        text_input_ids=train['input_ids'],
        text_attention_masks=train['attention_masks'],
        audio_processor=processor
    )
    
    
    test = TextAudioDataset(
        test_df,
        text_input_ids=test['input_ids'],
        text_attention_masks=test['attention_masks'],
        audio_processor=processor
    )
    
    val = TextAudioDataset(
        val_df,
        text_input_ids=val['input_ids'],
        text_attention_masks=val['attention_masks'],
        audio_processor=processor
    )
    
    train = tensor_data_to_dataloader(train, batch_size=batch_size)
    test = tensor_data_to_dataloader(test, batch_size=batch_size)
    val = tensor_data_to_dataloader(val, batch_size=batch_size)
    
    return (train ,test, val)
    
    
    