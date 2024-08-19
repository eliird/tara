import os
import pandas as pd
from typing import Tuple
from tara.constants import DATA_PATH, EMOTION2LABEL, LABEL2EMOTION
import torch
from torch.utils.data import TensorDataset, RandomSampler
from transformers.models.distilbert.tokenization_distilbert_fast import DistilBertTokenizerFast
from torch.utils.data import DataLoader


def get_max_len(tokenizer, dataset):
    max_len = 0

    text = dataset['Utterance'].to_list()
    # For every sentence...
    for sent in text:

        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)

        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))

    # print('Max sentence length: ', max_len)
    
    return max_len


def encode_dataset(dataset, tokenizer, max_len, return_tensordataset=True):
    input_ids = []
    attention_masks = []
    
    text = dataset['Utterance'].to_list()
    labels = dataset['labels'].to_list()
    
    for sent in text:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens = True,
            max_length  = max_len,
            pad_to_max_length = True,
            return_attention_mask = True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    # Print sentence 0, now as a list of IDs.
    # print('Original: ', text[0])
    # print('Token IDs:', input_ids[0])
    if return_tensordataset:
        data = TensorDataset(input_ids, attention_masks, labels)
    else:
        data = {
            'input_ids': input_ids,
            'attention_masks': attention_masks,
            'labels': labels
        }
    return data


def encode_emotion(label: str) -> int:
    return EMOTION2LABEL[label]


def get_filename(row: pd.Series, extension: bool=True) -> str:
    dia_id = str(row['Dialogue_ID'])
    utt_id = str(row['Utterance_ID'])
    f_name = f'dia{dia_id}_utt{utt_id}'
    if extension:
        f_name += '.mp4'
    return f_name


def get_filepath(row: pd.Series, base_path: str, mode, extension: bool=True) -> str:
    f_name = get_filename(row, extension)
    return os.path.join(base_path, mode, f_name)


def preprocess_df(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    df['filename'] = df.apply(lambda row: get_filepath(row, DATA_PATH, mode, True), axis = 1)
    df['labels'] = df['Emotion'].apply(lambda emotion: encode_emotion(emotion))
    return df


def load_file(path: str, mode: str) -> pd.DataFrame:
    assert mode in ['train', 'test', 'val']
    if "val" in mode:
        mode = "dev"
    df = pd.read_csv(path)
    df = preprocess_df(df, mode)
    return df


def load_data(base_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test = load_file(os.path.join(base_path, 'test_updated.csv'), mode='test')
    val = load_file(os.path.join(base_path, 'val_updated.csv'), mode='val')
    train = load_file(os.path.join(base_path, 'train_updated.csv'), mode='train')
    return (train, test, val)


def load_data_encoded(base_path: str, tokenizer: DistilBertTokenizerFast, return_tensordataset=True) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    train, test, val = load_data(base_path)
    max_len = get_max_len(tokenizer, train)

    train_dataset = encode_dataset(train, tokenizer, max_len, return_tensordataset)
    test_dataset = encode_dataset(test, tokenizer, max_len, return_tensordataset)
    val_dataset = encode_dataset(val, tokenizer, max_len, return_tensordataset)
    
    return (train_dataset, test_dataset, val_dataset)


def tensor_data_to_dataloader(dataset: TensorDataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size = batch_size,
        sampler=RandomSampler(dataset)
    )

def load_dataloader(base_path: str, tokenzier: DistilBertTokenizerFast, batch_size: int=32):
    train, test, val = load_data_encoded(base_path, tokenzier)
    train = tensor_data_to_dataloader(train, batch_size)
    test = tensor_data_to_dataloader(test, batch_size)
    val = tensor_data_to_dataloader(val, batch_size)
    return (train, test, val)