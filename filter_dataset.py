from tara.data import load_data
import pandas as pd
import librosa as lb
from tqdm import tqdm
import os
import warnings

warnings.simplefilter("ignore")

tqdm.pandas()

def can_open_file(filename: str):
    try:
        _, _ = lb.load(filename, sr=16000)
        return True
    except:
        print("Could not load file: ", filename)
        return False

def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['filename'].progress_apply(lambda x: can_open_file(x))]
    return df    

base_path = './data'


train, test, val = load_data(base_path)

train = filter_df(train)
test = filter_df(test)
val = filter_df(val)

train.to_csv('train_updated.csv')
test.to_csv('test_updated.csv')
val.to_csv('val_updated.csv')

